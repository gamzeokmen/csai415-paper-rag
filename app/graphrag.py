"""
CSAI415 D3 — GraphRAG executor.

Four-stage pipeline: (1) select_subgraph — seed papers from vector top-k, then
multi-hop Cypher over CITES + shared-author + shared-topic; (2) expand_to_chunks
— fetch and score the subgraph papers' chunks so they're comparable to vector
hits; (3) blend — RRF-merge vector and graph candidates, optional rerank;
(4) answer — grounded citations (with page numbers) from the ranked chunks.

`mode` toggles which stages run (vector_only | graph_guided | hybrid), so the
same code path serves the D3 ablation (vector-only vs graph-guided vs hybrid).

Known limitation (documented, not hidden — see D3 brief gotcha #3): Topic
nodes are just the 5 arXiv categories and cs.IR alone covers 138/144 papers
(96%) of this corpus, so shared-topic overlap is a very weak/non-discriminating
signal at this corpus size. select_subgraph therefore treats CITES and
shared-author as the primary expansion paths and caps shared-topic's
contribution to TOPIC_FANOUT_CAP candidates per seed, so a single dominant
topic can't make "graph-guided" degenerate into "the whole corpus".

The `answer()` stage's LLM generation (Qwen2.5-1.5B-Instruct, 4-bit) is not
yet wired up — this machine's torch build has no CUDA support, so the
GPU-4-bit generator from the D3 brief needs a separate setup pass. answer()
currently returns a deterministic extractive stand-in plus full citations
(doc_id, chunk_id, title, pages) so the retrieval/citation mechanics can be
built and tested end-to-end first.
"""
import logging
import time
from typing import Literal, Optional

from app import retrieval as rt

log = logging.getLogger('csai415_rag.graphrag')

MAX_FANOUT        = 25   # total graph-reached papers per query, across all seeds
TOPIC_FANOUT_CAP  = 3    # shared-topic candidates per seed — see module docstring
AUTHOR_FANOUT_CAP = 10   # shared-author candidates per seed
CITES_FANOUT_CAP  = 10   # CITES-neighbor candidates per seed

_SUBGRAPH_CYPHER = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Paper {doc_id: seed_id})

OPTIONAL MATCH (seed)-[:CITES]-(cited:Paper)
WHERE cited.doc_id <> seed_id
WITH seed, seed_id, collect(DISTINCT cited.doc_id)[..$cites_cap] AS cites_ids

OPTIONAL MATCH (seed)<-[:WROTE]-(a:Author)-[:WROTE]->(co:Paper)
WHERE co.doc_id <> seed_id
WITH seed, seed_id, cites_ids, collect(DISTINCT co.doc_id)[..$author_cap] AS author_ids

OPTIONAL MATCH (seed)-[:ABOUT]->(t:Topic)<-[:ABOUT]-(sim:Paper)
WHERE sim.doc_id <> seed_id
WITH seed_id, cites_ids, author_ids, collect(DISTINCT sim.doc_id)[..$topic_cap] AS topic_ids

RETURN seed_id, cites_ids, author_ids, topic_ids
"""

_PROVENANCE_PRIORITY = {'cites': 0, 'shared_author': 1, 'shared_topic': 2}


class GraphRAGExecutor:
    def __init__(self, neo4j_driver, docs_col):
        self.neo4j = neo4j_driver
        self.docs_col = docs_col

    # ── stage 1 ──────────────────────────────────────────────────────────────
    async def select_subgraph(
        self,
        query: str,
        top_k_seed: int = 5,
        max_fanout: int = MAX_FANOUT,
        topic_cap: int = TOPIC_FANOUT_CAP,
        author_cap: int = AUTHOR_FANOUT_CAP,
        cites_cap: int = CITES_FANOUT_CAP,
    ) -> list[dict]:
        """Seed from vector top-k, expand via multi-hop Cypher. Returns each
        reached doc_id with provenance (which path(s) reached it, from which
        seed), capped to max_fanout total and prioritized CITES > shared_author
        > shared_topic when truncating."""
        if self.neo4j is None:
            log.info('select_subgraph: neo4j unavailable, returning empty subgraph')
            return []

        seed_hits = await rt.dense_search(query, limit=top_k_seed * 3)
        seed_ids: list[str] = []
        seen = set()
        for doc_id, _chunk_id, _score in seed_hits:
            if doc_id not in seen:
                seen.add(doc_id)
                seed_ids.append(doc_id)
            if len(seed_ids) >= top_k_seed:
                break
        if not seed_ids:
            return []

        try:
            async with self.neo4j.session() as session:
                result = await session.run(
                    _SUBGRAPH_CYPHER,
                    seed_ids=seed_ids, cites_cap=cites_cap,
                    author_cap=author_cap, topic_cap=topic_cap,
                )
                rows = [dict(r) async for r in result]
        except Exception as exc:  # noqa: BLE001 — degrade to vector-only, never crash
            log.warning('select_subgraph: Cypher failed, degrading: %s', exc)
            return []

        reached: dict[str, dict] = {}
        for row in rows:
            seed_id = row['seed_id']
            for path, doc_ids in (
                ('cites', row['cites_ids'] or []),
                ('shared_author', row['author_ids'] or []),
                ('shared_topic', row['topic_ids'] or []),
            ):
                for doc_id in doc_ids:
                    entry = reached.setdefault(
                        doc_id, {'doc_id': doc_id, 'provenance': set(), 'seed_ids': set()}
                    )
                    entry['provenance'].add(path)
                    entry['seed_ids'].add(seed_id)

        for sid in seed_ids:
            reached.pop(sid, None)  # don't re-surface the seeds as "expansion"

        ranked = sorted(
            reached.values(),
            key=lambda e: min(_PROVENANCE_PRIORITY[p] for p in e['provenance']),
        )[:max_fanout]

        return [
            {
                'doc_id': e['doc_id'],
                'provenance': sorted(e['provenance']),
                'seed_ids': sorted(e['seed_ids']),
            }
            for e in ranked
        ]

    # ── stage 2 ──────────────────────────────────────────────────────────────
    async def expand_to_chunks(self, doc_ids: list[str], query: str, limit: int = 50) -> list:
        """Fetch chunks for the subgraph papers and score them (dense + BM25,
        RRF-merged) so they're on the same comparable scale as vector hits."""
        if not doc_ids:
            return []
        dense_hits  = await rt.dense_search(query, limit=limit, doc_ids=doc_ids)
        sparse_hits = rt.sparse_search(query, limit=limit, doc_ids=doc_ids)
        d_keys = [x[1] for x in dense_hits]
        s_keys = [x[1] for x in sparse_hits]
        merged_keys = rt.rrf_merge([d_keys, s_keys], top_k=limit)
        doc_map   = {x[1]: x[0] for x in dense_hits} | {x[1]: x[0] for x in sparse_hits}
        score_map = {x[1]: x[2] for x in dense_hits} | {x[1]: x[2] for x in sparse_hits}
        return [(doc_map[key], key, score_map.get(key, 0.0)) for key in merged_keys]

    # ── stage 3 ──────────────────────────────────────────────────────────────
    def blend(self, vector_hits: list, graph_hits: list, top_k: int = 50) -> list:
        """RRF-merge vector-search candidates with graph-expansion candidates."""
        if not graph_hits:
            return vector_hits[:top_k]
        v_keys = [x[1] for x in vector_hits]
        g_keys = [x[1] for x in graph_hits]
        merged_keys = rt.rrf_merge([v_keys, g_keys], top_k=top_k)
        doc_map   = {x[1]: x[0] for x in vector_hits} | {x[1]: x[0] for x in graph_hits}
        score_map = {x[1]: x[2] for x in vector_hits} | {x[1]: x[2] for x in graph_hits}
        return [(doc_map[key], key, score_map.get(key, 0.0)) for key in merged_keys]

    # ── stage 4 ──────────────────────────────────────────────────────────────
    async def answer(self, query: str, ranked_chunks: list, top_n_context: int = 5) -> dict:
        """Grounded citations (with page numbers) from the ranked chunks.

        LLM generation is not wired up yet (see module docstring) — `answer`
        is a deterministic extractive stand-in so the citation/page-range
        mechanics are real and testable now, independent of the generator
        decision.
        """
        citations = []
        excerpts = []
        for doc_id, chunk_id, _score in ranked_chunks[:top_n_context]:
            entry = rt.state['chunk_lookup'].get(chunk_id)
            if not entry:
                continue
            doc = await self.docs_col.find_one({'doc_id': doc_id})
            citations.append({
                'doc_id'  : doc_id,
                'chunk_id': chunk_id,
                'title'   : doc.get('title') if doc else None,
                'pages'   : [entry.get('page_num')],
            })
            excerpts.append(entry['text'])

        answer_text = (
            '[D3 generator not yet wired — see app/graphrag.py module docstring] '
            + (excerpts[0][:300] if excerpts else 'No supporting context retrieved.')
        )
        return {'answer': answer_text, 'citations': citations, 'generator': 'stub'}

    # ── orchestration ────────────────────────────────────────────────────────
    async def run(
        self,
        query: str,
        mode: Literal['vector_only', 'graph_guided', 'hybrid'] = 'hybrid',
        top_k: int = 5,
        rerank: bool = True,
    ) -> dict:
        t0 = time.time()
        steps = []

        vector_hits = await rt.dense_search(query, limit=50)
        sparse_hits = rt.sparse_search(query, limit=50)
        v_keys = [x[1] for x in vector_hits]
        s_keys = [x[1] for x in sparse_hits]
        vector_merged_keys = rt.rrf_merge([v_keys, s_keys], top_k=50)
        doc_map   = {x[1]: x[0] for x in vector_hits} | {x[1]: x[0] for x in sparse_hits}
        score_map = {x[1]: x[2] for x in vector_hits}
        vector_candidates = [(doc_map[k], k, score_map.get(k, 0.0)) for k in vector_merged_keys]
        steps.append({'stage': 'vector_search', 'candidates': len(vector_candidates)})

        graph_candidates: list = []
        if mode in ('graph_guided', 'hybrid'):
            if self.neo4j is None:
                steps.append({'stage': 'select_subgraph', 'skipped': 'neo4j unavailable, degrading to vector-only'})
            else:
                subgraph = await self.select_subgraph(query)
                steps.append({
                    'stage': 'select_subgraph',
                    'papers_reached': len(subgraph),
                    'provenance_sample': subgraph[:5],
                })
                graph_doc_ids = [s['doc_id'] for s in subgraph]
                graph_candidates = await self.expand_to_chunks(graph_doc_ids, query)
                steps.append({'stage': 'expand_to_chunks', 'candidates': len(graph_candidates)})

        if mode == 'vector_only':
            merged = vector_candidates
        elif mode == 'graph_guided':
            merged = graph_candidates if graph_candidates else vector_candidates
        else:  # hybrid
            merged = self.blend(vector_candidates, graph_candidates)
        steps.append({'stage': 'blend', 'mode': mode, 'merged_candidates': len(merged)})

        if rerank:
            merged = rt.rerank(query, merged[:20], top_k=top_k)
            steps.append({'stage': 'rerank', 'final_count': len(merged)})
        else:
            merged = merged[:top_k]
            steps.append({'stage': 'truncate', 'final_count': len(merged)})

        result = await self.answer(query, merged)
        latency_ms = round((time.time() - t0) * 1000, 1)
        return {'query': query, 'mode': mode, 'steps': steps, 'latency_ms': latency_ms, **result}
