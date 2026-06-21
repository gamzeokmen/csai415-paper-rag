"""
Shared retrieval primitives — embedding cache, dense/sparse search, RRF merge,
cross-encoder rerank.

Extracted out of app/main.py so app/graphrag.py (the D3 GraphRAG executor) can
reuse the exact same retrieval logic without app.main and app.graphrag
importing each other.

`state` is the single shared dict populated by app.main's FastAPI lifespan
(embedder, reranker, qdrant client, bm25 index, chunks, chunk_lookup). Both
app.main and app.graphrag import this same dict instance.
"""
from functools import lru_cache

import numpy as np
from qdrant_client import models as qmodels

RRF_K = 60
QDRANT_COLL = 'csai415_papers'
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

state: dict = {}


@lru_cache(maxsize=512)
def embed_cached(query: str) -> tuple:
    """LRU-cached query embedding with BGE query instruction prefix."""
    prefixed = BGE_QUERY_PREFIX + query
    return tuple(state['embedder'].encode(prefixed, normalize_embeddings=True).tolist())


def rrf_merge(rankings: list, k: int = RRF_K, top_k: int = 50) -> list:
    """Reciprocal Rank Fusion over one or more rankings of the same key type."""
    scores: dict = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return [key for key, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


async def dense_search(query: str, limit: int = 50, doc_ids: list[str] | None = None) -> list:
    """Dense (Qdrant cosine) search. doc_ids, if given, restricts the search to
    chunks belonging to that set of papers — used by the GraphRAG executor to
    score chunks within a graph-selected subgraph."""
    qv = list(embed_cached(query))
    query_filter = None
    if doc_ids:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key='doc_id', match=qmodels.MatchAny(any=doc_ids))]
        )
    response = await state['qdrant'].query_points(
        collection_name = QDRANT_COLL,
        query           = qv,
        query_filter    = query_filter,
        limit           = limit,
    )
    return [(h.payload['doc_id'], h.payload['chunk_id'], float(h.score)) for h in response.points]


def sparse_search(query: str, limit: int = 50, doc_ids: list[str] | None = None) -> list:
    """BM25 search. doc_ids, if given, restricts ranking to chunks in that set."""
    bm25 = state['bm25']
    chunks = state['chunks']
    scores = bm25.get_scores(query.lower().split())
    if doc_ids:
        doc_id_set = set(doc_ids)
        idx = [i for i in range(len(chunks)) if chunks[i]['doc_id'] in doc_id_set]
        idx.sort(key=lambda i: -scores[i])
        top_idx = idx[:limit]
    else:
        top_idx = np.argsort(scores)[::-1][:limit]
    return [(chunks[i]['doc_id'], chunks[i]['chunk_id'], float(scores[i])) for i in top_idx]


def rerank(query: str, candidates: list, top_k: int) -> list:
    """Cross-encoder rerank. candidates: list of (doc_id, chunk_id, score)."""
    lookup = state['chunk_lookup']
    pairs, kept = [], []
    for doc_id, chunk_id, score in candidates:
        entry = lookup.get(chunk_id)
        if entry:
            pairs.append([query, entry['text'][:512]])
            kept.append((doc_id, chunk_id, score))
    if not pairs:
        return []
    scores = state['reranker'].predict(pairs)
    paired = sorted(zip(kept, scores), key=lambda x: -x[1])[:top_k]
    return [(c[0], c[1], float(s)) for c, s in paired]
