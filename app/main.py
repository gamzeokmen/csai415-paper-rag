"""
CSAI415 D2 — Paper RAG API (v2.0)

Hybrid retrieval (BM25 + dense + RRF + cross-encoder rerank) over an arXiv RAG
corpus. Uses motor for async MongoDB, AsyncQdrantClient for async vector search,
and AsyncGraphDatabase for async Neo4j queries.
"""
import os
import time
import logging
from functools import lru_cache
from typing import Literal, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

load_dotenv()

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
)
log = logging.getLogger('csai415_rag')

# ── config ───────────────────────────────────────────────────────────────────
MONGO_URI       = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
QDRANT_HOST     = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT     = int(os.getenv('QDRANT_PORT', 6333))
QDRANT_COLL     = 'csai415_papers'
NEO4J_URI       = os.getenv('NEO4J_URI')
NEO4J_USER      = os.getenv('NEO4J_USERNAME') or os.getenv('NEO4J_USER')
NEO4J_PASSWORD  = os.getenv('NEO4J_PASSWORD')
EMBED_MODEL     = 'BAAI/bge-small-en-v1.5'
RERANK_MODEL    = 'BAAI/bge-reranker-base'
RRF_K           = 60

state = {}


# ── lifespan: load models + connect to stores at startup ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info('startup: loading embedder + reranker (first run downloads ~480 MB)')
    state['embedder'] = SentenceTransformer(EMBED_MODEL)
    state['reranker'] = CrossEncoder(RERANK_MODEL, max_length=512)

    log.info('startup: connecting to MongoDB at %s', MONGO_URI)
    state['mongo'] = AsyncIOMotorClient(MONGO_URI)
    state['db']    = state['mongo'].csai415_rag

    log.info('startup: connecting to Qdrant at %s:%d', QDRANT_HOST, QDRANT_PORT)
    state['qdrant'] = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
        log.info('startup: connecting to Neo4j Aura')
        state['neo4j'] = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    else:
        log.warning('Neo4j credentials not set — /graph/* endpoints will return 503')
        state['neo4j'] = None

    log.info('startup: building in-memory BM25 index from MongoDB chunks')
    chunks = []
    cursor = state['db'].chunks.find({}, {'doc_id': 1, 'chunk_idx': 1, 'text': 1})
    async for c in cursor:
        chunks.append(c)
    state['chunks']     = chunks
    state['chunk_lookup'] = {(c['doc_id'], c.get('chunk_idx', i)): c['text']
                              for i, c in enumerate(chunks)}
    state['bm25']       = BM25Okapi([c['text'].lower().split() for c in chunks])
    log.info('startup: BM25 index ready (%d chunks)', len(chunks))

    yield  # ── app runs ────────────────────────────────────────────────────

    log.info('shutdown: closing connections')
    state['mongo'].close()
    await state['qdrant'].close()
    if state['neo4j']:
        await state['neo4j'].close()


app = FastAPI(
    title       = 'CSAI415 Paper RAG API',
    description = 'Hybrid retrieval (BM25 + dense + RRF + cross-encoder rerank) '
                  'over an arXiv corpus.',
    version     = '2.0.0',
    lifespan    = lifespan,
)


# ── pydantic models ─────────────────────────────────────────────────────────
class ChunkResult(BaseModel):
    doc_id      : str
    chunk_idx   : int
    text        : str
    score       : float
    rerank_score: Optional[float] = None
    title       : Optional[str] = None
    authors     : Optional[list[str]] = None
    year        : Optional[int] = None
    arxiv_id    : Optional[str] = None


class SearchResponse(BaseModel):
    query     : str
    mode      : str
    top_k     : int
    rerank    : bool
    latency_ms: float
    results   : list[ChunkResult]


class FeedbackRequest(BaseModel):
    query   : str    = Field(..., min_length=1)
    doc_id  : str    = Field(...)
    relevant: bool   = Field(...)


class StatsResponse(BaseModel):
    papers : int
    chunks : int
    vectors: int


# ── retrieval helpers ───────────────────────────────────────────────────────
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

@lru_cache(maxsize=512)
def _embed_cached(query: str) -> tuple:
    """LRU-cached query embedding with BGE query instruction prefix."""
    prefixed = BGE_QUERY_PREFIX + query
    return tuple(state['embedder'].encode(prefixed, normalize_embeddings=True).tolist())


def _rrf_merge(rankings: list, k: int = RRF_K, top_k: int = 50) -> list:
    """Reciprocal Rank Fusion."""
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return [k for k, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


async def _dense_search(query: str, limit: int = 50) -> list:
    qv = list(_embed_cached(query))
    response = await state['qdrant'].query_points(
        collection_name = QDRANT_COLL,
        query           = qv,
        limit           = limit,
    )
    return [(h.payload['doc_id'], h.payload.get('chunk_idx', 0), float(h.score)) for h in response.points]


def _sparse_search(query: str, limit: int = 50) -> list:
    bm25 = state['bm25']
    chunks = state['chunks']
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:limit]
    return [(chunks[i]['doc_id'], chunks[i].get('chunk_idx', i), float(scores[i]))
            for i in top_idx]


def _rerank(query: str, candidates: list, top_k: int) -> list:
    """Cross-encoder rerank. candidates: list of (doc_id, chunk_idx, score)."""
    lookup = state['chunk_lookup']
    pairs, kept = [], []
    for doc_id, chunk_idx, score in candidates:
        text = lookup.get((doc_id, chunk_idx))
        if text:
            pairs.append([query, text[:512]])
            kept.append((doc_id, chunk_idx, score))
    if not pairs:
        return []
    scores = state['reranker'].predict(pairs)
    paired = sorted(
        zip(kept, scores),
        key=lambda x: -x[1]
    )[:top_k]
    return [(c[0], c[1], float(s)) for c, s in paired]


# ── endpoints ───────────────────────────────────────────────────────────────
@app.get('/', tags=['health'])
async def root():
    return {'service': 'CSAI415 Paper RAG', 'version': '2.0.0', 'docs': '/docs'}


@app.get('/health', tags=['health'])
async def health():
    return {'status': 'ok', 'chunks_indexed': len(state.get('chunks', []))}


@app.get('/search', response_model=SearchResponse, tags=['retrieval'])
async def search(
    q     : str = Query(..., min_length=1, description='User query'),
    mode  : Literal['dense', 'sparse', 'hybrid'] = Query('hybrid'),
    top_k : int = Query(5, ge=1, le=50),
    rerank: bool = Query(True, description='Apply cross-encoder rerank to top-20'),
):
    t0 = time.time()

    if mode == 'dense':
        candidates = await _dense_search(q, limit=50)
    elif mode == 'sparse':
        candidates = _sparse_search(q, limit=50)
    else:
        d = await _dense_search(q, limit=50)
        s = _sparse_search(q, limit=50)
        # build keyed ranking for RRF
        d_keys = [(x[0], x[1]) for x in d]
        s_keys = [(x[0], x[1]) for x in s]
        merged_keys = _rrf_merge([d_keys, s_keys], top_k=50)
        score_map = {(x[0], x[1]): x[2] for x in d}
        candidates = [(k[0], k[1], score_map.get(k, 0.0)) for k in merged_keys]

    if rerank:
        candidates = _rerank(q, candidates[:20], top_k=top_k)
    else:
        candidates = candidates[:top_k]

    # hydrate with MongoDB metadata
    results = []
    seen_docs = set()
    for doc_id, chunk_idx, score in candidates:
        text = state['chunk_lookup'].get((doc_id, chunk_idx), '')
        doc = await state['db'].documents.find_one({'_id': doc_id})
        results.append(ChunkResult(
            doc_id       = str(doc_id),
            chunk_idx    = int(chunk_idx),
            text         = text[:400] + ('...' if len(text) > 400 else ''),
            score        = score if not rerank else 0.0,
            rerank_score = score if rerank else None,
            title        = doc.get('title') if doc else None,
            authors      = doc.get('authors') if doc else None,
            year         = doc.get('year') if doc else None,
            arxiv_id     = doc.get('arxiv_id') if doc else None,
        ))

    latency_ms = round((time.time() - t0) * 1000, 1)
    log.info('search mode=%s top_k=%d rerank=%s lat=%.1fms q=%r',
             mode, top_k, rerank, latency_ms, q[:50])

    return SearchResponse(
        query=q, mode=mode, top_k=top_k, rerank=rerank,
        latency_ms=latency_ms, results=results
    )


@app.get('/documents', tags=['corpus'])
async def list_documents(skip: int = 0, limit: int = 20):
    cursor = state['db'].documents.find(
        {}, {'_id': 1, 'title': 1, 'authors': 1, 'year': 1, 'arxiv_id': 1}
    ).skip(skip).limit(limit)
    docs = []
    async for d in cursor:
        d['_id'] = str(d['_id'])
        docs.append(d)
    total = await state['db'].documents.count_documents({})
    return {'total': total, 'skip': skip, 'limit': limit, 'documents': docs}


@app.get('/document/{doc_id}', tags=['corpus'])
async def get_document(doc_id: str):
    doc = await state['db'].documents.find_one({'_id': doc_id})
    if not doc:
        raise HTTPException(404, f'document {doc_id} not found')
    doc['_id'] = str(doc['_id'])
    chunk_count = await state['db'].chunks.count_documents({'doc_id': doc_id})
    doc['chunk_count'] = chunk_count
    return doc


@app.post('/feedback', tags=['learning'])
async def feedback(req: FeedbackRequest):
    """Store a user relevance signal. Will feed into D3 online learning."""
    await state['db'].feedback.insert_one({
        'query'    : req.query,
        'doc_id'   : req.doc_id,
        'relevant' : req.relevant,
        'timestamp': time.time(),
    })
    return {'stored': True}


@app.get('/stats', response_model=StatsResponse, tags=['health'])
async def stats():
    papers = await state['db'].documents.count_documents({})
    chunks = await state['db'].chunks.count_documents({})
    coll_info = await state['qdrant'].get_collection(QDRANT_COLL)
    return StatsResponse(papers=papers, chunks=chunks, vectors=coll_info.points_count)


async def _cypher(query: str, **params) -> list[dict]:
    """Run a read-only Cypher query and return rows as dicts.

    Translates an unconfigured *or unreachable* Neo4j (e.g. a paused/expired
    Aura instance) into a graceful HTTP 503 instead of an unhandled 500.
    """
    if not state['neo4j']:
        raise HTTPException(503, 'Neo4j not configured')
    try:
        async with state['neo4j'].session() as session:
            result = await session.run(query, **params)
            return [dict(r) async for r in result]
    except Exception as exc:  # noqa: BLE001 — DNS / ServiceUnavailable / driver errors
        log.warning('Neo4j unavailable: %s', exc)
        raise HTTPException(503, 'Neo4j unavailable') from exc


@app.get('/graph/topics', tags=['graph'])
async def graph_topics():
    rows = await _cypher("""
        MATCH (t:Topic)<-[:ABOUT]-(p:Paper)
        RETURN t.name AS topic, count(p) AS papers
        ORDER BY papers DESC
    """)
    return {'topics': rows}


@app.get('/graph/authors', tags=['graph'])
async def graph_authors(limit: int = 10):
    rows = await _cypher("""
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        RETURN a.name AS author, count(p) AS papers
        ORDER BY papers DESC LIMIT $limit
    """, limit=limit)
    return {'authors': rows}


@app.get('/document/{doc_id}/citations', tags=['graph'])
async def get_citations(doc_id: str, min_confidence: float = Query(0.0, ge=0.0, le=1.0)):
    """Outgoing citations — papers that {doc_id} cites. Supports min_confidence
    filtering (synthetic edges default to confidence 1.0 when unset)."""
    edges = await _cypher("""
        MATCH (p:Paper {doc_id: $doc_id})-[r:CITES]->(c:Paper)
        WHERE coalesce(r.confidence, 1.0) >= $min_conf
        RETURN c.doc_id AS doc_id, c.title AS title,
               coalesce(r.confidence, 1.0) AS confidence,
               coalesce(r.synthetic, false) AS synthetic
        ORDER BY confidence DESC
    """, doc_id=doc_id, min_conf=min_confidence)
    return {'doc_id': doc_id, 'min_confidence': min_confidence, 'citations': edges}


@app.get('/document/{doc_id}/cited_by', tags=['graph'])
async def get_cited_by(doc_id: str):
    """Incoming citations — papers that cite {doc_id}."""
    edges = await _cypher("""
        MATCH (p:Paper {doc_id: $doc_id})<-[r:CITES]-(c:Paper)
        RETURN c.doc_id AS doc_id, c.title AS title,
               coalesce(r.synthetic, false) AS synthetic
    """, doc_id=doc_id)
    return {'doc_id': doc_id, 'cited_by': edges}


@app.get('/graph/cites', tags=['graph'])
async def graph_most_cited(limit: int = 10):
    """Most-cited papers in the corpus (via CITES edges)."""
    rows = await _cypher("""
        MATCH (p:Paper)<-[:CITES]-(c:Paper)
        RETURN p.title AS title, p.arxiv_id AS arxiv_id, count(c) AS citations
        ORDER BY citations DESC LIMIT $limit
    """, limit=limit)
    return {'most_cited': rows}
