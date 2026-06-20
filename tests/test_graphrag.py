"""
Tests for the D3 GraphRAG executor (app/graphrag.py).

Drives app.main's lifespan directly inside the test's own event loop (rather
than via Starlette's TestClient, which runs lifespan in a separate thread/loop
— motor's MongoDB client binds strictly to its creation loop, so cross-loop
use raises RuntimeError). This loads its own embedder/reranker/BM25 index and
connects to Mongo/Qdrant/Neo4j independently of any already-running uvicorn
process.
"""
import pytest

from app.main import app as fastapi_app, lifespan
from app.retrieval import state
from app.graphrag import GraphRAGExecutor


@pytest.mark.anyio
async def test_graphrag_executor_end_to_end():
    async with lifespan(fastapi_app):
        assert state.get('neo4j') is not None, 'Neo4j must be connected for this test'
        executor = GraphRAGExecutor(neo4j_driver=state['neo4j'], docs_col=state['db'].documents)

        # select_subgraph: provenance present, fan-out capped
        subgraph = await executor.select_subgraph('retrieval augmented generation for question answering')
        assert isinstance(subgraph, list)
        assert len(subgraph) <= 25, 'fan-out must be capped'
        for entry in subgraph:
            assert entry['doc_id']
            assert entry['provenance'], 'every reached paper must carry provenance'
            assert set(entry['provenance']) <= {'cites', 'shared_author', 'shared_topic'}

        # gotcha #3 regression: cs.IR covers 138/144 papers — shared_topic
        # expansion must stay capped, not dominate the subgraph
        subgraph2 = await executor.select_subgraph('information retrieval systems')
        topic_only = [e for e in subgraph2 if e['provenance'] == ['shared_topic']]
        assert len(topic_only) < 50

        # expand_to_chunks: scored candidates restricted to the given doc_ids
        doc_ids = [s['doc_id'] for s in subgraph]
        chunks = await executor.expand_to_chunks(doc_ids, 'vector database indexing')
        for doc_id, chunk_id, score in chunks:
            assert doc_id in doc_ids
            assert chunk_id
            assert isinstance(score, float)

        # run(): all three ablation modes return citations with page numbers
        for mode in ['vector_only', 'graph_guided', 'hybrid']:
            result = await executor.run('hybrid retrieval fusion strategies', mode=mode, top_k=3)
            assert result['mode'] == mode
            assert result['steps'], 'run() must report its stage trace'
            assert result['latency_ms'] > 0
            assert 'citations' in result
            for c in result['citations']:
                assert c['doc_id']
                assert c['chunk_id']
                assert c['pages'] is not None

        # must never crash if Neo4j is unavailable — degrade to vector-only
        degraded = GraphRAGExecutor(neo4j_driver=None, docs_col=state['db'].documents)
        assert await degraded.select_subgraph('test query') == []
        degraded_result = await degraded.run('test query', mode='graph_guided', top_k=3)
        assert degraded_result['mode'] == 'graph_guided'
        assert isinstance(degraded_result['citations'], list)
