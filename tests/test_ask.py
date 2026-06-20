"""
Smoke tests for POST /ask (D3 GraphRAG executor endpoint).

Run with:
    pytest tests/test_ask.py -v

Server must be running at http://localhost:8000 (start with `uvicorn app.main:app`).
"""
import httpx

BASE = 'http://localhost:8000'
TIMEOUT = 30.0  # rerank pipeline can take several seconds on CPU


def test_ask_default_hybrid():
    r = httpx.post(f'{BASE}/ask', json={'query': 'retrieval augmented generation'}, timeout=TIMEOUT)
    assert r.status_code == 200
    body = r.json()
    assert body['mode'] == 'hybrid'
    assert body['answer']
    assert body['latency_ms'] > 0
    assert body['steps'], 'steps[] must expose the agent stage trace'


def test_ask_all_modes_return_citations_with_pages():
    for mode in ['vector_only', 'graph_guided', 'hybrid']:
        r = httpx.post(f'{BASE}/ask', json={'query': 'vector database indexing', 'mode': mode, 'top_k': 3}, timeout=TIMEOUT)
        assert r.status_code == 200, f'mode={mode} failed'
        body = r.json()
        assert body['mode'] == mode
        for c in body['citations']:
            assert c['doc_id']
            assert c['chunk_id']
            assert c['pages'] is not None


def test_ask_validates_empty_query():
    r = httpx.post(f'{BASE}/ask', json={'query': ''}, timeout=TIMEOUT)
    assert r.status_code == 422


def test_ask_validates_top_k():
    r = httpx.post(f'{BASE}/ask', json={'query': 'test', 'top_k': 999}, timeout=TIMEOUT)
    assert r.status_code == 422


def test_ask_validates_mode():
    r = httpx.post(f'{BASE}/ask', json={'query': 'test', 'mode': 'not_a_real_mode'}, timeout=TIMEOUT)
    assert r.status_code == 422
