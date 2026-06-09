"""
Smoke tests for the CSAI415 Paper RAG API.

Run with:
    pytest tests/ -v

Server must be running at http://localhost:8000 (start with `uvicorn app.main:app`).
"""
import httpx
import pytest

BASE = 'http://localhost:8000'


def test_root():
    r = httpx.get(f'{BASE}/')
    assert r.status_code == 200
    assert r.json()['service'] == 'CSAI415 Paper RAG'


def test_health():
    r = httpx.get(f'{BASE}/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert body['chunks_indexed'] > 0


def test_stats():
    r = httpx.get(f'{BASE}/stats')
    assert r.status_code == 200
    body = r.json()
    assert body['papers'] > 0
    assert body['chunks'] > 0
    assert body['vectors'] > 0


def test_search_hybrid_default():
    r = httpx.get(f'{BASE}/search', params={'q': 'retrieval augmented generation'})
    assert r.status_code == 200
    body = r.json()
    assert body['mode'] == 'hybrid'
    assert body['rerank'] is True
    assert len(body['results']) > 0
    assert body['latency_ms'] > 0


def test_search_all_modes():
    for mode in ['dense', 'sparse', 'hybrid']:
        r = httpx.get(f'{BASE}/search', params={'q': 'vector database', 'mode': mode, 'top_k': 5})
        assert r.status_code == 200, f'mode={mode} failed'
        body = r.json()
        assert body['mode'] == mode
        assert len(body['results']) <= 5


def test_search_validates_top_k():
    r = httpx.get(f'{BASE}/search', params={'q': 'test', 'top_k': 999})
    assert r.status_code == 422  # validation error


def test_search_validates_empty_query():
    r = httpx.get(f'{BASE}/search', params={'q': ''})
    assert r.status_code == 422


def test_documents_pagination():
    r = httpx.get(f'{BASE}/documents', params={'skip': 0, 'limit': 5})
    assert r.status_code == 200
    body = r.json()
    assert body['total'] > 0
    assert len(body['documents']) <= 5


def test_document_not_found():
    r = httpx.get(f'{BASE}/document/nonexistent_id_xxx')
    assert r.status_code == 404


def test_feedback_stored():
    r = httpx.post(f'{BASE}/feedback', json={
        'query': 'test query',
        'doc_id': 'test_doc',
        'relevant': True
    })
    assert r.status_code == 200
    assert r.json()['stored'] is True


def test_graph_topics():
    r = httpx.get(f'{BASE}/graph/topics')
    # 200 if Neo4j connected, 503 if not — both acceptable
    assert r.status_code in (200, 503)
