"""
D3 evaluator — runs every eval/gold_qa.json question through POST /ask for
each ablation mode, computes:
  - faithfulness: fraction of answer sentences entailed by the retrieved
    context, via cross-encoder/nli-deberta-v3-small (proxy for RAGAS)
  - answer-relevance: cosine(answer embedding, question embedding), via
    bge-small (same embedder the API uses for retrieval)
  - IR metrics: Recall@5, MRR, nDCG@5 (notebook-09 convention: gold doc_id
    must appear in the ranked retrieved doc_ids)
  - p50/p95 latency, measured from /ask's own reported latency_ms

Writes results/d3_eval.json + results/d3_eval.png.

IMPORTANT — read before citing these numbers: GraphRAGExecutor.answer()
(app/graphrag.py) is currently a deterministic EXTRACTIVE stand-in, not the
brief's Qwen2.5-1.5B-Instruct generator (this machine's torch build has no
CUDA support yet). Because the "answer" is literally an excerpt of the
retrieved context, faithfulness will trivially run high — this validates the
entailment-proxy PIPELINE end-to-end, it does not yet measure real generation
quality. Re-run this script once the generator is wired in; until then, these
numbers belong in the report as a pipeline sanity-check, not the final D3
headline metric.

Usage (server must be running: uvicorn app.main:app --port 8000):
    python scripts/evaluate_d3.py
"""
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / 'eval' / 'gold_qa.json'
OUT_JSON = ROOT / 'results' / 'd3_eval.json'
OUT_PNG = ROOT / 'results' / 'd3_eval.png'
BASE = 'http://localhost:8000'
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')

MODES = ['vector_only', 'graph_guided', 'hybrid']
TOP_K = 5  # matches "Recall@5" directly

EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
NLI_MODEL = 'cross-encoder/nli-deberta-v3-small'
ENTAILMENT_LABEL_IDX = 1  # verified empirically — see module history / git log
ENTAILMENT_THRESHOLD = 0.5


def recall_at_k(retrieved_doc_ids: list, relevant_doc: str, k: int) -> float:
    return 1.0 if relevant_doc in retrieved_doc_ids[:k] else 0.0


def mrr(retrieved_doc_ids: list, relevant_doc: str) -> float:
    for i, d in enumerate(retrieved_doc_ids, 1):
        if d == relevant_doc:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list, relevant_doc: str, k: int) -> float:
    for i, d in enumerate(retrieved_doc_ids[:k], 1):
        if d == relevant_doc:
            return 1.0 / math.log2(i + 1)
    return 0.0


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def faithfulness(nli: CrossEncoder, answer_text: str, context_texts: list[str]) -> float:
    sentences = split_sentences(answer_text)
    if not sentences or not context_texts:
        return 0.0
    entailed = 0
    for sent in sentences:
        pairs = [(ctx[:512], sent) for ctx in context_texts]
        logits = nli.predict(pairs)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        if probs[:, ENTAILMENT_LABEL_IDX].max() > ENTAILMENT_THRESHOLD:
            entailed += 1
    return entailed / len(sentences)


def answer_relevance(embedder: SentenceTransformer, question: str, answer_text: str) -> float:
    if not answer_text.strip():
        return 0.0
    qv = embedder.encode(question, normalize_embeddings=True)
    av = embedder.encode(answer_text, normalize_embeddings=True)
    return float(np.dot(qv, av))


def main() -> int:
    gold = json.loads(GOLD_PATH.read_text(encoding='utf-8'))
    print(f'Loaded {len(gold)} gold items from {GOLD_PATH}')

    print('Loading embedder + NLI cross-encoder (own process, separate from the API)...')
    embedder = SentenceTransformer(EMBED_MODEL)
    nli = CrossEncoder(NLI_MODEL)
    mongo = MongoClient(MONGO_URI).csai415_rag

    per_mode: dict = {mode: [] for mode in MODES}

    with httpx.Client(timeout=60.0) as client:
        for mode in MODES:
            print(f'--- mode={mode} ---')
            for item in gold:
                t0 = time.time()
                r = client.post(
                    f'{BASE}/ask',
                    json={'query': item['question'], 'mode': mode, 'top_k': TOP_K},
                )
                r.raise_for_status()
                body = r.json()
                wall_ms = (time.time() - t0) * 1000

                retrieved_doc_ids, seen = [], set()
                context_texts = []
                for c in body['citations']:
                    if c['doc_id'] not in seen:
                        seen.add(c['doc_id'])
                        retrieved_doc_ids.append(c['doc_id'])
                    chunk = mongo.chunks.find_one({'chunk_id': c['chunk_id']})
                    if chunk:
                        context_texts.append(chunk['text'])

                gold_doc_id = item['gold_doc_id']
                per_mode[mode].append({
                    'question'       : item['question'],
                    'gold_doc_id'    : gold_doc_id,
                    'retrieved_doc_ids': retrieved_doc_ids,
                    'recall@5'       : recall_at_k(retrieved_doc_ids, gold_doc_id, 5),
                    'mrr'            : mrr(retrieved_doc_ids, gold_doc_id),
                    'ndcg@5'         : ndcg_at_k(retrieved_doc_ids, gold_doc_id, 5),
                    'faithfulness'   : faithfulness(nli, body['answer'], context_texts),
                    'answer_relevance': answer_relevance(embedder, item['question'], body['answer']),
                    'latency_ms'     : body['latency_ms'],
                    'wall_latency_ms': round(wall_ms, 1),
                })
                print(f"  q={item['question'][:60]!r:62} "
                      f"R@5={per_mode[mode][-1]['recall@5']:.0f} "
                      f"faith={per_mode[mode][-1]['faithfulness']:.2f} "
                      f"rel={per_mode[mode][-1]['answer_relevance']:.2f} "
                      f"lat={body['latency_ms']:.0f}ms")

    summary = {}
    for mode in MODES:
        rows = per_mode[mode]
        latencies = [row['latency_ms'] for row in rows]
        summary[mode] = {
            'recall@5'        : float(np.mean([r['recall@5'] for r in rows])),
            'mrr'             : float(np.mean([r['mrr'] for r in rows])),
            'ndcg@5'          : float(np.mean([r['ndcg@5'] for r in rows])),
            'faithfulness'    : float(np.mean([r['faithfulness'] for r in rows])),
            'answer_relevance': float(np.mean([r['answer_relevance'] for r in rows])),
            'p50_ms'          : float(np.percentile(latencies, 50)),
            'p95_ms'          : float(np.percentile(latencies, 95)),
            'mean_ms'         : float(np.mean(latencies)),
        }

    output = {
        'timestamp'  : datetime.now(timezone.utc).isoformat(),
        'gold_set_size': len(gold),
        'top_k'      : TOP_K,
        'embedding_model': EMBED_MODEL,
        'nli_model'  : NLI_MODEL,
        'generator'  : 'stub (extractive) — see app/graphrag.py; not yet the Qwen2.5-1.5B-Instruct brief decision',
        'summary'    : summary,
        'per_query'  : per_mode,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nWrote {OUT_JSON}')

    _plot(summary)
    print(f'Wrote {OUT_PNG}')
    return 0


def _plot(summary: dict) -> None:
    metrics = ['recall@5', 'mrr', 'ndcg@5', 'faithfulness', 'answer_relevance']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(metrics))
    width = 0.25
    for i, mode in enumerate(MODES):
        values = [summary[mode][m] for m in metrics]
        axes[0].bar(x + i * width, values, width, label=mode)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(metrics, rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('D3 quality metrics by mode')
    axes[0].legend()

    p95s = [summary[mode]['p95_ms'] for mode in MODES]
    axes[1].bar(MODES, p95s, color=['#4c72b0', '#dd8452', '#55a868'])
    axes[1].set_title('p95 latency by mode (ms)')
    axes[1].set_ylabel('ms')

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)


if __name__ == '__main__':
    raise SystemExit(main())
