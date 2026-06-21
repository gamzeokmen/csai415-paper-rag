"""
D3 evaluator — runs every eval/gold_qa.json question through POST /ask for
each ablation mode, computes:
  - faithfulness: app.safety.filter_ungrounded's grounded_fraction — the
    same contradiction+similarity check that gates the live /ask pipeline
    (proxy for RAGAS), NOT a strict entailment-only score. A pure-entailment
    version was tried first and found empirically to score near-zero for
    any real, paraphrased LLM answer even when clearly grounded (NLI
    cross-encoders are trained for verbatim logical entailment, not
    paraphrase recognition) — see app/safety.py's filter_ungrounded
    docstring and D3_Report.md for the empirical comparison.
  - answer-relevance: cosine(answer embedding, question embedding), via
    bge-small (same embedder the API uses for retrieval)
  - IR metrics: Recall@5, MRR, nDCG@5 (notebook-09 convention: gold doc_id
    must appear in the ranked retrieved doc_ids)
  - p50/p95 latency, measured from /ask's own reported latency_ms

Writes results/d3_eval.json + results/d3_eval.png.

The generator behind GraphRAGExecutor.answer() (app/graphrag.py) is
app.generator.OllamaGenerator (qwen2.5:1.5b, local, CPU) — not the brief's
bitsandbytes-4-bit-on-GPU path (this machine has no CUDA), but a real LLM
generating real prose, not an extractive stand-in. If Ollama is unreachable,
answer() degrades to an extractive excerpt instead of crashing; check the
`generator` field on individual /ask responses if you need to confirm which
path produced a given answer.

Usage (server must be running: uvicorn app.main:app --port 8000):
    python scripts/evaluate_d3.py
"""
import json
import math
import os
import sys
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `from app import safety` resolves when run as a script
from app import safety  # noqa: E402

load_dotenv()
GOLD_PATH = ROOT / 'eval' / 'gold_qa.json'
OUT_JSON = ROOT / 'results' / 'd3_eval.json'
OUT_PNG = ROOT / 'results' / 'd3_eval.png'
BASE = 'http://localhost:8000'
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')

MODES = ['vector_only', 'graph_guided', 'hybrid']
TOP_K = 5  # matches "Recall@5" directly

EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
NLI_MODEL = 'cross-encoder/nli-deberta-v3-small'


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


def faithfulness(nli: CrossEncoder, embedder: SentenceTransformer, answer_text: str, context_texts: list[str]) -> float:
    """Reuses app.safety.filter_ungrounded's grounded_fraction — same
    contradiction+similarity check that gates the live /ask pipeline, so the
    reported metric and the actual safety mechanism can't silently diverge.
    See app/safety.py's filter_ungrounded docstring for why this isn't a
    strict entailment-only check (real LLM paraphrases score near-zero
    entailment against single-sentence premises even when well-grounded)."""
    return safety.filter_ungrounded(nli, embedder, answer_text, context_texts)['grounded_fraction']


def answer_relevance(embedder: SentenceTransformer, question: str, answer_text: str) -> float:
    if not answer_text.strip():
        return 0.0
    qv = embedder.encode(question, normalize_embeddings=True)
    av = embedder.encode(answer_text, normalize_embeddings=True)
    return float(np.dot(qv, av))


def run_evaluation(arms: list[tuple[str, str, bool]] | None = None) -> tuple[dict, dict]:
    """Runs every gold question through POST /ask for each (label, mode,
    rerank) arm and returns (summary, per_arm). Shared by this script's CLI
    and scripts/ablation_d3.py so both report the same real, freshly-computed
    numbers rather than duplicating the evaluation logic.

    arms: list of (label, mode, rerank) — label is what shows up in the
    output (lets ablation_d3.py distinguish e.g. 'hybrid' from
    'hybrid_no_rerank', both using mode='hybrid'). Defaults to MODES, all
    with rerank=True.
    """
    if arms is None:
        arms = [(mode, mode, True) for mode in MODES]

    gold = json.loads(GOLD_PATH.read_text(encoding='utf-8'))
    print(f'Loaded {len(gold)} gold items from {GOLD_PATH}')

    print('Loading embedder + NLI cross-encoder (own process, separate from the API)...')
    embedder = SentenceTransformer(EMBED_MODEL)
    nli = CrossEncoder(NLI_MODEL)
    mongo = MongoClient(MONGO_URI).csai415_rag

    per_arm: dict = {label: [] for label, _, _ in arms}

    with httpx.Client(timeout=60.0) as client:
        for label, mode, rerank in arms:
            print(f'--- arm={label} (mode={mode}, rerank={rerank}) ---')
            for item in gold:
                t0 = time.time()
                r = client.post(
                    f'{BASE}/ask',
                    json={'query': item['question'], 'mode': mode, 'top_k': TOP_K, 'rerank': rerank},
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
                per_arm[label].append({
                    'question'       : item['question'],
                    'gold_doc_id'    : gold_doc_id,
                    'retrieved_doc_ids': retrieved_doc_ids,
                    'recall@5'       : recall_at_k(retrieved_doc_ids, gold_doc_id, 5),
                    'mrr'            : mrr(retrieved_doc_ids, gold_doc_id),
                    'ndcg@5'         : ndcg_at_k(retrieved_doc_ids, gold_doc_id, 5),
                    'faithfulness'   : faithfulness(nli, embedder, body['answer'], context_texts),
                    'answer_relevance': answer_relevance(embedder, item['question'], body['answer']),
                    'latency_ms'     : body['latency_ms'],
                    'wall_latency_ms': round(wall_ms, 1),
                    'generator'      : body['generator'],
                })
                print(f"  q={item['question'][:60]!r:62} "
                      f"R@5={per_arm[label][-1]['recall@5']:.0f} "
                      f"faith={per_arm[label][-1]['faithfulness']:.2f} "
                      f"rel={per_arm[label][-1]['answer_relevance']:.2f} "
                      f"lat={body['latency_ms']:.0f}ms")

    summary = {}
    for label, _, _ in arms:
        rows = per_arm[label]
        latencies = [row['latency_ms'] for row in rows]
        summary[label] = {
            'recall@5'        : float(np.mean([r['recall@5'] for r in rows])),
            'mrr'             : float(np.mean([r['mrr'] for r in rows])),
            'ndcg@5'          : float(np.mean([r['ndcg@5'] for r in rows])),
            'faithfulness'    : float(np.mean([r['faithfulness'] for r in rows])),
            'answer_relevance': float(np.mean([r['answer_relevance'] for r in rows])),
            'p50_ms'          : float(np.percentile(latencies, 50)),
            'p95_ms'          : float(np.percentile(latencies, 95)),
            'mean_ms'         : float(np.mean(latencies)),
        }
    return summary, per_arm


def main() -> int:
    summary, per_mode = run_evaluation()
    gold = json.loads(GOLD_PATH.read_text(encoding='utf-8'))

    generators_observed = sorted({
        row['generator'] for rows in per_mode.values() for row in rows
    })

    output = {
        'timestamp'  : datetime.now(timezone.utc).isoformat(),
        'gold_set_size': len(gold),
        'top_k'      : TOP_K,
        'embedding_model': EMBED_MODEL,
        'nli_model'  : NLI_MODEL,
        'generators_observed': generators_observed,  # derived from actual /ask responses, not hardcoded
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
