"""
One-command D3 runner — regenerates every D3 result artifact from a single
invocation, so every number in the report traces to a real, reproducible run
rather than a hand-edited file.

Runs, in order:
  1. scripts/evaluate_d3.py    -> results/d3_eval.json, d3_eval.png
  2. scripts/ablation_d3.py    -> results/d3_ablation.json, d3_ablation.png
  3. pytest tests/test_safety.py -> results/d3_safety_before_after.json
  4. pytest tests/ -v          -> final regression check
  5. writes results/d3_run_card.yaml (model ids, seeds, dataset sizes,
     timestamp, headline metrics)

Prerequisites (not started by this script):
  - docker compose up -d   (MongoDB + Qdrant)
  - uvicorn app.main:app --port 8000   (the live API; loads embedder,
    reranker, NLI cross-encoder, and connects to Mongo/Qdrant/Neo4j)

Usage:
    python scripts/run_d3.py
"""
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://localhost:8000'
PY = sys.executable


def check_server() -> None:
    try:
        urllib.request.urlopen(f'{BASE}/health', timeout=3)
    except Exception as exc:
        raise SystemExit(
            f'Cannot reach {BASE}/health ({exc}). Start the API first:\n'
            '  docker compose up -d\n'
            '  uvicorn app.main:app --port 8000'
        )


def run(cmd: list[str]) -> None:
    print(f'\n$ {" ".join(cmd)}')
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_run_card() -> None:
    eval_data = json.loads((ROOT / 'results' / 'd3_eval.json').read_text(encoding='utf-8'))
    ablation_data = json.loads((ROOT / 'results' / 'd3_ablation.json').read_text(encoding='utf-8'))
    safety_data = json.loads((ROOT / 'results' / 'd3_safety_before_after.json').read_text(encoding='utf-8'))

    card = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'd3_complete': True,
        'models': {
            'embedder': eval_data['embedding_model'],
            'reranker': 'BAAI/bge-reranker-base',
            'nli': eval_data['nli_model'],
            'generators_observed': eval_data['generators_observed'],
        },
        'seeds': {
            'gold_qa_sample_seed': 415,  # scripts/build_gold_qa.py SEED
        },
        'dataset_sizes': {
            'corpus_papers': 144,
            'corpus_chunks': 6858,
            'gold_qa_items': eval_data['gold_set_size'],
            'gold_qa_top_k': eval_data['top_k'],
        },
        'eval_summary': eval_data['summary'],
        'ablation_summary': ablation_data['summary'],
        'ablation_interpretation': ablation_data['interpretation'],
        'ablation_limitations': ablation_data['limitations'],
        'safety_demo': {
            'scenario': safety_data['scenario'],
            'before_hijacked': safety_data['before']['hijacked'],
            'after_hijacked': safety_data['after']['hijacked'],
            'after_cites_poisoned_chunk': safety_data['after']['citations_include_poisoned_chunk'],
        },
    }
    out = ROOT / 'results' / 'd3_run_card.yaml'
    out.write_text(yaml.safe_dump(card, sort_keys=False), encoding='utf-8')
    print(f'\nWrote {out}')


def main() -> int:
    check_server()
    run([PY, 'scripts/evaluate_d3.py'])
    run([PY, 'scripts/ablation_d3.py'])
    run([PY, '-m', 'pytest', 'tests/test_safety.py', '-v'])
    run([PY, '-m', 'pytest', 'tests/', '-v'])
    write_run_card()
    print('\nD3 run complete. See results/d3_eval.json, d3_ablation.json, '
          'd3_safety_before_after.json, d3_run_card.yaml.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
