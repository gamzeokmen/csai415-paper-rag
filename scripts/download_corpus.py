"""Reproducibly download the paper corpus from the manifest.

Reads ``data/corpus_manifest.csv`` and downloads any PDF that is not already
present in ``data/papers/``. Downloads are rate-limited with retry/backoff so a
fresh clone can rebuild the corpus without tripping arXiv's HTTP 429 limiter.

Usage:
    python scripts/download_corpus.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import arxiv

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = REPO_ROOT / "data" / "papers"
MANIFEST = REPO_ROOT / "data" / "corpus_manifest.csv"


def manifest_ids() -> list[str]:
    """Return the list of paper ids recorded in the manifest."""
    if not MANIFEST.exists():
        sys.exit(f"Manifest not found: {MANIFEST}. Run build_manifest.py first.")
    with MANIFEST.open(encoding="utf-8") as fh:
        return [r["paper_id"] for r in csv.DictReader(fh) if r.get("paper_id")]


def main() -> int:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    ids = manifest_ids()
    missing = [pid for pid in ids if not (PAPERS_DIR / f"{pid}.pdf").exists()]
    print(f"{len(ids)} papers in manifest; {len(missing)} missing locally.")
    if not missing:
        print("Corpus already complete.")
        return 0

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    ok = 0
    for result in client.results(arxiv.Search(id_list=missing)):
        pid = result.get_short_id().split("v")[0]
        try:
            result.download_pdf(dirpath=str(PAPERS_DIR), filename=f"{pid}.pdf")
            ok += 1
            print(f"  + {pid}.pdf")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {pid} failed: {exc}", file=sys.stderr)
        time.sleep(1.0)
    print(f"Downloaded {ok}/{len(missing)} missing PDFs.")
    return 0 if ok == len(missing) else 2


if __name__ == "__main__":
    raise SystemExit(main())
