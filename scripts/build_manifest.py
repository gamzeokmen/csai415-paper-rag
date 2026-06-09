"""Build a complete dataset manifest for the paper corpus.

Scans ``data/papers/*.pdf`` (each file is named by its arXiv id), fetches
metadata from the arXiv API in rate-limited batches, and writes
``data/corpus_manifest.csv`` with one row per paper.

This satisfies the project brief's *Dataset guidance*: a CSV with
``paper_id, title, authors, venue, year, pdf_path/url, topics``.

Usage:
    python scripts/build_manifest.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import arxiv

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = REPO_ROOT / "data" / "papers"
OUT_CSV = REPO_ROOT / "data" / "corpus_manifest.csv"

FIELDS = [
    "paper_id",
    "title",
    "authors",
    "venue",
    "year",
    "primary_category",
    "topics",
    "pdf_filename",
    "pdf_path",
    "abs_url",
    "pdf_url",
]


def local_ids() -> list[str]:
    """Return sorted arXiv ids derived from the PDF filenames on disk."""
    return sorted(p.stem for p in PAPERS_DIR.glob("*.pdf"))


def fetch_metadata(ids: list[str], batch: int = 40) -> dict[str, arxiv.Result]:
    """Fetch arXiv metadata for ``ids``, keyed by version-stripped id.

    Queries in small id_list batches with retries + a polite delay; small
    batches are more reliable than one large paginated query and stay under
    arXiv's HTTP 429 rate limiter.
    """
    client = arxiv.Client(page_size=batch, delay_seconds=3.0, num_retries=5)
    by_id: dict[str, arxiv.Result] = {}
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        try:
            for result in client.results(arxiv.Search(id_list=chunk)):
                by_id[result.get_short_id().split("v")[0]] = result
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
            print(f"  ! batch {start // batch} incomplete ({exc})", file=sys.stderr)
    return by_id


def row_for(paper_id: str, meta: arxiv.Result | None) -> dict[str, str]:
    """Build one CSV row, falling back to id/path/url-only when metadata is missing."""
    base = {
        "paper_id": paper_id,
        "pdf_filename": f"{paper_id}.pdf",
        "pdf_path": f"data/papers/{paper_id}.pdf",
        "abs_url": f"https://arxiv.org/abs/{paper_id}",
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
    }
    if meta is None:
        return {**{f: "" for f in FIELDS}, **base}
    return {
        **base,
        "title": meta.title.replace("\n", " ").strip(),
        "authors": "; ".join(a.name for a in meta.authors),
        "venue": (meta.journal_ref or "").strip(),
        "year": str(meta.published.year) if meta.published else "",
        "primary_category": meta.primary_category or "",
        "topics": ", ".join(meta.categories),
    }


def main() -> int:
    ids = local_ids()
    if not ids:
        print(f"No PDFs found in {PAPERS_DIR}", file=sys.stderr)
        return 1
    print(f"Found {len(ids)} PDFs; fetching arXiv metadata...")
    meta = fetch_metadata(ids)
    print(f"Got metadata for {len(meta)}/{len(ids)} papers.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for pid in ids:
            writer.writerow(row_for(pid, meta.get(pid)))
    print(f"Wrote {len(ids)} rows -> {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
