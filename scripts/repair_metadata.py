"""Normalize document metadata in MongoDB from the corpus manifest.

PDF ingestion stored the arXiv header line (e.g. "arXiv:2002.08909v1 [cs.CL]")
as the title for ~12 papers and never set a top-level ``arxiv_id`` field. This
script repairs both from the authoritative arXiv titles in
``data/corpus_manifest.csv``:

* ``documents.title``   <- manifest title (only where it differs)
* ``documents.arxiv_id``<- ``doc_id`` (the arXiv id) for every document

It is idempotent — safe to run repeatedly.

Usage:
    python scripts/repair_metadata.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus_manifest.csv"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST}. Run build_manifest.py first.")
    titles = {
        r["paper_id"]: r["title"].strip()
        for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))
        if r.get("title", "").strip()
    }

    db = MongoClient(MONGO_URI).csai415_rag
    titles_fixed = 0
    ids_set = 0
    for doc in db.documents.find({}, {"doc_id": 1, "title": 1}):
        pid = doc.get("doc_id")
        update: dict[str, str] = {"arxiv_id": pid}
        good = titles.get(pid)
        if good and good != doc.get("title"):
            update["title"] = good
        res = db.documents.update_one({"_id": doc["_id"]}, {"$set": update})
        if res.modified_count:
            ids_set += 1
        if "title" in update:
            titles_fixed += 1
    print(f"Updated {ids_set} documents; corrected {titles_fixed} titles from the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
