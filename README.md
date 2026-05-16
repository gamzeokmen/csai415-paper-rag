# CSAI415 Paper RAG — Deliverable 1

A retrieval-augmented generation system over a corpus of 10 arXiv papers, with an auto-tuned hybrid retriever and an online learning component with drift detection.

## Team

| Member | Role | D1 Contributions |
|--------|------|------------------|
| Gamze Okmen | Lead | System architecture, AutoML pipeline (notebook 04), online learning + drift detection (notebook 05), index build (notebook 02), evaluation framework, technical report sections (methodology + results), repo setup |
| Alfarouq Alsharif | Member | Corpus curation — selected the 10 arXiv papers and downloaded source PDFs (notebook 01 inputs); drafted and formatted the report (intro, related work, conclusion sections) |
| Kenan Almukhllati | Member | Gold set queries — authored the 10 manually-crafted evaluation queries (notebook 03); reproducibility verification — ran all notebooks end-to-end on a clean environment to validate outputs |

## System overview

- **Corpus:** 10 arXiv papers, 2118 chunks (500-char sliding window, 80-char overlap)
- **Embeddings:** BAAI/bge-small-en-v1.5 (384-dim)
- **Retriever:** Hybrid BM25 + dense (Qdrant in-memory)
- **AutoML:** Optuna TPE, 30 trials, 5 hyperparameters (k, alpha, metric, normalize, svd_dim)
- **Online learning:** River GaussianNB with ADWIN drift detector (delta=0.002)

## Setup
git clone <repo-url>
cd csai415-paper-rag
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## How to run

Run the notebooks in order:

1. `notebooks/01_ingest_corpus.ipynb` — download and chunk the 10 arXiv papers
2. `notebooks/02_build_index.ipynb` — build BM25 + dense index, baseline evaluation
3. `notebooks/03_gold_set.ipynb` — create the 10-query gold set
4. `notebooks/04_automl.ipynb` — Optuna hyperparameter tuning (30 trials)
5. `notebooks/05_online_learning.ipynb` — online learning + ADWIN drift detection

## Results

- **Retrieval (test split):** Recall@5 = 1.000, NDCG@5 = 1.000, p95 latency = 423.7 ms
- **Online learning:** Adaptive learner = 0.260, static baseline = 0.113, relative improvement = +131.1% (PASS)
- **Best AutoML params:** k=7, alpha=0.879, metric=dot, normalize=True, svd_dim=384

See `results/automl_run_card.yaml` and `results/online_run_card.yaml` for full details.