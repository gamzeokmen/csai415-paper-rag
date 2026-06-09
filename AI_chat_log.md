# AI Chat Logs — CSAI415 Deliverables 1 & 2

This file contains links to the complete unedited AI conversations (Claude and ChatGPT) used during the development of D1 and D2, submitted as required by the course rubric.

---

## D1 Chats

### Chat 1 — AI project with four deliverables and lab implementation

This was the primary development chat where the D1 technical work (corpus ingestion, retriever build, AutoML pipeline, online learning + drift detection, evaluation framework) was discussed and implemented.

**Link:** https://claude.ai/share/34e15e54-546e-4ed6-837c-b6a566e96eec

### Chat 2 — CSAI415 D1 project wrap-up steps

This chat covered the D1 final wrap-up work: finalizing the report, setting up the GitHub repository, coordinating teammate commits, exporting the chat logs, and building the submission package.

**Link:** https://claude.ai/share/fbab7507-be56-403b-aa85-862f2d551216

---

## D2 Chats

### Chat 3 — D2 production stack, evaluation, and graph build

This was the primary D2 development chat covering: Docker Compose setup (MongoDB + Qdrant), 144-paper ingestion pipeline, async FastAPI app with 10 endpoints, hybrid retrieval (BM25 + dense + RRF + cross-encoder rerank), 30-query gold set evaluation achieving R@5=1.000 with reranker, Neo4j Aura knowledge graph (144 Paper / 50 Author / 5 Topic nodes, WROTE + ABOUT + 300 synthetic CITES edges), 11 pytest smoke tests, per-stage latency breakdown, per-query analysis, and the D2 report.

Key debugging sessions included: Docker volume/database name mismatches, qdrant-client API migration (search → query_points), MongoDB ObjectId vs arXiv ID type mismatches in the gold set, and synthetic CITES edge generation using the venue field.

**Links:**
- https://chatgpt.com/share/6a206f70-a78c-832a-958b-cb2ebe9c28e1
- https://chatgpt.com/share/6a207d4c-498c-8331-bef5-0eef4de4260c

---

## Summary of AI usage

AI assistants (Claude and ChatGPT) were used across the project for:
- Architectural decisions (RRF over weighted sum, async driver selection, cross-encoder reranker integration)
- Code generation for FastAPI endpoints, evaluation scripts, plotting code, and Jupyter notebooks
- Debugging cascading errors during Docker/database integration
- Writing and iterating on the D2 report
- Pre-submission audit identifying and fixing metric inconsistencies

All code was reviewed, executed, and verified by the team. All metrics in the report come from actual notebook outputs, not generated text.
