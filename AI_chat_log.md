# AI Chat Logs — CSAI415 Deliverables 1, 2 & 3

This file contains links to the complete unedited AI conversations (Claude and ChatGPT) used during the development of D1 and D2, and a summary of the D3 AI-assisted session, submitted as required by the course rubric.

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

## D3 Session

### Chat 4 — D3 GraphRAG executor, evaluation, safety, and generator integration

This was the primary D3 development chat, driven end-to-end by **Kenan
Almukhllati**, covering the full D3 scope in one continuous session: GraphRAG
executor design, evaluation harness, safety mitigations, ablation study,
hygiene/documentation pass, and the LLM generator integration. It was run via
Claude Code (CLI), not a claude.ai web chat, so no shareable
`claude.ai/share/...` link exists for it — Claude Code sessions run against a
local repository checkout instead. The full session transcript is included
directly in this repo so the same level of detail D1/D2's share links provide
is still available:

**Full transcript:** [`D3_chat_log.md`](D3_chat_log.md) — a cleaned, readable
export of the complete local session log (74 exchanges). Typos and informal
phrasing in the human turns were corrected for readability; no requests,
decisions, or technical content were altered. One credential leaked mid-session
during a debugging exchange (a Neo4j Aura database password) has been redacted.

The summary below covers the same scope, condensed.

**Scope:** built D3 (GraphRAG executor, evaluation harness, safety mitigations,
ablation, hygiene, report) on top of the graded D2 system, end to end, plus a
follow-up to wire in a real LLM generator. All 9 D3 PRs (#2–#10) plus a
follow-up generator PR (#11) were opened as drafts, reviewed by the team
member present, and merged into `main` only after explicit approval at each
stage — no direct pushes to `main`.

**Real bugs found and fixed during this session** (not hidden, all called out
in their respective PR descriptions and commit messages):
1. `app/main.py` keyed retrieval on a `chunk_idx` field that doesn't exist in
   the real schema — every dense hit silently fell back to index 0, making
   real page-numbered citations impossible until fixed.
2. The new D3 gold-set's fallback question template collapsed to ~3 generic
   strings shared across multiple papers (zero retrieval signal) — caught
   because a first evaluation run scored Recall@5 ≈ 0.06–0.11 across all
   three modes; fixed with content-anchored question generation.
3. A stray PDF-extraction punctuation artifact broke the same gold-set logic
   for one item, producing a grammatically broken question.
4. A generator-stub disclaimer string was leaking into the text graded for
   faithfulness/relevance, dragging both metrics down for no real reason.
5. An early draft of the provenance filter fell back to *unfiltered* text
   whenever nothing survived the filter — which would have silently defeated
   the entire safety mitigation if shipped.
6. The provenance filter's NLI-entailment-only check, once a real LLM
   generator was wired in, rejected nearly every well-grounded paraphrased
   sentence (verified empirically — a clearly-grounded paraphrase scored
   0.001 entailment probability against its own source). Fixed by switching
   to an NLI-contradiction + embedding-similarity check instead.
7. A flaky test (`tests/test_api.py`) had no explicit HTTP timeout and
   started intermittently failing once the server's startup load increased.
8. Random sampling while building the gold set surfaced 7 papers in the
   144-paper corpus with no connection to RAG/retrieval at all (particle
   physics, pure math, geophysics) — flagged for the team, not silently
   dropped from awareness.

**Key decisions made, with trade-offs surfaced for the team rather than
decided unilaterally:** scoping Step 1 to build retrieval mechanics before
the generator; capping shared-topic graph expansion given a 96%-dominant
single category; using Ollama (`qwen2.5:1.5b`, CPU) as a substitute for the
brief's GPU-bitsandbytes-4-bit generator plan, since the brief's own
`Generator` interface explicitly allowed an `ollama` backend and the dev
machine had no CUDA.

**Verification discipline:** every numeric claim in `README.md` and
`D3_Report.md` was cross-checked against `results/d3_run_card.yaml` (the
generated, non-hand-edited source of truth) before being written. The full
test suite (23/23) and the one-command `scripts/run_d3.py` pipeline were
re-run multiple times throughout the session, including after each
significant code change, not just once at the end.

---

## Summary of AI usage

AI assistants (Claude and ChatGPT) were used across the project for:
- Architectural decisions (RRF over weighted sum, async driver selection, cross-encoder reranker integration, multi-hop Cypher design, NLI-based safety/faithfulness checks)
- Code generation for FastAPI endpoints, evaluation scripts, plotting code, Jupyter notebooks, and the D3 GraphRAG executor
- Debugging cascading errors during Docker/database integration and D3 evaluation pipeline development
- Writing and iterating on the D2 and D3 reports
- Pre-submission audits identifying and fixing metric inconsistencies, both in D2 and during D3 (gold-set defects, a leaking disclaimer string, an over-strict NLI filter)

All code was reviewed, executed, and verified by the team. All metrics in the report come from actual notebook outputs, not generated text.
