"""
D3 safety — two mitigations:

1. Prompt-injection / retrieval-poisoning defense: detect instruction-like
   text inside retrieved chunks ("ignore previous instructions", "you are
   now in developer mode", embedded "system:" prompts, etc.) so a poisoned
   document in the corpus can never become part of an answer or citation.

2. Provenance filtering / source pinning: drop answer sentences that aren't
   entailed by any retrieved context chunk, using the same NLI proxy as
   Step 4's faithfulness metric — an answer may only say things its cited
   sources actually support.

Both are pattern/entailment-based, not an LLM judge. Documented limitation:
a sufficiently paraphrased injection, or a confidently-worded but ungrounded
claim phrased to superficially echo the context's wording, could still slip
through either check. See results/d3_safety_before_after.json for a concrete
case both mitigations are demonstrated against.
"""
import re

import numpy as np

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'ignore (?:all|any|the )?(?:previous|above|prior)\s+instructions',
        r'disregard (?:all|any|the )?(?:previous|above|prior)',
        r'you are now (?:in )?(?:developer|admin|dan|jailbreak)\s*mode',
        r'\bsystem\s*:\s*',
        r'new instructions?[:.]',
        r'reveal (?:your|the) (?:system )?prompt',
        r'act as (?:if you (?:are|were)|an?)\s',
        r'forget (?:everything|all)\s+(?:you (?:know|were told)|previous)',
    ]
]


def detect_injection(text: str) -> list[str]:
    """Returns the matched injection-pattern regexes, empty if the text looks clean."""
    return [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]


def sanitize_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """chunks: list of dicts with a 'text' key. Returns (clean, flagged) —
    flagged chunks carry an added 'injection_patterns' key for audit."""
    clean, flagged = [], []
    for c in chunks:
        hits = detect_injection(c.get('text', ''))
        if hits:
            flagged.append({**c, 'injection_patterns': hits})
        else:
            clean.append(c)
    return clean, flagged


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def filter_ungrounded(nli, answer_text: str, context_texts: list[str], threshold: float = 0.5) -> dict:
    """Provenance filtering: drop answer sentences not entailed by any
    retrieved context chunk. `nli` is a CrossEncoder
    (cross-encoder/nli-deberta-v3-small); index 1 = entailment, verified
    empirically — see scripts/evaluate_d3.py."""
    sentences = _split_sentences(answer_text)
    if not sentences:
        return {'filtered_answer': '', 'dropped': [], 'grounded_fraction': 0.0}
    if not context_texts or nli is None:
        return {'filtered_answer': '', 'dropped': sentences, 'grounded_fraction': 0.0}

    kept, dropped = [], []
    for sent in sentences:
        pairs = [(ctx[:512], sent) for ctx in context_texts]
        logits = nli.predict(pairs)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        if probs[:, 1].max() > threshold:
            kept.append(sent)
        else:
            dropped.append(sent)

    return {
        'filtered_answer': ' '.join(kept),
        'dropped': dropped,
        'grounded_fraction': len(kept) / len(sentences),
    }


def validate_citations(citations: list[dict], allowed_chunk_ids: set[str]) -> list[dict]:
    """Source pinning: a citation may only reference a chunk that was
    actually part of the retrieved/ranked candidate set for this query —
    never an arbitrary or fabricated chunk_id. Returns any citations that
    fail this check (should always be empty if the executor is implemented
    correctly; this is a defense-in-depth assertion, not the primary
    mechanism)."""
    return [c for c in citations if c['chunk_id'] not in allowed_chunk_ids]
