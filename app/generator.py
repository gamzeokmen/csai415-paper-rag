"""
D3 — pluggable Generator interface (brief: hf default, ollama, api).

Implemented here: OllamaGenerator, calling a local Ollama server
(http://localhost:11434) running qwen2.5:1.5b — the same model id the D3
brief specifies, just GGUF-quantized for CPU instead of bitsandbytes 4-bit
on GPU. Chosen because this dev machine has no CUDA support; Ollama keeps
the model warm in memory (~3s per short generation after the first
~45s cold load) and needed no GPU/bitsandbytes setup. Stays fully
offline — nothing leaves the laptop, consistent with the brief's ethics
rationale.

Model id is overridable via the GEN_MODEL env var, per the brief's note
that D4's PEFT/QLoRA-tuned adapter should be swappable in later.
"""
import logging
import os

import httpx

log = logging.getLogger('csai415_rag.generator')

GEN_MODEL = os.getenv('GEN_MODEL', 'qwen2.5:1.5b')
OLLAMA_URL = 'http://localhost:11434/api/generate'

SYSTEM_PROMPT = (
    'You are a research assistant answering questions about retrieval-augmented '
    'generation papers. Answer ONLY using the numbered context below. Cite sources '
    "inline using the [doc_id p.X] tags shown. If the context doesn't contain the "
    'answer, say so plainly — do not make anything up.'
)


class OllamaGenerator:
    def __init__(self, model: str = GEN_MODEL, timeout: float = 60.0):
        self.model = model
        self.timeout = timeout

    async def generate(self, query: str, numbered_context: str) -> str:
        prompt = (
            f'{SYSTEM_PROMPT}\n\n'
            f'Context:\n{numbered_context}\n\n'
            f'Question: {query}\n'
            f'Answer:'
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(OLLAMA_URL, json={
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.2, 'seed': 415},
            })
            r.raise_for_status()
            return r.json()['response'].strip()
