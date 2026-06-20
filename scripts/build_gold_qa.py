"""
Build the D3 gold Q/A set — eval/gold_qa.json.

Draws each item from a real paper's own abstract text (not fabricated) and
drafts {question, reference_answer, gold_doc_id, gold_pages}. This is
explicitly a DRAFT for human review, not an oracle: the D3 brief calls for a
"human-checkable" gold set, and the D1/D2 instructor feedback flagged AI
doing the thinking instead of the team. Run this once, then have each of the
3 team members read through a slice of eval/gold_qa.json and edit/replace any
question or reference_answer that doesn't read naturally, then flip its
`reviewed` flag to true.

Schema note: only the original 10 D1 papers have a standalone chunk_type
"abstract" / page_num 0. The other 134 papers (D2's 144-paper ingestion)
never split the abstract into its own chunk — it's embedded inline in each
paper's first body chunk (page_num 1, chunk_seq 0), after the literal word
"Abstract". This script extracts it from there when a dedicated abstract
chunk doesn't exist, and records gold_pages accordingly (0 for the D1 set,
1 for everything else) — so the D3 gold set, unlike gold_set_d2.json, is
standardized on arXiv doc_id throughout and pages are real, not assumed.

Usage:
    python scripts/build_gold_qa.py
"""
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'eval' / 'gold_qa.json'
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')

SEED = 415  # course code — fixed for reproducibility
N_ITEMS = 18

_CONTRIBUTION_RE = re.compile(
    r'(?:we propose|we present|we introduce|this paper proposes|this paper presents|'
    r'this work presents|this work proposes)\s+(.{10,90}?)[.,;]',
    re.IGNORECASE,
)

_QUESTION_TEMPLATES = [
    'What approach is proposed for {topic}, and how does it work?',
    'How does the described method address {topic}?',
    'What technique is introduced to improve {topic}?',
    'What is the key idea behind the proposed solution for {topic}?',
]

_GENERIC_FALLBACK_TEMPLATES = [
    'What problem motivates this paper, and what solution does it propose?',
    'What is the main contribution described in this paper?',
    'What method does this paper introduce, and what is it used for?',
]

# Corpus is supposed to be ~144 papers on RAG/retrieval research, but contains
# some off-topic papers (e.g. particle physics, pure math) that slipped into
# the manifest — filter the gold-set candidate pool to genuinely on-topic
# papers so evaluation actually tests the system's intended domain.
_TOPIC_KEYWORDS = (
    'retriev', 'rag', 'generation', 'language model', 'llm', 'embedding',
    'question answering', 'dense passage', 'knowledge', 'search', 'corpus',
    'document', 'index', 'reranker', 'rerank', 'graph', 'augment',
)


def is_on_topic(title: str, reference_answer: str) -> bool:
    text = f'{title} {reference_answer}'.lower()
    return any(kw in text for kw in _TOPIC_KEYWORDS)


def extract_abstract_text(db, doc_id: str) -> tuple[str, int] | None:
    """Returns (abstract_text, page_num) or None if nothing usable is found."""
    abstract_chunk = db.chunks.find_one({'doc_id': doc_id, 'chunk_type': 'abstract'})
    if abstract_chunk and abstract_chunk.get('text'):
        return abstract_chunk['text'], abstract_chunk.get('page_num', 0)

    first_chunk = db.chunks.find_one(
        {'doc_id': doc_id, 'page_num': 1, 'chunk_seq': 0}
    )
    if not first_chunk or not first_chunk.get('text'):
        return None
    text = first_chunk['text']
    marker = text.find('Abstract')
    if marker == -1:
        return None
    return text[marker + len('Abstract'):].strip(), first_chunk['page_num']


def first_sentences(text: str, n: int = 2) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:n]).strip()


def draft_question(reference_answer: str, rng: random.Random) -> str:
    m = _CONTRIBUTION_RE.search(reference_answer)
    if m:
        topic = m.group(1).strip().rstrip('.,;')
        return rng.choice(_QUESTION_TEMPLATES).format(topic=topic)
    return rng.choice(_GENERIC_FALLBACK_TEMPLATES)


def main() -> int:
    db = MongoClient(MONGO_URI).csai415_rag
    doc_ids = [d['doc_id'] for d in db.documents.find({}, {'doc_id': 1})]

    rng = random.Random(SEED)
    rng.shuffle(doc_ids)

    gold = []
    skipped_off_topic = []
    for doc_id in doc_ids:
        if len(gold) >= N_ITEMS:
            break
        extracted = extract_abstract_text(db, doc_id)
        if not extracted:
            continue
        abstract_text, page_num = extracted
        reference_answer = first_sentences(abstract_text, 2)
        if len(reference_answer.split()) < 8:
            continue  # too short to be a usable reference answer

        doc = db.documents.find_one({'doc_id': doc_id}, {'title': 1})
        title = doc.get('title') if doc else ''
        if not is_on_topic(title or '', reference_answer):
            skipped_off_topic.append(doc_id)
            continue

        gold.append({
            'question'        : draft_question(reference_answer, rng),
            'reference_answer': reference_answer,
            'gold_doc_id'     : doc_id,
            'gold_pages'      : [page_num],
            'title'           : title,
            'reviewed'        : False,
        })

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Drafted {len(gold)} gold Q/A items -> {OUT}')
    if skipped_off_topic:
        print(f'Skipped {len(skipped_off_topic)} off-topic papers found in the corpus: {skipped_off_topic}')
        print('      (these likely don\'t belong in a RAG/retrieval-research corpus — worth flagging)')
    print('NOTE: this is a DRAFT. Each item has reviewed=false — a human must')
    print('      read it, fix the question/reference_answer if needed, and')
    print('      flip reviewed=true before this is used for D3 evaluation.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
