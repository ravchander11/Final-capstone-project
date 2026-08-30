r"""Capstone Checkpoint 2.1 — baseline retrieval for the Wikipedia corpus.

This implementation follows the structure of the lab solution but chooses a lexical
baseline for this scenario. Wikipedia queries often mention specific entities, titles,
and events, so a keyword/BM25-style retriever is a sensible starting point. It is
strong on exact matches and title overlap, while it struggles on paraphrased queries
that a semantic retriever would handle better. A hybrid approach would combine both,
but this file implements a clear, testable baseline that reflects what we learned in
Module 2.
"""

from __future__ import annotations

import html
import math
import os
import re
import warnings
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
TOP_K = 5
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval.log"
CORPUS_DIR = Path(__file__).resolve().parent / "Wikipedia"
SCENARIO = "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


def check_api_key() -> str:
    """Load the API key only when the LLM is actually needed."""
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Retrieval still works without an API key, "
            "but to answer questions with the LLM, add a .env file next to this script "
            "with OPENROUTER_API_KEY=sk-or-..."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _html_to_text(raw_html: str) -> str:
    cleaned = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.I | re.S)
    cleaned = re.sub(r"<style.*?</style>", " ", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


@lru_cache(maxsize=1)
def load_wikipedia_docs() -> tuple[dict[str, str], ...]:
    docs: list[dict[str, str]] = []
    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"Wikipedia corpus directory not found: {CORPUS_DIR}")

    for html_file in sorted(CORPUS_DIR.glob("*.html")):
        text = html_file.read_text(encoding="utf-8", errors="replace")
        title_match = re.search(r"<title>(.*?)</title>", text, flags=re.I | re.S)
        title = title_match.group(1).strip() if title_match else html_file.stem
        title = re.sub(r"\s*-\s*Wikipedia\s*$", "", title, flags=re.I)
        body = _html_to_text(text)
        if not body:
            continue
        docs.append({
            "id": html_file.stem,
            "title": title,
            "text": body,
        })

    return tuple(docs)


@lru_cache(maxsize=1)
def build_bm25_index() -> dict[str, object]:
    docs = load_wikipedia_docs()
    num_docs = max(len(docs), 1)
    doc_freq: dict[str, int] = {}
    doc_lengths: list[int] = []

    for doc in docs:
        tokens = _tokens(f"{doc['title']} {doc['text']}")
        doc_lengths.append(len(tokens))
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    idf = {
        term: math.log((num_docs + 1) / (freq + 1)) + 1.0
        for term, freq in doc_freq.items()
    }
    avg_doc_length = sum(doc_lengths) / num_docs if doc_lengths else 1.0
    return {"idf": idf, "avg_doc_length": avg_doc_length}


def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, str, float]]:
    """BM25-style lexical retrieval for the local Wikipedia corpus.

    This baseline favors exact term overlap and title matches, which is effective for
    many Wikipedia queries that name a person, place, work, or event. It is weaker for
    paraphrased wording that uses synonyms or different phrasing from the source page.
    """
    docs = load_wikipedia_docs()
    if not query:
        return []

    query_terms = set(_tokens(query))
    if not query_terms:
        return []

    index = build_bm25_index()
    idf = index["idf"]
    avg_doc_length = float(index["avg_doc_length"])
    k1 = 1.5
    b = 0.75

    scored: list[tuple[str, str, float]] = []
    for doc in docs:
        full_text = f"{doc['title']} {doc['text']}"
        tokens = _tokens(full_text)
        counts = Counter(tokens)
        title_tokens = set(_tokens(doc["title"]))

        score = 0.0
        for term in query_terms & counts.keys():
            tf = counts[term]
            idf_term = idf.get(term, 1.0)
            doc_length = max(len(tokens), 1)
            denom = doc_length * (1 - b) + b * avg_doc_length
            score += idf_term * ((tf * (k1 + 1)) / (tf + k1 * (denom / avg_doc_length if avg_doc_length else 1.0)))

        title_overlap = len(query_terms & title_tokens)
        if title_overlap:
            score += 2.5 * title_overlap

        if score > 0:
            scored.append((doc["id"], doc["title"], float(score)))

    scored.sort(key=lambda item: item[2], reverse=True)
    return [(doc_id, title, round(score, 3)) for doc_id, title, score in scored[:k]]


def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    docs = {doc["id"]: doc for doc in load_wikipedia_docs()}
    context = "\n\n".join(
        f"[{i}] {docs[i]['title']}\n{docs[i]['text']}" for i in doc_ids if i in docs
    )
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


def my_representative_queries() -> list[str]:
    """Three to five representative queries for the Wikipedia scenario.

    They cover title-exact retrieval, natural-language paraphrase, and event-based
    factual queries that should be answerable from the corresponding page.
    """
    return [
        "1984 NBA Finals",
        "Who won the 1984 NBA Finals?",
        "1854 Broad Street cholera outbreak",
        "What caused the 1854 Broad Street cholera outbreak?",
        "1964 Alaska earthquake",
    ]


def run() -> None:
    docs = load_wikipedia_docs()
    print(f"Checkpoint 2.1 — baseline retrieval | scenario: {SCENARIO} | articles: {len(docs)}\n")

    llm = None
    try:
        llm = make_llm()
        print("LLM available: OpenRouter key detected. Grounded answers enabled.\n")
    except RuntimeError:
        print("LLM unavailable: OPENROUTER_API_KEY not set. Retrieval-only validation is running.\n")

    for i, query in enumerate(my_representative_queries(), 1):
        hits = retrieve(query, TOP_K)
        print("=" * 72)
        print(f"QUERY {i}: {query}")
        if not hits:
            print("  retrieved: []")
            print("  (no matching article found)")
            continue

        print("  retrieved:")
        for doc_id, title, score in hits:
            print(f"    - {doc_id}: {title} (score={score})")

        if llm is not None:
            ans = answer(llm, query, [doc_id for doc_id, _, _ in hits])
            print(f"  answer: {ans}\n")
            log(f"QUERY {i}: {query}", f"retrieved={hits}\nanswer={ans}")
        else:
            print("  answer: LLM answer generation skipped because the API key is missing.\n")
            log(f"QUERY {i}: {query}", f"retrieved={hits}\nanswer=SKIPPED (no API key)")

    print("=" * 72)
    print(
        "Baseline evidence complete. This BM25-style lexical retriever is strong on exact title "
        "matches and entity-heavy queries, but it is less robust when the query is paraphrased or "
        "uses different wording than the article itself."
    )


if __name__ == "__main__":
    run()
