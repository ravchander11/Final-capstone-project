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

import argparse
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


def semantic_retrieve(query: str, k: int = TOP_K) -> list[tuple[str, str, float]]:
    """Simple semantic-style retrieval using TF-IDF cosine similarity.

    This intentionally does not depend on a vector DB or external embeddings package.
    It still captures the key semantic idea: documents with similar term distributions
    to the query can rank highly even when the exact words differ.
    """
    docs = load_wikipedia_docs()
    if not docs or not query:
        return []

    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    all_terms = sorted({term for doc in docs for term in _tokens(f"{doc['title']} {doc['text']}")})
    doc_term_counts = [{term: counts for term, counts in Counter(_tokens(f"{doc['title']} {doc['text']}")) .items()} for doc in docs]
    doc_freq = {}
    for counts in doc_term_counts:
        for term in counts:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    idf = {term: math.log((len(docs) + 1) / (doc_freq.get(term, 1) + 1)) + 1.0 for term in all_terms}

    query_counts = Counter(query_tokens)
    q_vector = [idf.get(term, 1.0) * query_counts.get(term, 0) for term in all_terms]
    q_norm = math.sqrt(sum(v * v for v in q_vector))

    results: list[tuple[str, str, float]] = []
    for idx, doc in enumerate(docs):
        counts = Counter(_tokens(f"{doc['title']} {doc['text']}"))
        d_vector = [idf.get(term, 1.0) * counts.get(term, 0) for term in all_terms]
        d_norm = math.sqrt(sum(v * v for v in d_vector))
        if q_norm == 0 or d_norm == 0:
            sim = 0.0
        else:
            sim = sum(a * b for a, b in zip(q_vector, d_vector)) / (q_norm * d_norm)
        if sim > 0:
            results.append((doc["id"], doc["title"], float(sim)))

    results.sort(key=lambda item: item[2], reverse=True)
    return [(doc_id, title, round(score, 4)) for doc_id, title, score in results[:k]]


def hybrid_retrieve(query: str, k: int = TOP_K, lexical_weight: float = 0.6, semantic_weight: float = 0.4) -> list[tuple[str, str, float]]:
    """Combine keyword retrieval and semantic-style retrieval into a single ranked list."""
    docs = {doc["id"]: doc for doc in load_wikipedia_docs()}
    lexical_hits = retrieve(query, k=max(k * 3, 15))
    semantic_hits = semantic_retrieve(query, k=max(k * 3, 15))

    lex_map = {doc_id: score for doc_id, _, score in lexical_hits}
    sem_map = {doc_id: score for doc_id, _, score in semantic_hits}

    all_doc_ids = sorted(set(lex_map) | set(sem_map))
    if not all_doc_ids:
        return []

    lex_max = max(lex_map.values()) if lex_map else 1.0
    sem_max = max(sem_map.values()) if sem_map else 1.0

    scored: list[tuple[str, str, float]] = []
    for doc_id in all_doc_ids:
        lex_score = lex_map.get(doc_id, 0.0) / lex_max if lex_max else 0.0
        sem_score = sem_map.get(doc_id, 0.0) / sem_max if sem_max else 0.0
        combined = lexical_weight * lex_score + semantic_weight * sem_score
        if combined > 0:
            scored.append((doc_id, docs[doc_id]["title"], combined))

    scored.sort(key=lambda item: item[2], reverse=True)
    return [(doc_id, title, round(score, 4)) for doc_id, title, score in scored[:k]]


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


def _run_retrieval_section(label: str, queries: list[str], retriever, llm: ChatOpenAI | None = None) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    for i, query in enumerate(queries, 1):
        hits = retriever(query, TOP_K)
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
            log(f"{label} | QUERY {i}: {query}", f"retrieved={hits}\nanswer={ans}")
        else:
            print("  answer: LLM answer generation skipped because the API key is missing.\n")
            log(f"{label} | QUERY {i}: {query}", f"retrieved={hits}\nanswer=SKIPPED (no API key)")


def run() -> None:
    parser = argparse.ArgumentParser(description="Compare lexical, semantic, and hybrid Wikipedia retrieval.")
    parser.add_argument("--question", help="Ask a single custom question across the selected retrieval section(s).")
    parser.add_argument(
        "--section",
        choices=["text", "semantic", "hybrid", "all"],
        default="all",
        help="Choose which retrieval section to run.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt interactively for a custom question instead of using the default representative queries.",
    )
    args = parser.parse_args()

    docs = load_wikipedia_docs()
    print(f"Checkpoint 2.1 — retrieval comparison | scenario: {SCENARIO} | articles: {len(docs)}\n")

    llm = None
    try:
        llm = make_llm()
        print("LLM available: OpenRouter key detected. Grounded answers are enabled.\n")
    except RuntimeError:
        print("LLM unavailable: OPENROUTER_API_KEY not set. Retrieval-only validation is running.\n")

    if args.question:
        queries = [args.question.strip()]
    elif args.interactive:
        custom_question = input("Enter a question for the selected retrieval section(s): ").strip()
        queries = [custom_question] if custom_question else my_representative_queries()
    else:
        queries = my_representative_queries()

    section_map = {
        "text": ("TEXT-BASED SEARCH (BM25-style lexical baseline)", retrieve),
        "semantic": ("SEMANTIC SEARCH (TF-IDF cosine similarity)", semantic_retrieve),
        "hybrid": ("HYBRID SEARCH (lexical + semantic)", hybrid_retrieve),
    }

    selected_sections = [args.section] if args.section != "all" else ["text", "semantic", "hybrid"]
    for section_name in selected_sections:
        label, retriever = section_map[section_name]
        _run_retrieval_section(label, queries, retriever, llm)

    if args.section == "all":
        print("\n" + "=" * 72)
        print(
            "Comparison complete. The lexical baseline performs best on exact titles and entity-heavy "
            "queries, semantic retrieval is better for paraphrased wording, and hybrid retrieval balances "
            "both behaviors for a more robust default search strategy."
        )


if __name__ == "__main__":
    run()
