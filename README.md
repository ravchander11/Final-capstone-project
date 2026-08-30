# Capstone Checkpoint 2.1 — Baseline Retrieval

This folder contains the completed baseline retrieval implementation for the Wikipedia scenario used in the capstone checkpoint. The goal is to retrieve relevant article pages for a query using a lexical BM25-style baseline, then optionally use an LLM grounded in the retrieved documents.

Files included:
- `capstone_checkpoint_2_1_baseline_retrieval_starter.py` — main retrieval script
- `lab_2_1_vector_retrieval_solution.py` — reference implementation from the lab
- `Wikipedia/` — local corpus of Wikipedia HTML pages used for testing
- `README.md` — project overview and usage
- `docs-MAINTAINING.md` — maintenance notes
- `requirements.txt` — Python dependencies

## Retrieval approach

This implementation selects a keyword-based baseline because the Wikipedia corpus is made up of many entity-focused pages (people, places, events, championships, and historical disasters). In practice, this baseline works well for queries that include exact article titles or highly specific topic terms.

It is intentionally simple and transparent:
- load the local Wikipedia HTML documents
- extract titles + text content
- score articles by term overlap and title relevance
- return the top-k results for a query
- optionally call the LLM with only the retrieved document context

This reflects the retrieval tradeoff learned in the labs:
- keyword/BM25 is strong on exact-match queries
- semantic/vector retrieval helps more with paraphrases and synonyms
- a hybrid system would combine both, but the baseline is deliberately lexical and easy to test

## Running the script

From the project root:

```bash
python capstone_checkpoint_2_1_baseline_retrieval_starter.py
```

If you want the LLM to answer the questions after retrieval, create a `.env` file in this folder with:

```bash
OPENROUTER_API_KEY=sk-or-...
```

The retrieval results still work without that key; the LLM answer step simply becomes unavailable until the key is configured.

## Output and evidence

When the script runs, it prints the representative queries and the retrieved article hits. It also appends results to `checkpoint_2_1_retrieval.log`, which can be used as evidence for the checkpoint write-up.

## Notes

- This script is designed for the local Wikipedia corpus bundled in this folder.
- If you later switch to a different corpus, update the retrieval logic and representative queries accordingly.
- The file is structured to be readable as a script and easy to adapt for a written submission.
