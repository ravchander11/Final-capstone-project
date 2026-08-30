# Maintaining this project

This file tracks the operational notes for the Capstone Checkpoint 2.1 baseline retrieval implementation.

1. Update dependencies

- Edit `requirements.txt` when new runtime libraries are needed.
- Reinstall dependencies with:

```bash
pip install -r requirements.txt
```

2. Update the retrieval baseline

- Keep the logic in `capstone_checkpoint_2_1_baseline_retrieval_starter.py` aligned with the chosen capstone scenario.
- If the corpus or query patterns change, update the `my_representative_queries()` list and verify the top-k retrieval results still make sense.
- If moving beyond the lexical baseline, document the change clearly in the code comments and in the write-up.

3. Run validation locally

```bash
python capstone_checkpoint_2_1_baseline_retrieval_starter.py
```

4. Logging and evidence

- The script writes retrieval and answer evidence to `checkpoint_2_1_retrieval.log`.
- Use the printed query results and the log file as evidence for the checkpoint worksheet.
- When the OpenRouter key is missing, the script still validates retrieval; the LLM answer step is skipped gracefully.

5. Documentation updates

- Add brief notes here when the retrieval strategy, dataset, or query set changes.
- Keep entries short and factual so the maintenance notes remain useful for later checkpoints.

6. Committing changes

- Commit code and documentation together with a clear message such as: `Update Wikipedia retrieval baseline and docs`.
