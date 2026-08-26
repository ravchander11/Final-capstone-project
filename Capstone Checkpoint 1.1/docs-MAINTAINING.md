# Maintaining this project

This file mirrors docs/MAINTAINING.md in the project root. Use it to describe how to keep the checkpoint starter and its documentation up to date.

1. Update dependencies

- Edit `requirements.txt` to add or pin packages.
- Run `pip install -r requirements.txt` in your environment.

2. Change prompts or code

- Edit `my_probe_prompts()` in `capstone_checkpoint_1_1_baseline_starter.py` to add, remove, or modify probes.
- When changing the LLM call or model settings, test the script locally and confirm the log outputs are sensible.

3. Running the baseline

```
python capstone_checkpoint_1_1_baseline_starter.py
```

4. Logging and evidence

- Check `checkpoint_1_1_responses.log` for appended entries. Use this as evidence for the write-up.

5. Documentation updates

- Add short notes in this file when you change behavior (what/why/date). Keep entries brief.

6. Committing changes

- Commit code and documentation together with a clear message, e.g. "Update probes and docs: add cricket prompt".
