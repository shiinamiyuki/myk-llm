# AGENTS.md

## Instructions
- Use the uv venv `. setenv.sh` or its python directly
- In the chat or code comments, the first time you say an acronym, you must spell it out. For example, if you say "NLP", you must first say "Natural Language Processing (NLP)". 
- Code lives in the `myk_llm` package under `src/` (installed editable: `pip install -e .`). Layout and workflow: README.md and notes/pipeline.md.
- Every training run is a YAML config under `configs/`; add experiments as new config files rather than new CLI flags.
- Run `python -m pytest` after changes; GPU tests need CUDA. `configs/experiments/smoke.yaml` is a few-minute end-to-end run.
- You are running inside a sandbox. Ask user if you encountered environment issues.