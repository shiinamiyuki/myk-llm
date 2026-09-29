# AGENTS.md

## Instructions
- Use the uv venv `/home/miyuki/repos/venvs/cu13/bin/activate` or its python directly
- Don't `rm -f` on anything. If you wrote something to `/tmp`, just leave it there.
- In the chat or code comments, the first time you say an acronym, you must spell it out. For example, if you say "NLP", you must first say "Natural Language Processing (NLP)". 
- `old` folder contains a block-diffusion LM code. It is not used. Please ignore it.
- Code lives in the `myk_llm` package under `src/` (installed editable: `pip install -e .`). Layout and workflow: README.md and notes/pipeline.md.
- Every training run is a YAML config under `configs/`; add experiments as new config files rather than new CLI flags.
- Run `python -m pytest` after changes; GPU tests need CUDA. `configs/experiments/smoke.yaml` is a few-minute end-to-end run.
- Never read outside project directory. Ask user if you encountered environment issues.