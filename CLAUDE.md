# Read first

1. Read `BRIEF.md` in full before writing any code. It is the specification.
2. `harness.py`, `v6_belief.py`, `verify.py`, `cache.py`, `steer.py`, `vectors.py`, `gates.py`
   are verified or gated. Reuse them. Do not rewrite, "clean up", or reformat them.
3. Every experimental parameter lives in `config.py`. Never hard-code one elsewhere; never change one.
4. Stop at every STOP point named in `BRIEF.md`: print what it asks for and wait for Tony.
5. Halting gates halt. Diagnostics report. Never remove or skip data to make something pass.
6. No interpretation of results anywhere in code, logs or reports. Numbers only.
7. Raw text prompts, no chat template. bf16. Fixed, recorded seeds. Log wall-clock per section.
