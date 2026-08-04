# Contributing

Thanks for taking a look. This is a research prototype released so others can
study, reuse and extend the operator-feedback loop — issues and pull requests
are welcome, with the caveats below.

## Before you start

- **No real machining data ships, and none should ever be added.** `data/` is
  gitignored wholesale. Use `python scripts/generate_sample_dataset.py` for a
  synthetic dataset, or point the loader at your own data outside the repo.
- **Never commit secrets.** All configuration is environment-driven;
  `.env.example` documents every variable and `.env` is gitignored.
- Please don't add identifying information about specific sites, machines or
  operators to code, comments, tests or fixtures.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ui && npm install && cd ..
cp .env.example .env
python scripts/generate_sample_dataset.py
```

## Checks to run before opening a PR

```bash
pytest                  # backend suite — must stay green
mypy                    # gradual typing — see the baseline note below
cd ui && npm run build  # tsc + vite build (this is the UI type-check)
cd ui && npm run test   # vitest
```

Tests that need a site-specific tool-master dataset skip themselves (see
`tests/conftest.py`), as do end-to-end tests when no server is running. A clean
clone with no data should report **passed + skipped, never failed**.

**`mypy` is not clean yet.** It currently reports around 140 errors — typing was
adopted gradually and the config is deliberately lenient (`disallow_untyped_defs`
and `check_untyped_defs` are off). Please don't add to the count; fixing existing
errors in code you touch is welcome, and a PR that only reduces the number is
very welcome. Compare against the count on `main` rather than expecting zero.

## Conventions

- **Optional dependencies degrade, they don't break.** Heavy or external
  components (LLM providers, Neo4j, SINDIT, PyTorch, FAISS) are imported
  defensively: `try/except` + a logged warning + a working fallback. A missing
  dependency must disable one feature, never prevent start-up. Follow this
  pattern for anything new.
- **New machine types are YAML, not Python.** Sensor semantics live in
  `domain_packs/*.yaml`, which map channel names onto semantic roles. Adding
  support for a new sensor layout means adding a pack, not editing
  `domain_config.py`.
- **New dataset layouts go through the loader.** Extend `KEY_COLUMNS` in
  `backend/agents/processing/dataset_loader.py`, or add an adapter beside it.
  Channel groups are detected from CSV headers — please keep it that way rather
  than hard-coding vendor filenames.
- Match the style of the surrounding code; there is no separate formatter step.

## Scope

The loop is deliberately **not autonomous**: every reconfiguration output is a
proposal requiring operator confirmation. Please keep that property. Changes
that would let the system act on its own are out of scope.

Claims in documentation should stay honest about what has been validated — see
"Scope and honest limitations" in the [README](README.md).

## Licence

By contributing you agree that your contributions are licensed under the
[MIT Licence](LICENSE).
