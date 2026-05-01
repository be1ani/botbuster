# Botbuster

Botbuster turns recorded user-interaction sessions (JSON) into numeric features, trains a gradient-boosted classifier to separate human traffic from synthetic bots, and runs inference on new sessions.

## Repository layout

| Path | Role |
|------|------|
| `data/human/` | Labeled human interaction JSON (one file per collection) |
| `data/synthetic/` | Labeled synthetic/bot JSON |
| `extract_features.py` | Builds `features.csv` from the directories above |
| `train_model.py` | Trains `sklearn` `GradientBoostingClassifier` and saves a joblib model |
| `inference.py` | Loads the model + column order from CSV and scores a JSON file |
| `botbuster/` | Shared paths and constants used by the scripts |
| `models/v0/` | Default location for the trained `.pkl` |
| `FEATURES.md` | Per-feature definitions for the engineering feature set |

Architecture diagrams (Mermaid and optional PNG exports) live under [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

Create a virtual environment, install dependencies, then run the pipeline from the repository root so `import botbuster` resolves.

```bash
cd /path/to/botbuster
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python extract_features.py
python train_model.py
python inference.py data/synthetic/some-session.json
```

### CLI options

- **`extract_features.py`**: `--human-dir`, `--bot-dir`, `-o` / `--output` override defaults (`data/human`, `data/synthetic`, `features.csv` at repo root).
- **`train_model.py`**: `--csv` and `-o` / `--model-out` override `features.csv` and `models/v0/bot_detection_model.pkl`.
- **`inference.py`**: `--model` and `--csv` (paths relative to the script directory unless absolute).

Labels in the CSV: bot/synthetic = `0`, human = `1` (see `botbuster/constants.py`).

## Documentation graphs

To regenerate PNG diagrams from the Mermaid sources:

```bash
./docs/generate_graphs.sh
```

This uses [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli) via `npx` when `mmdc` is not on your `PATH`. Chromium runs with the repo-local [docs/puppeteer.json](docs/puppeteer.json) flags so headless rendering works on locked-down Linux hosts. Output is written to `docs/images/`.

## Interaction JSON format

Each file is a nested object: `user_id` → `session_id` → list of events. Events include `action` (`mouse_move`, `click`, `keypress`, `scroll`, …), `timestamp` (milliseconds), and action-specific fields such as `x` / `y` for pointer events.
