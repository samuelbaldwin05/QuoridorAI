# Quoridor AI

Quoridor game engine and AI agents built from scratch for CS XXXX.

## Setup

**With uv:**
```bash
uv venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
uv pip install -e ".[dev]"
```

**With pip:**
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

To include training dependencies:
```bash
uv pip install -e ".[train,dev]"   # or pip install -e ".[train,dev]"
```

## Project Structure

```
quoridor/
├── quoridor/
│   ├── game.py          # core game rules and state
│   ├── display.py       # terminal rendering
│   └── env.py           # ML interface (action encoding, observations)
├── agents/              # AI agents (random, minimax, RL)
├── tests/               # unit tests
├── scripts/             # runnable entry points (play, benchmark)
├── docs/                # proposal, summary
└── pyproject.toml
```