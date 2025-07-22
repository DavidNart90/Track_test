# Task 3

This task introduces scripts and code updates for training and loading role specific language models.

## Changes Made
- Added `scripts/prepare_training_data.py` and `scripts/fine_tune_models.py` for generating datasets and fine tuning models.
- Implemented simple preprocessing and fine tuning utilities in `src/data_preparation.py` and `src/fine_tuning.py`.
- Extended `BaseAgent` to accept a `model_path` argument so agents can reference fine‑tuned models.
- Updated all role agents to define default paths to their models.
- Created `create_agent` in `src/trackrealties/agents/factory.py` to load these paths and instantiate agents.
- Updated orchestrator and agent API route to use this factory function.
- Documented training instructions in `README.md`.

## Usage
1. Prepare datasets:
   ```bash
   python scripts/prepare_training_data.py
   ```
2. Fine tune the models:
   ```bash
   python scripts/fine_tune_models.py
   ```
   Models are stored in `models/{role}_llm/` and automatically loaded when agents are created.

## Files Changed
- `scripts/prepare_training_data.py`
- `scripts/fine_tune_models.py`
- `src/data_preparation.py`
- `src/fine_tuning.py`
- `src/trackrealties/agents/base.py`
- `src/trackrealties/agents/investor.py`
- `src/trackrealties/agents/developer.py`
- `src/trackrealties/agents/buyer.py`
- `src/trackrealties/agents/agent.py`
- `src/trackrealties/agents/factory.py`
- `src/trackrealties/agents/orchestrator.py`
- `src/trackrealties/api/routes/agents.py`
- `src/trackrealties/agents/__init__.py`
- `README.md`
- `Task 3.md` (this document)

No additional Python packages were installed during this task.
