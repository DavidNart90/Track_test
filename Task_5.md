# Task 5

This task integrates the prompt engineering templates into the TrackRealties agents and prepares them to load role specific fine‑tuned models.

## Summary of Work

- Added `src/trackrealties/agents/prompts.py` containing the prompt templates from `trackrealities_prompt_engineering_doc.md`.
- Updated `InvestorAgent`, `DeveloperAgent`, `BuyerAgent` and `AgentAgent` to:
  - Use the imported prompt templates combined with the base system context for their system prompts.
  - Retrieve a fine‑tuned model for their role from `deps.rag_pipeline.role_models` when created.
  - Build their tool list via a helper method so all role specific tools from `tools.py` are included.
- Created helper methods to accept optional dependencies for tool creation.

## Usage

Instantiate any agent as before. If `AgentDependencies.rag_pipeline.role_models` contains a model for the role, it will automatically be used. System prompts now follow the standardized templates.

## Files Changed

- `src/trackrealties/agents/investor.py`
- `src/trackrealties/agents/developer.py`
- `src/trackrealties/agents/buyer.py`
- `src/trackrealties/agents/agent.py`
- `src/trackrealties/agents/prompts.py` *(new)*

