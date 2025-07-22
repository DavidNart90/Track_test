# Task 2

This update introduces an intelligent query router and improved context handling for TrackRealties AI.

## Highlights

- **New router module** `src/trackrealties/rag/router.py` containing:
  - `RealEstateEntityExtractor` for regex based entity extraction.
  - `QueryIntentClassifier` to detect user intent from a query.
  - `IntelligentQueryRouter` which chooses between vector, graph or hybrid search and can execute the selected strategy.
- **Enhanced RAG pipeline** now routes each query through this router before searching.
- **Context manager updates** expose `create_session`, `read_session`, `update_session`, and `delete_session` helpers for CRUD style access.
- **Agents** record conversation history using the `ContextManager` on every `run()` call.

## Usage

1. Instantiate your agent as before. When `run()` is called, the agent saves user and assistant messages to the session context.
2. The router analyses each query and selects the best search method automatically.
3. Session information and preferences can be managed programmatically through the new CRUD methods of `ContextManager`.

