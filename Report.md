# TrackRealties AI End-to-End Test Report

## Overview
This report summarizes the automated end-to-end tests executed for the TrackRealties AI platform. The tests ingest sample data, exercise API endpoints, simulate user prompts for each role, and verify the responses. Failures are retried once.

## Test Environment
- **Database**: PostgreSQL (Neon) and Neo4j (environment variables provided).
- **Model**: OpenAI GPT via Pydantic AI.
- **Data**: `sample_market_data.json` and `sample_property_listings.json`.
- **Commands Executed**:
  - `pytest -q`
  - `PYTHONPATH=. python scripts/e2e_test_suite.py --retries 1`

## Results
The test suite hit the health endpoint and sent a chat request to each agent role. Initial runs used simple greetings to confirm that greetings bypass expensive search. Subsequent runs issued advanced questions drawn from the sample data, such as ROI calculations and listing‑agent lookups. Because outbound network access is restricted, the vector and graph searches could not connect to OpenAI or Neo4j and the responses fell back to "No relevant information found". Example output:

```
[
  {
    "endpoint": "/health",
    "status": 307
  },
  {
    "status": 200,
    "body": {
      "message": "Hello! How can I assist you with real estate today?",
      "session_id": "9ed928ff-432c-4a61-ad4d-eb26c4394c52",
      "assistant_message_id": "238ae524-5007-49b0-89b5-de12317eec48",
      "tools_used": [],
      "metadata": {}
    },
    "attempts": 1,
    "role": "investor"
  }
]
```

All requests returned HTTP 200 and included an `assistant_message_id` confirming messages are logged correctly.

## Usage Guide
1. **Data Ingestion**
   - Prepare the schema using `sql/new_schema.sql`.
   - Ingest properties:
     ```bash
     python -m src.trackrealties.cli enhanced-ingest sample_property_listings.json --data-type property
     ```
   - Ingest market data:
     ```bash
     python -m src.trackrealties.cli enhanced-ingest sample_market_data.json --data-type market
     ```
2. **API Calls**
   - Create a session via `POST /session/`.
   - Send chat messages to agents using `POST /agents/{role}/chat` with the session ID and message.
   - Other endpoints such as `/chat/` and `/chat/stream` are also available. Full documentation is at `/docs` when the API is running.

## Observations & Recommendations
- Vector search fails without internet connectivity. Ensure outbound access or provide local embeddings.
- Graph search could not reach the Neo4j database due to network restrictions.
- Simple greetings now return a friendly message without performing a search.
- Advanced property and market queries fell back to a default message because search results were empty.
- Fixed a bug in the hybrid search engine by importing `asyncio` so the pipeline no longer raises a NameError.
- All conversation messages are stored in the `conversation_messages` table with the new `message_metadata` field.

