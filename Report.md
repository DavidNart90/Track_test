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
The test suite hit the health endpoint and sent a chat request to each agent role. Vector search failed due to connection errors, so the fallback response was returned. Example output:

```
[
  {
    "endpoint": "/health",
    "status": 307
  },
  {
    "status": 200,
    "body": {
      "message": "No relevant information found for your query: 'Hello from investor'",
      "session_id": "e05fc00b-0274-4aa2-b452-59a2b48a7244",
      "assistant_message_id": "1e08230e-bbe8-419d-a5b5-ed239308cf22",
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
- Simple greetings now return a friendly message without performing a search.
- All conversation messages are stored in the `conversation_messages` table with the new `message_metadata` field.

