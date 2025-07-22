Task 6 - Add diagnostic and context management endpoints
=======================================================

This task introduces new API endpoints for diagnostics and context
management and updates the project documentation.

### Changes

- Added `/rag/query-router` endpoint for analysing a query and returning
  the selected routing strategy as well as any extracted entities.
- Added `/conversations/context/{session_id}` GET and DELETE endpoints
  to retrieve or clear conversation context from the in-memory
  `ContextManager`.
- Updated `README.md` with documentation for the new endpoints.
- Created this summary file.

### Files Modified

- `src/trackrealties/api/routes/rag.py`
- `src/trackrealties/api/routes/conversation.py`
- `README.md`
- `Task 6.md`
