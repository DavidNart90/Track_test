import pytest
from src.trackrealties.rag.enhanced_rag_pipeline import TrackRealitiesEnhancedRAG


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["investor", "developer", "buyer", "agent"])
async def test_role_specific_response(role, monkeypatch):
    pipeline = TrackRealitiesEnhancedRAG()

    async def fake_search(query, analysis):
        return []

    async def fake_generate(query, results, user_role, user_context):
        return {"response": f"{user_role} reply"}

    monkeypatch.setattr(pipeline, "_execute_smart_search", fake_search)
    monkeypatch.setattr(pipeline, "_generate_role_specific_response", fake_generate)

    result = await pipeline.process_query("hi", {"role": role})
    assert result["response"] == f"{role} reply"
