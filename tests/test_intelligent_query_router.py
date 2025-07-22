import pytest
from src.trackrealties.rag.intelligent_router import IntelligentQueryRouter

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is the connection between A and B?", "graph_only"),
        ("How many properties are there?", "vector_only"),
        ("Should I invest in Austin?", "hybrid"),
        ("Tell me something", "hybrid"),
    ],
)
async def test_router_routing(query, expected):
    router = IntelligentQueryRouter()
    result = await router.analyze_query(query, {})
    assert result.primary_strategy == expected
