import asyncio
import pytest
from src.trackrealties.rag.validation import RealEstateHallucinationDetector


@pytest.mark.asyncio
async def test_hallucination_detection():
    detector = RealEstateHallucinationDetector()
    response = "The property sold for $500,000 yesterday."
    search_results = {"results": [{"content": "The property sold for $400,000 yesterday."}]}
    result = await detector.detect_hallucinations(response, search_results)
    assert result["has_hallucinations"] is True
    assert result["issues"]
