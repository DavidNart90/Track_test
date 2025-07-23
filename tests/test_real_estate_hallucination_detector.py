import pytest
from src.trackrealties.validation.hallucination import RealEstateHallucinationDetector


@pytest.mark.asyncio
async def test_hallucination_detection():
    detector = RealEstateHallucinationDetector()
    response = "This property is priced at $250,000,000 and offers an ROI of 150%."
    result = await detector.validate(response, {})
    assert not result.is_valid
    assert result.issues
