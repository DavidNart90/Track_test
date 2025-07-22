import pytest
from src.trackrealties.validation.hallucination import RealEstateHallucinationDetector

@pytest.mark.asyncio
async def test_hallucination_detector_flags_unrealistic_values():
    detector = RealEstateHallucinationDetector()
    text = "This property is priced at $5,000,000,000 and offers an ROI of 150%."
    result = await detector.validate(text, {})
    assert not result.is_valid
    assert len(result.issues) >= 2

@pytest.mark.asyncio
async def test_hallucination_detector_accepts_normal_text():
    detector = RealEstateHallucinationDetector()
    text = "The property is priced at $500,000 with an expected ROI of 10%."
    result = await detector.validate(text, {})
    assert result.is_valid
    assert result.issues == []
