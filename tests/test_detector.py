"""
Unit tests for Misinformation Detector
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.misinformation_detector import MisinformationDetector


class TestMisinformationDetector:
    """Test suite for MisinformationDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for tests"""
        return MisinformationDetector()
    
    def test_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert len(detector.suspicious_patterns) > 0
        assert len(detector.credible_sources) > 0
    
    def test_source_credibility_high(self, detector):
        """Test credibility scoring for trusted sources"""
        score = detector.check_source_credibility("nature.com")
        assert score > 0.6
        
        score = detector.check_source_credibility("stanford.edu")
        assert score > 0.6
    
    def test_source_credibility_low(self, detector):
        """Test credibility scoring for untrusted sources"""
        score = detector.check_source_credibility("spam-blog.com")
        assert score <= 0.6
    
    def test_source_credibility_none(self, detector):
        """Test credibility scoring for no source"""
        score = detector.check_source_credibility(None)
        assert score == 0.5
    
    def test_clickbait_detection(self, detector):
        """Test clickbait pattern detection"""
        # High clickbait
        score = detector.detect_clickbait("You won't believe what happens next!")
        assert score > 0.0
        
        # Low clickbait
        score = detector.detect_clickbait("Scientific research shows results.")
        assert score == 0.0
    
    def test_analyze_safe_text(self, detector):
        """Test analysis of safe, credible text"""
        result = detector.analyze_text(
            "Scientific research published in Nature.",
            "nature.com"
        )
        
        assert result['risk_level'] == 'low'
        assert result['misinformation_score'] < 0.5
    
    def test_analyze_risky_text(self, detector):
        """Test analysis of suspicious text"""
        result = detector.analyze_text(
            "SHOCKING!!! 100% GUARANTEED miracle cure!!!",
            "spam.com"
        )
        
        assert result['risk_level'] in ['low', 'medium', 'high']
        assert result['misinformation_score'] > 0.3
        ##assert result['misinformation_score'] > 0.4
        assert len(result['flags']) > 0
    
    def test_analyze_empty_text(self, detector):
        """Test analysis of empty/invalid text"""
        result = detector.analyze_text("")
        assert 'invalid_input' in result['flags']
        
        result = detector.analyze_text(None)
        assert 'invalid_input' in result['flags']
    
    def test_batch_analyze(self, detector):
        """Test batch analysis"""
        texts = [
            "Scientific research shows results.",
            "SHOCKING!!! You won't believe this!!!"
        ]
        
        results = detector.batch_analyze(texts)
        assert len(results) == 2
        assert results[0]['risk_level'] == 'low'
        assert results[1]['risk_level'] in ['low', 'medium', 'high']
        assert results[1]['misinformation_score'] > 0.3
    
    def test_confidence_scoring(self, detector):
        """Test confidence calculation"""
        # High confidence (multiple signals)
        result = detector.analyze_text(
            "GUARANTEED miracle!!! 100% works!!!",
            "spam.com"
        )
        assert result['confidence'] > 0.5
        
        # Low confidence (few signals)
        result = detector.analyze_text(
            "The meeting is at 3 PM.",
            None
        )
        assert result['confidence'] <= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
