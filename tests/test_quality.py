"""
Watchdog AI - Quality Scorer Tests
Unit tests for the data quality scoring module
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quality_scorer import DataQualityScorer


@pytest.fixture
def scorer():
    """Create a DataQualityScorer instance for testing"""
    return DataQualityScorer()


class TestTextCompleteness:
    """Test completeness scoring"""
    
    def test_minimal_data(self, scorer):
        """Test with only text field"""
        data = {'text': 'This is a test'}
        result = scorer.score_data(data)
        assert 0 <= result['individual_scores']['completeness'] <= 1
    
    def test_complete_data(self, scorer):
        """Test with all fields present"""
        data = {
            'text': 'This is a comprehensive test with all fields',
            'title': 'Test Article',
            'source': 'test.com',
            'author': 'Test Author',
            'date': '2024-01-01',
            'category': 'test'
        }
        result = scorer.score_data(data)
        assert result['individual_scores']['completeness'] >= 0.8
    
    def test_missing_text(self, scorer):
        """Test with missing text field"""
        data = {'title': 'Only title'}
        result = scorer.score_data(data)
        assert result['individual_scores']['completeness'] < 0.5


class TestTextLength:
    """Test length scoring"""
    
    def test_very_short_text(self, scorer):
        """Test very short text"""
        data = {'text': 'Hi'}
        result = scorer.score_data(data)
        assert result['individual_scores']['length'] < 0.5
    
    def test_optimal_length(self, scorer):
        """Test optimal length text"""
        data = {'text': 'This is a well-sized text with multiple sentences. It contains enough content to be meaningful and informative.'}
        result = scorer.score_data(data)
        assert result['individual_scores']['length'] >= 0.7
    
    def test_very_long_text(self, scorer):
        """Test extremely long text"""
        data = {'text': 'A' * 15000}
        result = scorer.score_data(data)
        assert result['individual_scores']['length'] < 1.0


class TestWordCount:
    """Test word count scoring"""
    
    def test_insufficient_words(self, scorer):
        """Test text with too few words"""
        data = {'text': 'Only two'}
        result = scorer.score_data(data)
        assert result['individual_scores']['word_count'] < 0.5
    
    def test_good_word_count(self, scorer):
        """Test text with good word count"""
        data = {'text': 'This text has a reasonable number of words to convey meaningful information and context.'}
        result = scorer.score_data(data)
        assert result['individual_scores']['word_count'] >= 0.7


class TestLanguageQuality:
    """Test language quality scoring"""
    
    def test_proper_capitalization(self, scorer):
        """Test text with proper capitalization"""
        data = {'text': 'This is a proper sentence. It has good capitalization and punctuation.'}
        result = scorer.score_data(data)
        assert result['individual_scores']['language_quality'] > 0.5
    
    def test_poor_capitalization(self, scorer):
        """Test text with poor capitalization"""
        data = {'text': 'this has no capitalization or proper punctuation'}
        result = scorer.score_data(data)
        assert result['individual_scores']['language_quality'] < 0.8
    
    def test_excessive_caps(self, scorer):
        """Test text with excessive capitalization (spam-like)"""
        data = {'text': 'THIS IS ALL CAPS AND LOOKS LIKE SPAM!!!'}
        result = scorer.score_data(data)
        assert result['individual_scores']['language_quality'] < 0.7


class TestInformationDensity:
    """Test information density scoring"""
    
    def test_repetitive_text(self, scorer):
        """Test highly repetitive text"""
        data = {'text': 'same word same word same word same word same word'}
        result = scorer.score_data(data)
        assert result['individual_scores']['information_density'] < 0.5
    
    def test_diverse_text(self, scorer):
        """Test text with good vocabulary diversity"""
        data = {'text': 'Artificial intelligence revolutionizes healthcare through innovative diagnostic algorithms and predictive analytics.'}
        result = scorer.score_data(data)
        assert result['individual_scores']['information_density'] > 0.5


class TestSpamIndicators:
    """Test spam detection"""
    
    def test_spam_keywords(self, scorer):
        """Test text with spam keywords"""
        data = {'text': 'SHOCKING!!! MIRACLE CURE 100% GUARANTEED FREE CLICK HERE NOW!!!'}
        result = scorer.score_data(data)
        assert result['individual_scores']['spam_check'] < 0.6
    
    def test_normal_text(self, scorer):
        """Test normal non-spam text"""
        data = {'text': 'Scientific research shows promising results in renewable energy technology.'}
        result = scorer.score_data(data)
        assert result['individual_scores']['spam_check'] > 0.7
    
    def test_excessive_exclamations(self, scorer):
        """Test text with too many exclamation marks"""
        data = {'text': 'Amazing!! Incredible!! Unbelievable!! You won\'t believe this!!!!!'}
        result = scorer.score_data(data)
        assert result['individual_scores']['spam_check'] < 0.8


class TestOverallScoring:
    """Test overall quality assessment"""
    
    def test_high_quality_text(self, scorer):
        """Test high-quality text"""
        data = {
            'text': 'Scientific research published in peer-reviewed journals demonstrates significant advancements in renewable energy technology. These developments suggest promising applications for sustainable infrastructure.',
            'source': 'nature.com',
            'title': 'Renewable Energy Advances'
        }
        result = scorer.score_data(data)
        assert result['quality_level'] in ['high', 'medium']
        assert result['overall_score'] >= 0.6
    
    def test_low_quality_text(self, scorer):
        """Test low-quality text"""
        data = {'text': 'bad'}
        result = scorer.score_data(data)
        assert result['quality_level'] in ['low', 'very_low']
        assert result['overall_score'] < 0.6
    
    def test_spam_text(self, scorer):
        """Test obvious spam"""
        data = {'text': 'CLICK HERE NOW!!! FREE MONEY GUARANTEED 100%!!! ACT NOW!!!'}
        result = scorer.score_data(data)
        assert result['quality_level'] in ['low', 'very_low']
        assert 'spam indicators' in ' '.join(result['issues']).lower()


class TestQualityIssues:
    """Test issue detection"""
    
    def test_issue_detection(self, scorer):
        """Test that issues are properly detected"""
        data = {'text': 'bad'}
        result = scorer.score_data(data)
        assert len(result['issues']) > 0
    
    def test_no_issues_for_good_text(self, scorer):
        """Test that good text has fewer issues"""
        data = {
            'text': 'This is a well-written article with proper grammar, punctuation, and sufficient length. It provides valuable information and maintains good quality throughout.',
            'source': 'example.com',
            'title': 'Quality Article'
        }
        result = scorer.score_data(data)
        assert len(result['issues']) < 3


class TestRecommendations:
    """Test recommendation generation"""
    
    def test_recommendation_included(self, scorer):
        """Test that recommendations are provided"""
        data = {'text': 'This is a test article with reasonable quality.'}
        result = scorer.score_data(data)
        assert 'recommendation' in result
        assert isinstance(result['recommendation'], str)
        assert len(result['recommendation']) > 0


class TestDatasetScoring:
    """Test scoring of entire datasets"""
    
    def test_empty_dataset(self, scorer):
        """Test with empty dataset"""
        result = scorer.score_dataset([])
        assert result['total_items'] == 0
        assert result['average_score'] == 0.0
    
    def test_small_dataset(self, scorer):
        """Test with small dataset"""
        dataset = [
            {'text': 'High quality scientific article with proper formatting and comprehensive information.'},
            {'text': 'bad'},
            {'text': 'SPAM!!! CLICK HERE NOW!!!'}
        ]
        result = scorer.score_dataset(dataset)
        assert result['total_items'] == 3
        assert 0 <= result['average_score'] <= 1
        assert 'quality_distribution' in result
        assert 'common_issues' in result
    
    def test_quality_distribution(self, scorer):
        """Test quality distribution calculation"""
        dataset = [
            {'text': 'Excellent scientific research article with comprehensive analysis and proper citations.', 'source': 'nature.com'},
            {'text': 'bad'},
        ]
        result = scorer.score_dataset(dataset)
        assert 'quality_distribution' in result
        assert sum(result['quality_distribution'].values()) == 2


class TestEdgeCases:
    """Test edge cases"""
    
    def test_empty_text(self, scorer):
        """Test with empty text"""
        data = {'text': ''}
        result = scorer.score_data(data)
        assert result['overall_score'] < 0.5
    
    def test_none_text(self, scorer):
        """Test with None as text"""
        data = {'text': None}
        result = scorer.score_data(data)
        assert result['overall_score'] < 0.6
    
    def test_numeric_text(self, scorer):
        """Test with numeric content"""
        data = {'text': '12345 67890'}
        result = scorer.score_data(data)
        assert 0 <= result['overall_score'] <= 1
    
    def test_special_characters(self, scorer):
        """Test with special characters"""
        data = {'text': '!@#$%^&*() special chars ###'}
        result = scorer.score_data(data)
        assert 0 <= result['overall_score'] <= 1
    
    def test_mixed_content(self, scorer):
        """Test with mixed quality content"""
        data = {'text': 'Good start. But then... CLICK HERE!!! 100% FREE!!!'}
        result = scorer.score_data(data)
        assert result['quality_level'] in ['low', 'medium', 'very_low']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])