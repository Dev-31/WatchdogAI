"""
Integration tests for full pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from src.dataset_processor import DatasetProcessor


class TestPipeline:
    """Test suite for full data processing pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        return pd.DataFrame([
            {"id": 1, "text": "Scientific research shows climate effects.", "source": "nature.com"},
            {"id": 2, "text": "SHOCKING!!! GUARANTEED results!!!", "source": "spam.com"},
            {"id": 3, "text": "Quarterly earnings up 15%.", "source": "company.com"},
            {"id": 4, "text": "Scientific research shows climate effects.", "source": "science.org"},  # Duplicate
            {"id": 5, "text": "bad", "source": ""},  # Low quality
        ])
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return DatasetProcessor()
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor is not None
        assert processor.detector is not None
        assert processor.scorer is not None
        assert processor.redundancy is not None
        assert processor.tracker is not None
    
    def test_process_dataframe_basic(self, processor, sample_data):
        """Test basic dataframe processing"""
        results = processor.process_dataframe(
            sample_data,
            text_column='text',
            source_column='source',
            verbose=False
        )
        
        assert 'final_df' in results
        assert 'original_count' in results
        assert 'final_count' in results
        assert results['final_count'] <= results['original_count']
    
    def test_process_removes_duplicates(self, processor, sample_data):
        """Test that duplicates are removed"""
        results = processor.process_dataframe(
            sample_data,
            text_column='text',
            remove_duplicates=True,
            verbose=False
        )
        
        # Should remove at least the duplicate entry
        assert results['final_count'] < results['original_count']
        assert 'redundancy' in results['steps']
    
    def test_process_removes_low_quality(self, processor, sample_data):
        """Test that low quality items are removed"""
        results = processor.process_dataframe(
            sample_data,
            text_column='text',
            quality_threshold=0.5,
            verbose=False
        )
        
        assert 'quality' in results['steps']
        # "bad" should be removed
        assert results['final_count'] < results['original_count']
    
    def test_process_keeps_high_risk(self, processor, sample_data):
        """Test option to keep high-risk items"""
        results = processor.process_dataframe(
            sample_data,
            text_column='text',
            remove_high_risk=False,
            verbose=False
        )
        
        # Should process but not remove high-risk items
        assert 'final_df' in results
    
    def test_sustainability_tracking(self, processor, sample_data):
        """Test sustainability metrics are calculated"""
        results = processor.process_dataframe(
            sample_data,
            text_column='text',
            verbose=False
        )
        
        assert 'sustainability' in results
        assert 'immediate_savings' in results['sustainability']
        assert 'projected_annual_savings' in results['sustainability']
    
    def test_process_list(self, processor):
        """Test processing list of dicts"""
        data = [
            {"text": "Good content here", "source": "nature.com"},
            {"text": "More good content", "source": "science.org"},
        ]
        
        results = processor.process_list(data, verbose=False)
        assert results['final_count'] > 0
    
    def test_invalid_column_raises_error(self, processor, sample_data):
        """Test that invalid column name raises error"""
        with pytest.raises(ValueError):
            processor.process_dataframe(
                sample_data,
                text_column='nonexistent_column',
                verbose=False
            )
    
    def test_retention_rate(self, processor, sample_data):
        """Test retention rate calculation"""
        results = processor.process_dataframe(
            sample_data,
            text_column='text',
            verbose=False
        )
        
        assert 'retention_rate' in results
        assert 0 <= results['retention_rate'] <= 100
    
    def test_processing_time_recorded(self, processor, sample_data):
        """Test that processing time is recorded"""
        results = processor.process_dataframe(
            sample_data,
            text_column='text',
            verbose=False
        )
        
        assert 'processing_time' in results
        assert results['processing_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
