"""
Watchdog AI - Data Quality Scoring Module
Multi-dimensional quality assessment for text data
"""

import re
import numpy as np
from typing import Dict, List, Any
from collections import Counter


class DataQualityScorer:
    """
    Evaluates data quality across multiple dimensions
    """
    
    def __init__(self):
        """Initialize quality scorer with scoring criteria"""
        self.min_length = 10
        self.max_length = 10000
        self.min_words = 3
        
    def _calculate_text_completeness(self, data: Dict[str, Any]) -> float:
        """
        Calculate completeness score based on available fields
        
        Args:
            data: Dictionary containing text and optional fields
            
        Returns:
            Completeness score (0-1)
        """
        required_fields = ['text']
        optional_fields = ['title', 'source', 'author', 'date', 'category']
        
        score = 0.0
        
        # Required fields (60% of score)
        for field in required_fields:
            if field in data and data[field] and len(str(data[field]).strip()) > 0:
                score += 0.6
        
        # Optional fields (40% of score, 8% each)
        present_optional = sum(1 for field in optional_fields 
                              if field in data and data[field] and len(str(data[field]).strip()) > 0)
        score += (present_optional / len(optional_fields)) * 0.4
        
        return score
    
    def _calculate_text_length_score(self, text: str) -> float:
        """
        Score based on text length (not too short, not too long)
        
        Args:
            text: Text string
            
        Returns:
            Length score (0-1)
        """
        if not text:
            return 0.0
        
        length = len(text)
        
        if length < self.min_length:
            return 0.3
        elif length < 50:
            return 0.5
        elif length < 100:
            return 0.7
        elif length <= self.max_length:
            return 1.0
        else:
            # Penalize extremely long texts
            return max(0.3, 1.0 - (length - self.max_length) / self.max_length)
    
    def _calculate_word_count_score(self, text: str) -> float:
        """
        Score based on word count
        
        Args:
            text: Text string
            
        Returns:
            Word count score (0-1)
        """
        if not text:
            return 0.0
        
        words = text.split()
        word_count = len(words)
        
        if word_count < self.min_words:
            return 0.2
        elif word_count < 10:
            return 0.5
        elif word_count < 20:
            return 0.7
        elif word_count < 500:
            return 1.0
        else:
            return 0.9
    
    def _calculate_language_quality(self, text: str) -> float:
        """
        Evaluate language quality (grammar indicators, punctuation, etc.)
        
        Args:
            text: Text string
            
        Returns:
            Language quality score (0-1)
        """
        if not text:
            return 0.0
        
        score = 0.0
        
        # Check for proper capitalization
        sentences = re.split(r'[.!?]+', text)
        capitalized_sentences = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        if sentences and len([s for s in sentences if s.strip()]) > 0:
            score += (capitalized_sentences / len([s for s in sentences if s.strip()])) * 0.3
        
        # Check for punctuation
        has_punctuation = bool(re.search(r'[.!?,;:]', text))
        score += 0.2 if has_punctuation else 0.0
        
        # Check for reasonable sentence structure
        avg_sentence_length = len(text) / max(len(sentences), 1)
        if 20 <= avg_sentence_length <= 200:
            score += 0.2
        
        # Check for excessive capitalization (SPAM indicator)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio < 0.3:  # Less than 30% caps is good
            score += 0.15
        
        # Check for excessive punctuation/special chars
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(len(text), 1)
        if special_ratio < 0.2:  # Less than 20% special chars is good
            score += 0.15
        
        return min(score, 1.0)
    
    def _calculate_information_density(self, text: str) -> float:
        """
        Measure information density (unique words, vocabulary richness)
        
        Args:
            text: Text string
            
        Returns:
            Information density score (0-1)
        """
        if not text:
            return 0.0
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 0.0
        
        # Lexical diversity (unique words / total words)
        unique_words = len(set(words))
        total_words = len(words)
        lexical_diversity = unique_words / total_words
        
        # Penalize very low diversity (repetitive text)
        if lexical_diversity < 0.3:
            diversity_score = 0.3
        elif lexical_diversity < 0.5:
            diversity_score = 0.6
        else:
            diversity_score = min(lexical_diversity, 1.0)
        
        # Check average word length (complexity indicator)
        avg_word_length = sum(len(w) for w in words) / len(words)
        complexity_score = min(avg_word_length / 6, 1.0)  # Normalized to 6 chars
        
        # Combined score
        return (diversity_score * 0.7) + (complexity_score * 0.3)
    
    def _calculate_spam_indicators(self, text: str) -> float:
        """
        Check for spam/junk indicators (returns inverse score - lower is worse)
        
        Args:
            text: Text string
            
        Returns:
            Anti-spam score (0-1, where 1 is not spam)
        """
        if not text:
            return 0.5
        
        spam_score = 0.0
        
        # Excessive exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count > 5:
            spam_score += 0.2
        elif exclamation_count > 10:
            spam_score += 0.4
        
        # Excessive capitalization
        caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
        if caps_words > 5:
            spam_score += 0.2
        
        # Spam keywords
        spam_keywords = [
            'SHOCKING', 'MIRACLE', 'GUARANTEE', 'FREE', 'CLICK HERE',
            'LIMITED TIME', 'ACT NOW', 'URGENT', 'WINNER', '100%',
            'CONGRATULATIONS', 'CLAIM NOW', 'EXCLUSIVE'
        ]
        keyword_matches = sum(1 for keyword in spam_keywords if keyword in text.upper())
        if keyword_matches > 2:
            spam_score += 0.3
        
        # Excessive special characters
        special_count = len(re.findall(r'[!@#$%^&*()]{3,}', text))
        if special_count > 0:
            spam_score += 0.2
        
        # Return inverse (1 = good, 0 = spam)
        return max(0.0, 1.0 - min(spam_score, 1.0))
    
    def _detect_quality_issues(self, data: Dict[str, Any], individual_scores: Dict[str, float]) -> List[str]:
        """
        Detect specific quality issues
        
        Args:
            data: Data dictionary
            individual_scores: Individual scoring results
            
        Returns:
            List of quality issues detected
        """
        issues = []
        
        text = str(data.get('text', ''))
        
        if individual_scores['completeness'] < 0.5:
            issues.append('Missing important fields')
        
        if individual_scores['length'] < 0.5:
            issues.append('Text too short or too long')
        
        if individual_scores['word_count'] < 0.5:
            issues.append('Insufficient word count')
        
        if individual_scores['language_quality'] < 0.5:
            issues.append('Poor language quality')
        
        if individual_scores['information_density'] < 0.4:
            issues.append('Low information density')
        
        if individual_scores['spam_check'] < 0.6:
            issues.append('Contains spam indicators')
        
        return issues
    
    def score_data(self, data: Dict[str, Any]) -> Dict:
        """
        Calculate overall quality score for a data item
        
        Args:
            data: Dictionary containing 'text' and optional fields
            
        Returns:
            Dictionary with quality scores and assessment
        """
        text = str(data.get('text', ''))
        
        # Calculate individual scores
        individual_scores = {
            'completeness': self._calculate_text_completeness(data),
            'length': self._calculate_text_length_score(text),
            'word_count': self._calculate_word_count_score(text),
            'language_quality': self._calculate_language_quality(text),
            'information_density': self._calculate_information_density(text),
            'spam_check': self._calculate_spam_indicators(text)
        }
        
        # Weighted overall score
        weights = {
            'completeness': 0.15,
            'length': 0.10,
            'word_count': 0.15,
            'language_quality': 0.25,
            'information_density': 0.20,
            'spam_check': 0.15
        }
        
        overall_score = sum(individual_scores[key] * weights[key] 
                           for key in individual_scores)
        
        # Determine quality level
        if overall_score >= 0.8:
            quality_level = 'high'
        elif overall_score >= 0.6:
            quality_level = 'medium'
        elif overall_score >= 0.4:
            quality_level = 'low'
        else:
            quality_level = 'very_low'
        
        # Detect issues
        issues = self._detect_quality_issues(data, individual_scores)
        
        return {
            'overall_score': overall_score,
            'quality_level': quality_level,
            'individual_scores': individual_scores,
            'issues': issues,
            'recommendation': self._get_recommendation(overall_score, issues)
        }
    
    def _get_recommendation(self, score: float, issues: List[str]) -> str:
        """Get recommendation based on quality score"""
        if score >= 0.8:
            return 'Excellent quality - keep'
        elif score >= 0.6:
            return 'Good quality - keep with minor review'
        elif score >= 0.4:
            return 'Fair quality - review before keeping'
        else:
            return 'Poor quality - consider removing'
    
    def score_dataset(self, dataset: List[Dict[str, Any]]) -> Dict:
        """
        Score an entire dataset
        
        Args:
            dataset: List of data dictionaries
            
        Returns:
            Aggregate statistics for the dataset
        """
        if not dataset:
            return {
                'total_items': 0,
                'average_score': 0.0,
                'quality_distribution': {},
                'common_issues': []
            }
        
        scores = []
        quality_levels = []
        all_issues = []
        
        for item in dataset:
            result = self.score_data(item)
            scores.append(result['overall_score'])
            quality_levels.append(result['quality_level'])
            all_issues.extend(result['issues'])
        
        # Calculate statistics
        quality_distribution = Counter(quality_levels)
        issue_distribution = Counter(all_issues)
        
        return {
            'total_items': len(dataset),
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'quality_distribution': dict(quality_distribution),
            'common_issues': issue_distribution.most_common(5),
            'high_quality_count': quality_distribution.get('high', 0),
            'low_quality_count': quality_distribution.get('low', 0) + quality_distribution.get('very_low', 0)
        }