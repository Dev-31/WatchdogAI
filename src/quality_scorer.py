"""
Watchdog AI - Enhanced Quality Scorer
"""

import re
import numpy as np
from typing import Dict, List, Any
from collections import Counter


class DataQualityScorer:
    
    def __init__(self):
        self.min_length = 20
        self.max_length = 10000
        self.min_words = 5
        
        self.filler_words = [
            'good', 'bad', 'nice', 'great', 'fine', 'okay', 'well',
            'many', 'some', 'things', 'stuff', 'very', 'really', 'quite',
            'several', 'lot', 'much', 'often', 'always', 'never'
        ]
    
    def _calculate_text_completeness(self, data: Dict[str, Any]) -> float:
        required_fields = ['text']
        optional_fields = ['title', 'source', 'author', 'date', 'category']
        
        score = 0.0
        
        for field in required_fields:
            if field in data and data[field] and len(str(data[field]).strip()) > 0:
                score += 0.6
        
        present_optional = sum(1 for field in optional_fields 
                              if field in data and data[field] and len(str(data[field]).strip()) > 0)
        score += (present_optional / len(optional_fields)) * 0.4
        
        return score
    
    def _calculate_text_length_score(self, text: str) -> float:
        if not text:
            return 0.0
        
        length = len(text)
        
        if length < 20:
            return 0.2
        elif length < 50:
            return 0.4
        elif length < 100:
            return 0.6
        elif length < 200:
            return 0.8
        elif length <= self.max_length:
            return 1.0
        else:
            return max(0.4, 1.0 - (length - self.max_length) / self.max_length)
    
    def _calculate_word_count_score(self, text: str) -> float:
        if not text:
            return 0.0
        
        words = text.split()
        word_count = len(words)
        
        if word_count < 5:
            return 0.1
        elif word_count < 10:
            return 0.3
        elif word_count < 20:
            return 0.5
        elif word_count < 50:
            return 0.8
        elif word_count < 500:
            return 1.0
        else:
            return 0.85
    
    def _calculate_language_quality(self, text: str) -> float:
        if not text:
            return 0.0
        
        score = 0.0
        
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return 0.1
        
        capitalized_sentences = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        if sentences:
            capitalization_ratio = capitalized_sentences / len(sentences)
            score += capitalization_ratio * 0.25
        
        has_punctuation = bool(re.search(r'[.!?,;:]', text))
        score += 0.2 if has_punctuation else 0.0
        
        avg_sentence_length = len(text) / max(len(sentences), 1)
        if 15 <= avg_sentence_length <= 150:
            score += 0.25
        elif 10 <= avg_sentence_length <= 200:
            score += 0.15
        
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio < 0.25:
            score += 0.15
        elif caps_ratio < 0.4:
            score += 0.05
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(len(text), 1)
        if special_ratio < 0.15:
            score += 0.15
        elif special_ratio < 0.25:
            score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_information_density(self, text: str) -> float:
        if not text:
            return 0.0
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        lexical_diversity = unique_words / total_words
        
        filler_count = sum(1 for word in words if word in self.filler_words)
        filler_ratio = filler_count / total_words
        
        repetition_penalty = 1.0 - (filler_ratio * 0.6)
        
        word_counter = Counter(words)
        most_common_count = word_counter.most_common(1)[0][1] if word_counter else 1
        repetition_score = 1.0 - min(most_common_count / total_words, 0.6)
        
        if lexical_diversity < 0.2:
            diversity_score = 0.1
        elif lexical_diversity < 0.4:
            diversity_score = 0.4
        elif lexical_diversity < 0.6:
            diversity_score = 0.7
        else:
            diversity_score = 1.0
        
        avg_word_length = sum(len(w) for w in words) / len(words)
        complexity_score = min(avg_word_length / 7, 1.0)
        
        final_score = (diversity_score * 0.4) + (complexity_score * 0.2) + (repetition_score * 0.2) + (repetition_penalty * 0.2)
        
        return min(final_score, 1.0)
    
    def _calculate_spam_indicators(self, text: str) -> float:
        if not text:
            return 0.5
        
        spam_score = 0.0
        
        exclamation_count = text.count('!')
        if exclamation_count > 8:
            spam_score += 0.4
        elif exclamation_count > 5:
            spam_score += 0.25
        elif exclamation_count > 3:
            spam_score += 0.1
        
        caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
        if caps_words > 8:
            spam_score += 0.3
        elif caps_words > 5:
            spam_score += 0.15
        
        spam_keywords = [
            'SHOCKING', 'MIRACLE', 'GUARANTEE', 'FREE', 'CLICK HERE',
            'LIMITED TIME', 'ACT NOW', 'URGENT', 'WINNER', '100%',
            'CONGRATULATIONS', 'CLAIM NOW', 'EXCLUSIVE'
        ]
        keyword_matches = sum(1 for keyword in spam_keywords if keyword in text.upper())
        if keyword_matches > 3:
            spam_score += 0.4
        elif keyword_matches > 1:
            spam_score += 0.2
        
        special_count = len(re.findall(r'[!@#$%^&*()]{3,}', text))
        if special_count > 2:
            spam_score += 0.3
        elif special_count > 0:
            spam_score += 0.15
        
        return max(0.0, 1.0 - min(spam_score, 1.0))
    
    def _detect_quality_issues(self, data: Dict[str, Any], individual_scores: Dict[str, float]) -> List[str]:
        issues = []
        
        if individual_scores['completeness'] < 0.5:
            issues.append('Missing important fields')
        
        if individual_scores['length'] < 0.5:
            issues.append('Text length inadequate')
        
        if individual_scores['word_count'] < 0.5:
            issues.append('Insufficient word count')
        
        if individual_scores['language_quality'] < 0.5:
            issues.append('Poor language quality')
        
        if individual_scores['information_density'] < 0.45:
            issues.append('Low information density / repetitive content')
        
        if individual_scores['spam_check'] < 0.6:
            issues.append('Contains spam indicators')
        
        return issues
    
    def score_data(self, data: Dict[str, Any]) -> Dict:
        text = str(data.get('text', ''))
        
        individual_scores = {
            'completeness': self._calculate_text_completeness(data),
            'length': self._calculate_text_length_score(text),
            'word_count': self._calculate_word_count_score(text),
            'language_quality': self._calculate_language_quality(text),
            'information_density': self._calculate_information_density(text),
            'spam_check': self._calculate_spam_indicators(text)
        }
        
        weights = {
            'completeness': 0.12,
            'length': 0.10,
            'word_count': 0.13,
            'language_quality': 0.25,
            'information_density': 0.28,
            'spam_check': 0.12
        }
        
        overall_score = sum(individual_scores[key] * weights[key] for key in individual_scores)
        
        if overall_score >= 0.80:
            quality_level = 'high'
        elif overall_score >= 0.60:
            quality_level = 'medium'
        elif overall_score >= 0.40:
            quality_level = 'low'
        else:
            quality_level = 'very_low'
        
        issues = self._detect_quality_issues(data, individual_scores)
        
        print(f"\n[QUALITY]")
        print(f"Text: {text[:80]}...")
        print(f"Scores: {individual_scores}")
        print(f"Overall: {overall_score:.3f}, Level: {quality_level}")
        
        return {
            'overall_score': overall_score,
            'quality_level': quality_level,
            'individual_scores': individual_scores,
            'issues': issues,
            'recommendation': self._get_recommendation(overall_score, issues)
        }
    
    def _get_recommendation(self, score: float, issues: List[str]) -> str:
        if score >= 0.80:
            return 'Excellent quality - keep'
        elif score >= 0.65:
            return 'Good quality - keep with minor review'
        elif score >= 0.45:
            return 'Fair quality - review before keeping'
        else:
            return 'Poor quality - consider removing'
    
    def score_dataset(self, dataset: List[Dict[str, Any]]) -> Dict:
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