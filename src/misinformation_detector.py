"""
Watchdog AI - Enhanced Misinformation Detection Module
"""

import re
from typing import Dict, List
from collections import Counter


class MisinformationDetector:
    
    def __init__(self):
        self.suspicious_patterns = [
            r'\b(miracle|shocking|unbelievable|secret|they don\'t want you to know)\b',
            r'\b(cure|100%|guaranteed|proven|scientifically)\b.*\b(cancer|disease|lose weight)\b',
            r'\b(doctors hate|big pharma|mainstream media|cover[- ]?up)\b',
            r'\b(conspiracy|illuminati|new world order|deep state)\b',
            r'\b(click here|act now|limited time|don\'t miss|urgent)\b',
            r'\$\d+.*\b(free|earn|make money|work from home)\b',
            r'\b(breaking|leaked|exposed|revealed|truth)\b.*!{2,}',
            r'\b(fake news|media lies|wake up|sheeple)\b',
            r'\b(you won\'t believe|number \d+ will shock you)\b',
            r'\b(share if you agree|repost|like and share)\b'
        ]
        
        self.clickbait_phrases = [
            'you won\'t believe', 'what happens next', 'will shock you',
            'number', 'the reason why', 'this is why', 'here\'s why',
            'the truth about', 'what really happened', 'mind blowing',
            'jaw dropping', 'this will change', 'wait until you see'
        ]
        
        self.low_credibility_indicators = [
            'news.com', 'breaking-news', 'real-truth', 'insider',
            'leaked', 'secret', 'exposed', '.blog', 'wordpress'
        ]
        
        self.vague_words = [
            'good', 'bad', 'nice', 'great', 'fine', 'many', 'some',
            'things', 'stuff', 'very', 'really', 'quite', 'several'
        ]
    
    def detect_generic_content(self, text: str) -> float:
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return 0.0
        
        vague_count = sum(1 for word in words if word in self.vague_words)
        vague_ratio = vague_count / len(words)
        
        sentence_similarity = self._check_sentence_repetition(text)
        
        generic_score = (vague_ratio * 0.6 + sentence_similarity * 0.4)
        
        return min(generic_score * 1.5, 1.0)
    
    def _check_sentence_repetition(self, text: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) <= 1:
            return 0.0
        
        unique_sentences = len(set(sentences))
        repetition_score = 1.0 - (unique_sentences / len(sentences))
        
        return repetition_score
    
    def detect_clickbait(self, text: str) -> float:
        if not text:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        matches = 0
        
        for phrase in self.clickbait_phrases:
            if phrase in text_lower:
                matches += 1
        
        if matches > 0:
            score = min(matches / 2.5, 1.0)
        
        question_count = text.count('?')
        if question_count >= 2:
            score = min(score + 0.25, 1.0)
        
        if re.search(r'\b\d+\s+(reasons|ways|things|secrets|tips|tricks)\b', text_lower):
            score = min(score + 0.35, 1.0)
        
        return score
    
    def detect_excessive_caps(self, text: str) -> float:
        if not text or len(text) < 10:
            return 0.0
        
        uppercase_count = sum(1 for c in text if c.isupper())
        letter_count = sum(1 for c in text if c.isalpha())
        
        if letter_count == 0:
            return 0.0
        
        caps_ratio = uppercase_count / letter_count
        
        if caps_ratio > 0.6:
            return 1.0
        elif caps_ratio > 0.4:
            return 0.8
        elif caps_ratio > 0.25:
            return 0.5
        elif caps_ratio > 0.15:
            return 0.2
        else:
            return 0.0
    
    def detect_excessive_punctuation(self, text: str) -> float:
        if not text:
            return 0.0
        
        exclamation_count = text.count('!')
        repeated_punct = len(re.findall(r'[!?]{2,}', text))
        
        score = 0.0
        
        if exclamation_count >= 5:
            score += 0.6
        elif exclamation_count >= 3:
            score += 0.4
        elif exclamation_count >= 2:
            score += 0.2
        
        if repeated_punct >= 3:
            score += 0.6
        elif repeated_punct >= 1:
            score += 0.3
        
        return min(score, 1.0)
    
    def check_source_credibility(self, source: str) -> float:
        if not source:
            return 0.5
        
        source_lower = source.lower()
        credibility = 0.7
        
        for indicator in self.low_credibility_indicators:
            if indicator in source_lower:
                credibility -= 0.2
        
        high_credibility = [
            'gov', 'edu', 'reuters', 'ap.org', 'bbc', 
            'nytimes', 'nature', 'science', 'ieee'
        ]
        
        for indicator in high_credibility:
            if indicator in source_lower:
                credibility += 0.25
        
        return max(0.0, min(credibility, 1.0))
    
    def analyze_text(self, text: str, source: str = None) -> Dict:
        if not text or not isinstance(text, str):
            return {
                'misinformation_score': 0.5,
                'confidence': 0.0,
                'risk_level': 'medium',
                'flags': ['invalid_input'],
                'explanations': ['Invalid or empty text']
            }
        
        result = {
            'misinformation_score': 0.0,
            'confidence': 0.0,
            'risk_level': 'low',
            'flags': [],
            'explanations': []
        }
        
        text_lower = text.lower()
        
        pattern_matches = []
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                pattern_matches.extend(matches)
        
        pattern_score = min(len(pattern_matches) / 4, 1.0)
        
        if len(pattern_matches) > 0:
            result['flags'].append('suspicious_patterns')
            result['explanations'].append(
                f'Detected {len(pattern_matches)} suspicious patterns'
            )
        
        clickbait_score = self.detect_clickbait(text)
        if clickbait_score > 0.25:
            result['flags'].append('clickbait')
            result['explanations'].append(f'Clickbait detected (score: {clickbait_score:.2f})')
        
        caps_score = self.detect_excessive_caps(text)
        if caps_score > 0.2:
            result['flags'].append('excessive_caps')
            result['explanations'].append(f'Excessive capitalization ({caps_score:.0%})')
        
        punct_score = self.detect_excessive_punctuation(text)
        if punct_score > 0.25:
            result['flags'].append('excessive_punctuation')
            result['explanations'].append(f'Excessive punctuation (score: {punct_score:.2f})')
        
        generic_score = self.detect_generic_content(text)
        if generic_score > 0.4:
            result['flags'].append('generic_content')
            result['explanations'].append(f'Generic/repetitive content detected ({generic_score:.0%})')
        
        source_credibility = self.check_source_credibility(source) if source else 0.5
        source_risk_score = 1.0 - source_credibility
        
        if source and source_risk_score > 0.6:
            result['flags'].append('low_credibility_source')
            result['explanations'].append(f'Low credibility source (score: {source_credibility:.2f})')
        
        scores = {
            'pattern': pattern_score,
            'clickbait': clickbait_score,
            'caps': caps_score,
            'punctuation': punct_score,
            'generic': generic_score,
            'source': source_risk_score
        }
        
        weights = {
            'pattern': 0.25,
            'clickbait': 0.20,
            'caps': 0.15,
            'punctuation': 0.10,
            'generic': 0.20,
            'source': 0.10
        }
        
        result['misinformation_score'] = sum(scores[key] * weights[key] for key in scores)
        
        active_signals = sum(1 for score in scores.values() if score > 0.15)
        signal_strength = sum(scores.values()) / len(scores)
        
        result['confidence'] = min((active_signals / len(scores)) * 0.7 + signal_strength * 0.3, 1.0)
        
        if len(result['flags']) > 0:
            result['misinformation_score'] = max(result['misinformation_score'], 0.30)
            result['confidence'] = max(result['confidence'], 0.40)
        
        if result['misinformation_score'] >= 0.65 or len(result['flags']) >= 4:
            result['risk_level'] = 'high'
            result['misinformation_score'] = max(result['misinformation_score'], 0.75)
        elif result['misinformation_score'] >= 0.40 or len(result['flags']) >= 2:
            result['risk_level'] = 'medium'
        else:
            result['risk_level'] = 'low'
        
        if not result['explanations']:
            result['explanations'].append('No significant misinformation indicators detected')
        
        print(f"\n[ANALYSIS]")
        print(f"Text: {text[:80]}...")
        print(f"Flags: {result['flags']}")
        print(f"Scores: {scores}")
        print(f"Final: {result['misinformation_score']:.3f}, Confidence: {result['confidence']:.3f}")
        
        return result
    
    def batch_analyze(self, texts: List[str], sources: List[str] = None) -> List[Dict]:
        if sources is None:
            sources = [None] * len(texts)
        
        return [self.analyze_text(text, source) for text, source in zip(texts, sources)]