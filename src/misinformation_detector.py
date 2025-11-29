"""
Watchdog AI - Misinformation Detection Module
Pattern-based detection with multi-dimensional risk scoring
"""

import re
from typing import Dict, List, Tuple


class MisinformationDetector:
    """
    Detects potential misinformation using pattern matching and heuristics
    """
    
    def __init__(self):
        """Initialize detector with suspicious patterns"""
        
        # Suspicious patterns (regex)
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
        
        # Clickbait phrases
        self.clickbait_phrases = [
            'you won\'t believe', 'what happens next', 'will shock you',
            'number', 'the reason why', 'this is why', 'here\'s why',
            'the truth about', 'what really happened', 'mind blowing',
            'jaw dropping', 'this will change', 'wait until you see'
        ]
        
        # Low credibility source indicators
        self.low_credibility_indicators = [
            'news.com', 'breaking-news', 'real-truth', 'insider',
            'leaked', 'secret', 'exposed', '.blog', 'wordpress'
        ]
    
    def detect_clickbait(self, text: str) -> float:
        """
        Detect clickbait patterns in text
        
        Args:
            text: Input text
            
        Returns:
            Clickbait score (0-1)
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        matches = 0
        
        # Check for clickbait phrases
        for phrase in self.clickbait_phrases:
            if phrase in text_lower:
                matches += 1
        
        # Calculate base score
        if matches > 0:
            score = min(matches / 3, 1.0)  # Normalize: 3+ matches = max score
        
        # Check for question marks (clickbait often poses questions)
        question_count = text.count('?')
        if question_count >= 2:
            score = min(score + 0.2, 1.0)
        
        # Check for numbers in headlines (listicles)
        if re.search(r'\b\d+\s+(reasons|ways|things|secrets|tips|tricks)\b', text_lower):
            score = min(score + 0.3, 1.0)
        
        return score
    
    def detect_excessive_caps(self, text: str) -> float:
        """
        Detect excessive capitalization (shouting)
        
        Args:
            text: Input text
            
        Returns:
            Capitalization score (0-1)
        """
        if not text or len(text) < 10:
            return 0.0
        
        # Count uppercase letters
        uppercase_count = sum(1 for c in text if c.isupper())
        letter_count = sum(1 for c in text if c.isalpha())
        
        if letter_count == 0:
            return 0.0
        
        caps_ratio = uppercase_count / letter_count
        
        # Score based on ratio
        if caps_ratio > 0.5:  # More than 50% caps
            return 1.0
        elif caps_ratio > 0.3:  # 30-50% caps
            return 0.7
        elif caps_ratio > 0.2:  # 20-30% caps
            return 0.4
        else:
            return 0.0
    
    def detect_excessive_punctuation(self, text: str) -> float:
        """
        Detect excessive punctuation marks
        
        Args:
            text: Input text
            
        Returns:
            Punctuation score (0-1)
        """
        if not text:
            return 0.0
        
        # Count exclamation marks
        exclamation_count = text.count('!')
        
        # Count repeated punctuation (!!!, ???, etc.)
        repeated_punct = len(re.findall(r'[!?]{2,}', text))
        
        score = 0.0
        
        # Score based on exclamation marks
        if exclamation_count >= 5:
            score += 0.5
        elif exclamation_count >= 3:
            score += 0.3
        elif exclamation_count >= 2:
            score += 0.1
        
        # Additional score for repeated punctuation
        if repeated_punct >= 3:
            score += 0.5
        elif repeated_punct >= 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def check_source_credibility(self, source: str) -> float:
        """
        Check source credibility
        
        Args:
            source: Source URL or name
            
        Returns:
            Credibility score (0-1, higher is more credible)
        """
        if not source:
            return 0.5  # Neutral for unknown sources
        
        source_lower = source.lower()
        credibility = 0.7  # Start with neutral-positive
        
        # Check for low credibility indicators
        for indicator in self.low_credibility_indicators:
            if indicator in source_lower:
                credibility -= 0.15
        
        # Check for high credibility indicators
        high_credibility = [
            'gov', 'edu', 'reuters', 'ap.org', 'bbc', 
            'nytimes', 'nature', 'science', 'ieee'
        ]
        
        for indicator in high_credibility:
            if indicator in source_lower:
                credibility += 0.2
        
        return max(0.0, min(credibility, 1.0))
    
    def analyze_text(self, text: str, source: str = None) -> Dict:
        """
        Comprehensive misinformation analysis
        
        Args:
            text: Text to analyze
            source: Source of the text (optional)
            
        Returns:
            Dict with scores, risk level, flags, and explanations
        """
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
        
        # 1. Pattern-based detection
        pattern_matches = []
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                pattern_matches.extend(matches)
        
        pattern_score = min(len(pattern_matches) / 5, 1.0)  # Normalize
        
        if len(pattern_matches) > 0:
            result['flags'].append('suspicious_patterns')
            result['explanations'].append(
                f'Contains {len(pattern_matches)} suspicious pattern(s): {", ".join(set(str(m) for m in pattern_matches[:3]))}'
            )
        
        # 2. Clickbait detection
        clickbait_score = self.detect_clickbait(text)
        if clickbait_score > 0.2:
            result['flags'].append('clickbait')
            result['explanations'].append(
                f'Clickbait indicators detected (score: {clickbait_score:.2f})'
            )
        
        # 3. Excessive capitalization
        caps_score = self.detect_excessive_caps(text)
        if caps_score > 0.3:
            result['flags'].append('excessive_caps')
            result['explanations'].append(
                f'Excessive capitalization detected ({caps_score:.0%} of text)'
            )
        
        # 4. Excessive punctuation
        punct_score = self.detect_excessive_punctuation(text)
        if punct_score > 0.2:
            result['flags'].append('excessive_punctuation')
            result['explanations'].append(
                f'Excessive punctuation marks detected (score: {punct_score:.2f})'
            )
        
        # 5. Source credibility
        source_credibility = self.check_source_credibility(source) if source else 0.5
        source_risk_score = 1.0 - source_credibility
        
        if source and source_risk_score > 0.5:
            result['flags'].append('low_credibility_source')
            result['explanations'].append(
                f'Source has low credibility (score: {source_credibility:.2f})'
            )
        
        # Combine all scores with weights
        scores = {
            'pattern': pattern_score,
            'clickbait': clickbait_score,
            'caps': caps_score,
            'punctuation': punct_score,
            'source': source_risk_score
        }
        
        weights = {
            'pattern': 0.30,
            'clickbait': 0.25,
            'caps': 0.15,
            'punctuation': 0.10,
            'source': 0.20
        }
        
        # Calculate weighted score
        result['misinformation_score'] = sum(
            scores[key] * weights[key] for key in scores
        )
        
        # Calculate confidence based on signals
        active_signals = sum(1 for score in scores.values() if score > 0.1)
        signal_strength = sum(scores.values()) / len(scores)
        
        result['confidence'] = min(
            (active_signals / len(scores)) * 0.6 + signal_strength * 0.4, 
            1.0
        )
        
        # Ensure minimum scores if flags are present
        if len(result['flags']) > 0:
            result['misinformation_score'] = max(result['misinformation_score'], 0.25)
            result['confidence'] = max(result['confidence'], 0.35)
        
        # Determine risk level
        if result['misinformation_score'] >= 0.6 or len(result['flags']) >= 3:
            result['risk_level'] = 'high'
            result['misinformation_score'] = max(result['misinformation_score'], 0.7)
        elif result['misinformation_score'] >= 0.35 or len(result['flags']) >= 2:
            result['risk_level'] = 'medium'
        else:
            result['risk_level'] = 'low'
        
        # Add default explanation if none exist
        if not result['explanations']:
            result['explanations'].append('No significant misinformation indicators detected')
        
        # Debug logging
        print(f"\n[MISINFO DEBUG]")
        print(f"Text: {text[:100]}...")
        print(f"Flags: {result['flags']}")
        print(f"Individual scores: {scores}")
        print(f"Final score: {result['misinformation_score']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Risk level: {result['risk_level']}")
        
        return result
    
    def batch_analyze(self, texts: List[str], sources: List[str] = None) -> List[Dict]:
        """
        Analyze multiple texts
        
        Args:
            texts: List of texts to analyze
            sources: Optional list of sources
            
        Returns:
            List of analysis results
        """
        if sources is None:
            sources = [None] * len(texts)
        
        return [
            self.analyze_text(text, source)
            for text, source in zip(texts, sources)
        ]