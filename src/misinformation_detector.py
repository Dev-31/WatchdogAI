"""
Misinformation Detection Module
Detects suspicious patterns, clickbait, and low-credibility sources
"""

import re
from typing import Dict, List


class MisinformationDetector:
    """Detect misinformation using pattern matching and source credibility analysis"""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'\b(guaranteed|miracle|secret|doctors hate|one weird trick)\b',
            r'\b(100%|absolutely|never|always|everyone|no one)\b.*\b(guaranteed|proven|works)\b',
            r'\b(conspiracy|coverup|they don\'t want you to know)\b',
            r'[!]{3,}',  # Excessive exclamation marks
            r'[A-Z]{10,}',  # Excessive capitalization
        ]
        
        self.credible_sources = {
            'academic': ['edu', 'scholar', 'journal', 'research', 'university'],
            'government': ['gov', 'official', 'state', 'federal'],
            'verified': ['reuters', 'ap', 'bbc', 'nature', 'science', 'nejm']
        }
        
        self.clickbait_indicators = [
            r'you won\'t believe',
            r'number \d+ will shock you',
            r'what happens next',
            r'doctors hate',
            r'one simple trick',
            r'this is why',
            r'the truth about',
        ]
    
    def check_source_credibility(self, source: str) -> float:
        """
        Score source credibility (0-1)
        
        Args:
            source: Source URL or name
            
        Returns:
            Credibility score (0=low, 1=high)
        """
        if not source:
            return 0.5
        
        source_lower = source.lower()
        score = 0.5
        
        for category, keywords in self.credible_sources.items():
            if any(kw in source_lower for kw in keywords):
                score += 0.15
        
        return min(score, 1.0)
    
    def detect_clickbait(self, text: str) -> float:
        """
        Detect clickbait patterns (0-1)
        
        Args:
            text: Text to analyze
            
        Returns:
            Clickbait score (0=none, 1=high)
        """
        if not text:
            return 0.0
        
        score = sum(
            1 for pattern in self.clickbait_indicators
            if re.search(pattern, text.lower())
        )
        return min(score / len(self.clickbait_indicators), 1.0)
    
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
        
        # 1. Pattern-based detection
        pattern_score = sum(
            1 for pattern in self.suspicious_patterns
            if re.search(pattern, text.lower())
        ) / len(self.suspicious_patterns)
        
        if pattern_score > 0:
            result['flags'].append('suspicious_patterns')
            result['explanations'].append(
                f'Contains {int(pattern_score * len(self.suspicious_patterns))} suspicious patterns'
            )
        
        # 2. Clickbait detection
        clickbait_score = self.detect_clickbait(text)
        if clickbait_score > 0.3:
            result['flags'].append('clickbait')
            result['explanations'].append(
                f'High clickbait score: {clickbait_score:.2f}'
            )
        
        # 3. Source credibility
        source_score = 1.0 - self.check_source_credibility(source) if source else 0.5
        if source_score > 0.6:
            result['flags'].append('low_credibility_source')
            result['explanations'].append('Source has low credibility indicators')
        
        # Combine scores (weighted average)
        weights = [0.4, 0.3, 0.3]
        scores = [pattern_score, clickbait_score, source_score]
        result['misinformation_score'] = sum(w * s for w, s in zip(weights, scores))
        
        # Calculate confidence based on number of signals
        signals = len([s for s in scores if s > 0])
        result['confidence'] = min(signals / 3, 1.0)
        
        # Determine risk level
        if result['misinformation_score'] > 0.7:
            result['risk_level'] = 'high'
        elif result['misinformation_score'] > 0.4:
            result['risk_level'] = 'medium'
        else:
            result['risk_level'] = 'low'
        
        return result
    
    def batch_analyze(self, texts: List[str], sources: List[str] = None) -> List[Dict]:
        """
        Analyze multiple texts efficiently
        
        Args:
            texts: List of texts to analyze
            sources: List of sources (optional)
            
        Returns:
            List of analysis results
        """
        if sources is None:
            sources = [None] * len(texts)
        
        return [
            self.analyze_text(text, source)
            for text, source in zip(texts, sources)
        ]


if __name__ == "__main__":
    # Example usage
    detector = MisinformationDetector()
    
    test_cases = [
        ("Scientific research shows climate change effects.", "nature.com"),
        ("SHOCKING!!! Doctors HATE this ONE WEIRD TRICK!!!", "spam-blog.com"),
        ("The meeting is scheduled for 3 PM.", None),
    ]
    
    print("=" * 70)
    print("MISINFORMATION DETECTOR TEST")
    print("=" * 70)
    
    for text, source in test_cases:
        result = detector.analyze_text(text, source)
        print(f"\nText: {text[:50]}...")
        print(f"Source: {source}")
        print(f"Risk Level: {result['risk_level'].upper()}")
        print(f"Score: {result['misinformation_score']:.3f}")
        print(f"Flags: {', '.join(result['flags']) if result['flags'] else 'None'}")
