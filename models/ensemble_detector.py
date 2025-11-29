"""
Ensemble Detector Model
Combines multiple detection methods with ML models
"""

import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib


class EnsembleDetector:
    """
    Ensemble model combining pattern-based and ML detection
    """
    
    def __init__(self):
        # TF-IDF vectorizer for text features
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Random Forest classifier
        self.rf_model = RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            random_state=42
        )
        
        # Conspiracy/misinformation lexicon
        self.conspiracy_words = set([
            'faked', 'hoax', 'conspiracy', 'coverup', 'secret', 
            'fake', 'staged', 'rigged', 'illuminati', 'deep state',
            'controlled', 'engineered', 'manipulated', 'lie'
        ])
        
        # Pattern detection
        self.suspicious_patterns = [
            r'miracle', r'one weird trick', r'doctors hate',
            r'100%', r'guarantee', r'conspiracy'
        ]
        
        self.trained = False
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from texts
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix
        """
        if not self.trained:
            tfidf_features = self.vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.vectorizer.transform(texts)
        
        # Add pattern-based features
        pattern_features = []
        conspiracy_features = []
        
        for text in texts:
            # Pattern score
            pattern_score = sum(
                bool(re.search(p, text.lower())) 
                for p in self.suspicious_patterns
            )
            pattern_features.append(pattern_score)
            
            # Conspiracy score
            tokens = re.findall(r'\w+', text.lower())
            conspiracy_score = sum(1 for t in tokens if t in self.conspiracy_words)
            conspiracy_features.append(min(conspiracy_score / 3.0, 1.0))
        
        # Combine all features
        additional_features = np.column_stack([pattern_features, conspiracy_features])
        combined_features = np.hstack([tfidf_features.toarray(), additional_features])
        
        return combined_features
    
    def train(self, texts: List[str], labels: List[int]) -> Dict:
        """
        Train the ensemble model
        
        Args:
            texts: Training texts
            labels: Binary labels (0=safe, 1=misinformation)
            
        Returns:
            Training metrics
        """
        print(f"Training ensemble on {len(texts)} examples...")
        
        # Extract features
        X = self.extract_features(texts)
        y = np.array(labels)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.rf_model.fit(X_train, y_train)
        self.trained = True
        
        # Evaluate
        train_score = self.rf_model.score(X_train, y_train)
        test_score = self.rf_model.score(X_test, y_test)
        
        print(f"✓ Training complete!")
        print(f"  Train accuracy: {train_score:.3f}")
        print(f"  Test accuracy: {test_score:.3f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_samples': len(texts)
        }
    
    def predict_proba(self, texts: List[str]) -> List[float]:
        """
        Predict misinformation probability
        
        Args:
            texts: List of texts
            
        Returns:
            List of probabilities (0-1)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.extract_features(texts)
        probas = self.rf_model.predict_proba(X)[:, 1]
        
        return probas.tolist()
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict misinformation labels
        
        Args:
            texts: List of texts
            
        Returns:
            List of labels (0=safe, 1=misinformation)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.extract_features(texts)
        predictions = self.rf_model.predict(X)
        
        return predictions.tolist()
    
    def analyze(self, text: str, weights: Dict[str, float] = None) -> Dict:
        """
        Comprehensive analysis of a single text
        
        Args:
            text: Text to analyze
            weights: Custom weights for scoring components
            
        Returns:
            Analysis result dict
        """
        if weights is None:
            weights = {
                'model': 0.6,
                'pattern': 0.2,
                'conspiracy': 0.2
            }
        
        # Model prediction
        if self.trained:
            model_score = self.predict_proba([text])[0]
        else:
            model_score = 0.5
        
        # Pattern score
        pattern_score = sum(
            bool(re.search(p, text.lower())) 
            for p in self.suspicious_patterns
        ) / len(self.suspicious_patterns)
        
        # Conspiracy score
        tokens = re.findall(r'\w+', text.lower())
        conspiracy_score = sum(1 for t in tokens if t in self.conspiracy_words)
        conspiracy_score = min(conspiracy_score / 3.0, 1.0)
        
        # Combined score
        final_score = (
            weights['model'] * model_score +
            weights['pattern'] * pattern_score +
            weights['conspiracy'] * conspiracy_score
        )
        
        # Risk level
        if final_score > 0.7:
            risk = 'high'
        elif final_score > 0.4:
            risk = 'medium'
        else:
            risk = 'low'
        
        return {
            'text': text,
            'misinformation_score': float(final_score),
            'risk_level': risk,
            'scores': {
                'model': float(model_score),
                'pattern': float(pattern_score),
                'conspiracy': float(conspiracy_score)
            }
        }
    
    def save(self, filepath: str):
        """Save model to file"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.rf_model,
            'trained': self.trained
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        data = joblib.load(filepath)
        self.vectorizer = data['vectorizer']
        self.rf_model = data['model']
        self.trained = data['trained']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("=" * 70)
    print("ENSEMBLE DETECTOR TEST")
    print("=" * 70)
    
    # Create sample training data
    train_texts = [
        "Scientific research shows climate change effects.",
        "SHOCKING!!! Doctors HATE this one weird trick!!!",
        "The quarterly report indicates steady growth.",
        "GUARANTEED miracle cure for everything!!!",
        "Government conspiracy to control minds with 5G.",
        "New study finds correlation between diet and health.",
    ]
    
    train_labels = [0, 1, 0, 1, 1, 0]
    
    # Train model
    detector = EnsembleDetector()
    metrics = detector.train(train_texts, train_labels)
    
    # Test predictions
    print("\nTest predictions:")
    test_texts = [
        "The moon landing was faked by Hollywood.",
        "Coffee may reduce risk of heart disease.",
    ]
    
    for text in test_texts:
        result = detector.analyze(text)
        print(f"\nText: {text}")
        print(f"Score: {result['misinformation_score']:.3f}")
        print(f"Risk: {result['risk_level']}")
        print(f"Components: {result['scores']}")
