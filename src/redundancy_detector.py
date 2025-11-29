"""
Watchdog AI - Redundancy Detection Module
Detects and removes duplicate/near-duplicate content
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Set
import hashlib


class RedundancyDetector:
    """
    Detects exact and semantic duplicates in text data
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize redundancy detector
        
        Args:
            similarity_threshold: Cosine similarity threshold (0-1) for considering texts as duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words='english'
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not isinstance(text, str):
            return ""
        # Remove extra whitespace and lowercase
        return ' '.join(text.lower().split())
    
    def _create_hash(self, text: str) -> str:
        """Create hash for exact duplicate detection"""
        normalized = self._normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def find_exact_duplicates(self, texts: List[str]) -> Dict:
        """
        Find exact duplicates using hash comparison
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with duplicate information
        """
        hash_map = {}
        duplicates = []
        unique_indices = []
        
        for idx, text in enumerate(texts):
            text_hash = self._create_hash(text)
            
            if text_hash in hash_map:
                # Duplicate found
                duplicates.append({
                    'index': idx,
                    'duplicate_of': hash_map[text_hash],
                    'text': text[:100]
                })
            else:
                # First occurrence
                hash_map[text_hash] = idx
                unique_indices.append(idx)
        
        return {
            'total_items': len(texts),
            'unique_count': len(unique_indices),
            'duplicate_count': len(duplicates),
            'unique_indices': unique_indices,
            'duplicates': duplicates,
            'reduction_percentage': (len(duplicates) / len(texts) * 100) if texts else 0
        }
    
    def find_semantic_duplicates(self, texts: List[str]) -> Dict:
        """
        Find semantic duplicates using TF-IDF and cosine similarity
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with semantic duplicate information
        """
        if len(texts) < 2:
            return {
                'total_items': len(texts),
                'unique_count': len(texts),
                'duplicate_count': 0,
                'unique_indices': list(range(len(texts))),
                'duplicate_pairs': [],
                'reduction_percentage': 0.0
            }
        
        # Filter out empty texts
        valid_texts = [(idx, text) for idx, text in enumerate(texts) if text and len(text.strip()) > 0]
        
        if len(valid_texts) < 2:
            return {
                'total_items': len(texts),
                'unique_count': len(texts),
                'duplicate_count': 0,
                'unique_indices': list(range(len(texts))),
                'duplicate_pairs': [],
                'reduction_percentage': 0.0
            }
        
        indices, valid_text_list = zip(*valid_texts)
        
        # Create TF-IDF vectors
        try:
            tfidf_matrix = self.vectorizer.fit_transform(valid_text_list)
        except ValueError:
            # If vectorization fails, return no duplicates
            return {
                'total_items': len(texts),
                'unique_count': len(texts),
                'duplicate_count': 0,
                'unique_indices': list(range(len(texts))),
                'duplicate_pairs': [],
                'reduction_percentage': 0.0
            }
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicate pairs
        marked_as_duplicate = set()
        duplicate_pairs = []
        
        for i in range(len(similarity_matrix)):
            if i in marked_as_duplicate:
                continue
                
            for j in range(i + 1, len(similarity_matrix)):
                if j in marked_as_duplicate:
                    continue
                    
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    duplicate_pairs.append({
                        'index1': indices[i],
                        'index2': indices[j],
                        'similarity': float(similarity_matrix[i][j]),
                        'text1': valid_text_list[i][:100],
                        'text2': valid_text_list[j][:100]
                    })
                    marked_as_duplicate.add(j)
        
        # Get unique indices
        unique_indices = [idx for idx in range(len(texts)) if idx not in marked_as_duplicate]
        
        return {
            'total_items': len(texts),
            'unique_count': len(unique_indices),
            'duplicate_count': len(marked_as_duplicate),
            'unique_indices': unique_indices,
            'duplicate_pairs': duplicate_pairs,
            'reduction_percentage': (len(marked_as_duplicate) / len(texts) * 100) if texts else 0
        }
    
    def find_duplicates(self, texts: List[str], method: str = 'both') -> Dict:
        """
        Find duplicates using specified method
        
        Args:
            texts: List of text strings
            method: 'exact', 'semantic', or 'both'
            
        Returns:
            Dictionary with duplicate detection results
        """
        if method == 'exact':
            return self.find_exact_duplicates(texts)
        elif method == 'semantic':
            return self.find_semantic_duplicates(texts)
        elif method == 'both':
            # First find exact duplicates
            exact_results = self.find_exact_duplicates(texts)
            
            # Then find semantic duplicates among unique texts
            unique_texts = [texts[i] for i in exact_results['unique_indices']]
            semantic_results = self.find_semantic_duplicates(unique_texts)
            
            # Map semantic results back to original indices
            original_indices_map = {i: exact_results['unique_indices'][i] 
                                   for i in range(len(unique_texts))}
            
            final_unique_indices = [original_indices_map[i] 
                                   for i in semantic_results['unique_indices']]
            
            total_duplicates = (exact_results['duplicate_count'] + 
                              semantic_results['duplicate_count'])
            
            return {
                'total_items': len(texts),
                'unique_count': len(final_unique_indices),
                'duplicate_count': total_duplicates,
                'unique_indices': final_unique_indices,
                'exact_duplicates': exact_results['duplicate_count'],
                'semantic_duplicates': semantic_results['duplicate_count'],
                'reduction_percentage': (total_duplicates / len(texts) * 100) if texts else 0,
                'details': {
                    'exact': exact_results,
                    'semantic': semantic_results
                }
            }
        else:
            raise ValueError(f"Invalid method: {method}. Use 'exact', 'semantic', or 'both'")
    
    def get_unique_texts(self, texts: List[str], method: str = 'both') -> List[str]:
        """
        Get list of unique texts with duplicates removed
        
        Args:
            texts: List of text strings
            method: 'exact', 'semantic', or 'both'
            
        Returns:
            List of unique texts
        """
        results = self.find_duplicates(texts, method=method)
        return [texts[i] for i in results['unique_indices']]
    
    def calculate_redundancy_score(self, texts: List[str]) -> float:
        """
        Calculate overall redundancy score for a dataset
        
        Args:
            texts: List of text strings
            
        Returns:
            Redundancy score (0-1, where 1 is highly redundant)
        """
        if len(texts) <= 1:
            return 0.0
        
        results = self.find_duplicates(texts, method='both')
        return results['duplicate_count'] / results['total_items']
    
    def get_unique_indices(self, texts: List[str], method: str = 'both') -> List[int]:
        """
        Get indices of unique texts (helper method for compatibility)
        
        Args:
            texts: List of text strings
            method: 'exact', 'semantic', or 'both'
            
        Returns:
            List of indices for unique texts
        """
        results = self.find_duplicates(texts, method=method)
        return results['unique_indices']