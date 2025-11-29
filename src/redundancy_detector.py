"""
Watchdog AI - Enhanced Redundancy Detection Module
"""

import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Set
import re


class RedundancyDetector:
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _compute_hash(self, text: str) -> str:
        normalized = self._normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def find_exact_duplicates(self, texts: List[str]) -> Dict:
        if not texts:
            return {'unique_indices': [], 'duplicate_groups': {}, 'duplicate_count': 0}
        
        hash_map = {}
        duplicate_groups = {}
        
        for idx, text in enumerate(texts):
            text_hash = self._compute_hash(text)
            
            if text_hash in hash_map:
                if text_hash not in duplicate_groups:
                    duplicate_groups[text_hash] = [hash_map[text_hash]]
                duplicate_groups[text_hash].append(idx)
            else:
                hash_map[text_hash] = idx
        
        unique_indices = list(hash_map.values())
        duplicate_count = len(texts) - len(unique_indices)
        
        return {
            'unique_indices': sorted(unique_indices),
            'duplicate_groups': duplicate_groups,
            'duplicate_count': duplicate_count,
            'unique_count': len(unique_indices)
        }
    
    def find_semantic_duplicates(self, texts: List[str], threshold: float = None) -> Dict:
        if threshold is None:
            threshold = self.similarity_threshold
        
        if not texts or len(texts) < 2:
            return {
                'unique_indices': list(range(len(texts))),
                'similar_pairs': [],
                'duplicate_count': 0
            }
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            seen = set()
            duplicate_indices = set()
            similar_pairs = []
            
            for i in range(len(texts)):
                if i in seen:
                    continue
                
                for j in range(i + 1, len(texts)):
                    if j in seen:
                        continue
                    
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= threshold:
                        similar_pairs.append({
                            'index1': i,
                            'index2': j,
                            'similarity': float(similarity)
                        })
                        
                        duplicate_indices.add(j)
                        seen.add(j)
            
            unique_indices = [i for i in range(len(texts)) if i not in duplicate_indices]
            
            return {
                'unique_indices': unique_indices,
                'similar_pairs': similar_pairs,
                'duplicate_count': len(duplicate_indices),
                'unique_count': len(unique_indices)
            }
        
        except Exception as e:
            print(f"[ERROR] Semantic duplicate detection failed: {str(e)}")
            return {
                'unique_indices': list(range(len(texts))),
                'similar_pairs': [],
                'duplicate_count': 0
            }
    
    def find_duplicates(self, texts: List[str], method: str = 'both', threshold: float = None) -> Dict:
        if threshold is None:
            threshold = self.similarity_threshold
        
        if not texts:
            return {
                'method': method,
                'original_count': 0,
                'unique_count': 0,
                'duplicate_count': 0,
                'reduction_percentage': 0.0,
                'unique_indices': []
            }
        
        if method == 'exact':
            result = self.find_exact_duplicates(texts)
            unique_indices = result['unique_indices']
        
        elif method == 'semantic':
            result = self.find_semantic_duplicates(texts, threshold)
            unique_indices = result['unique_indices']
        
        elif method == 'both':
            exact_result = self.find_exact_duplicates(texts)
            exact_unique_indices = set(exact_result['unique_indices'])
            
            filtered_texts = [texts[i] for i in exact_unique_indices]
            index_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(exact_unique_indices)}
            
            if len(filtered_texts) > 1:
                semantic_result = self.find_semantic_duplicates(filtered_texts, threshold)
                final_unique_indices = [index_mapping[i] for i in semantic_result['unique_indices']]
            else:
                final_unique_indices = list(exact_unique_indices)
            
            unique_indices = sorted(final_unique_indices)
            
            result = {
                'exact_duplicates': exact_result['duplicate_count'],
                'semantic_duplicates': len(exact_unique_indices) - len(unique_indices),
                'unique_indices': unique_indices
            }
        
        else:
            raise ValueError(f"Invalid method: {method}. Use 'exact', 'semantic', or 'both'")
        
        duplicate_count = len(texts) - len(unique_indices)
        reduction_percentage = (duplicate_count / len(texts)) * 100 if texts else 0
        
        print(f"\n[REDUNDANCY]")
        print(f"Original: {len(texts)}, Unique: {len(unique_indices)}, Removed: {duplicate_count}")
        print(f"Reduction: {reduction_percentage:.1f}%")
        
        return {
            'method': method,
            'original_count': len(texts),
            'unique_count': len(unique_indices),
            'duplicate_count': duplicate_count,
            'reduction_percentage': reduction_percentage,
            'unique_indices': unique_indices,
            **result
        }