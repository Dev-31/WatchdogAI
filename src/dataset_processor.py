"""
Dataset Processor - Main Pipeline
Orchestrates all detection and filtering modules
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

from .misinformation_detector import MisinformationDetector
from .quality_scorer import DataQualityScorer
from .redundancy_detector import RedundancyDetector
from .sustainability_tracker import SustainabilityTracker


class DatasetProcessor:
    """
    Main pipeline for processing datasets through Watchdog AI
    Handles CSV, JSON, DataFrames, and lists of dictionaries
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 region: str = 'global_average'):
        """
        Initialize dataset processor
        
        Args:
            similarity_threshold: Threshold for duplicate detection
            region: Energy grid region for sustainability tracking
        """
        self.detector = MisinformationDetector()
        self.scorer = DataQualityScorer()
        self.redundancy = RedundancyDetector(similarity_threshold)
        self.tracker = SustainabilityTracker(region)
    
    def process_dataframe(self,
                         df: pd.DataFrame,
                         text_column: str = 'text',
                         source_column: Optional[str] = None,
                         quality_threshold: float = 0.5,
                         remove_high_risk: bool = True,
                         remove_duplicates: bool = True,
                         verbose: bool = True) -> Dict:
        """
        Process a Pandas DataFrame through the complete pipeline
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            source_column: Name of column containing source (optional)
            quality_threshold: Minimum quality score (0-1)
            remove_high_risk: Remove high-risk misinformation
            remove_duplicates: Remove duplicate entries
            verbose: Print progress messages
            
        Returns:
            Dict with cleaned_df, statistics, and sustainability metrics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🛡️  WATCHDOG AI - PROCESSING DATASET")
            print(f"{'='*70}")
            print(f"Original dataset: {len(df)} rows")
        
        start_time = time.time()
        
        # Validate inputs
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        
        # Calculate original size
        original_size_mb = sum(
            self.tracker.calculate_data_size_mb(str(row))
            for _, row in df.iterrows()
        )
        
        # Initialize results
        results = {
            'original_count': len(df),
            'steps': {},
            'final_df': df.copy(),
            'removed_indices': {
                'misinformation': [],
                'low_quality': [],
                'duplicates': []
            }
        }
        
        current_df = df.copy()
        
        # STEP 1: Misinformation Detection
        if verbose:
            print(f"\n📌 Step 1: Misinformation Detection...")
        
        if remove_high_risk and text_column in current_df.columns:
            misinfo_results = []
            sources = (current_df[source_column].values 
                      if source_column and source_column in current_df.columns 
                      else [None] * len(current_df))
            
            for idx, (text, source) in enumerate(zip(current_df[text_column].values, sources)):
                result = self.detector.analyze_text(
                    str(text), 
                    str(source) if source else None
                )
                misinfo_results.append(result)
                
                if result['risk_level'] == 'high':
                    results['removed_indices']['misinformation'].append(idx)
            
            # Filter out high-risk items
            high_risk_mask = [r['risk_level'] != 'high' for r in misinfo_results]
            current_df = current_df[high_risk_mask].reset_index(drop=True)
            
            results['steps']['misinformation'] = {
                'removed': len(df) - len(current_df),
                'remaining': len(current_df),
                'high_risk_percentage': (len(df) - len(current_df)) / len(df) * 100
            }
            
            if verbose:
                print(f"   ✓ Removed {len(df) - len(current_df)} high-risk items "
                      f"({results['steps']['misinformation']['high_risk_percentage']:.1f}%)")
        
        # STEP 2: Quality Assessment
        if verbose:
            print(f"\n📌 Step 2: Quality Assessment...")
        
        if text_column in current_df.columns:
            quality_scores = []
            low_quality_indices = []
            
            for idx, row in current_df.iterrows():
                score_result = self.scorer.score_data(row.to_dict())
                quality_scores.append(score_result['overall_score'])
                
                if score_result['overall_score'] < quality_threshold:
                    low_quality_indices.append(idx)
                    results['removed_indices']['low_quality'].append(idx)
            
            # Filter out low quality
            before_len = len(current_df)
            current_df = current_df[
                ~current_df.index.isin(low_quality_indices)
            ].reset_index(drop=True)
            
            results['steps']['quality'] = {
                'removed': before_len - len(current_df),
                'remaining': len(current_df),
                'avg_quality_score': np.mean(quality_scores) if quality_scores else 0
            }
            
            if verbose:
                print(f"   ✓ Removed {before_len - len(current_df)} low-quality items")
                print(f"   ✓ Average quality score: {results['steps']['quality']['avg_quality_score']:.3f}")
        
        # STEP 3: Redundancy Detection
        if verbose:
            print(f"\n📌 Step 3: Duplicate Detection...")
        
        if remove_duplicates and text_column in current_df.columns:
            texts = current_df[text_column].astype(str).values.tolist()
            unique_indices = self.redundancy.get_unique_indices(texts)
            
            before_len = len(current_df)
            duplicate_indices = [i for i in range(len(current_df)) if i not in unique_indices]
            results['removed_indices']['duplicates'] = duplicate_indices
            
            current_df = current_df.iloc[unique_indices].reset_index(drop=True)
            
            results['steps']['redundancy'] = {
                'removed': before_len - len(current_df),
                'remaining': len(current_df),
                'redundancy_percentage': (before_len - len(current_df)) / before_len * 100 if before_len > 0 else 0
            }
            
            if verbose:
                print(f"   ✓ Removed {before_len - len(current_df)} duplicates "
                      f"({results['steps']['redundancy']['redundancy_percentage']:.1f}%)")
        
        # STEP 4: Sustainability Impact
        if verbose:
            print(f"\n📌 Step 4: Sustainability Impact...")
        
        final_size_mb = sum(
            self.tracker.calculate_data_size_mb(str(row))
            for _, row in current_df.iterrows()
        )
        
        savings = self.tracker.calculate_savings(original_size_mb, final_size_mb)
        
        # Compile final results
        results['sustainability'] = savings
        results['final_df'] = current_df
        results['final_count'] = len(current_df)
        results['total_removed'] = len(df) - len(current_df)
        results['retention_rate'] = len(current_df) / len(df) * 100 if len(df) > 0 else 0
        results['processing_time'] = time.time() - start_time
        
        # Print summary
        if verbose:
            self._print_summary(results, savings)
        
        return results
    
    def _print_summary(self, results: Dict, savings: Dict):
        """Print processing summary"""
        print(f"\n{'='*70}")
        print(f"📊 SUMMARY")
        print(f"{'='*70}")
        print(f"Original rows:     {results['original_count']:,}")
        print(f"Final rows:        {results['final_count']:,}")
        print(f"Removed:           {results['total_removed']:,} "
              f"({100 - results['retention_rate']:.1f}%)")
        print(f"Retention rate:    {results['retention_rate']:.1f}%")
        
        print(f"\n🌍 ENVIRONMENTAL IMPACT")
        print(f"Data reduced:      {savings['immediate_savings']['data_mb']:.2f} MB "
              f"({savings['immediate_savings']['reduction_percentage']:.1f}%)")
        print(f"Energy saved:      {savings['immediate_savings']['energy_kwh']:.6f} kWh")
        print(f"Carbon saved:      {savings['immediate_savings']['carbon_kg']:.6f} kg CO₂")
        print(f"Annual carbon:     {savings['projected_annual_savings']['carbon_kg']:.3f} kg CO₂/year")
        print(f"Trees equivalent:  {savings['projected_annual_savings']['carbon_trees_equivalent']:.2f} trees/year")
        
        print(f"\n⏱️  Processing time: {results['processing_time']:.2f} seconds")
        print(f"{'='*70}\n")
    
    def process_csv(self, 
                   filepath: str,
                   **kwargs) -> Dict:
        """
        Process a CSV file
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for process_dataframe
            
        Returns:
            Processing results
        """
        print(f"📂 Loading CSV: {filepath}")
        df = pd.read_csv(filepath)
        return self.process_dataframe(df, **kwargs)
    
    def process_json(self,
                    filepath: str,
                    **kwargs) -> Dict:
        """
        Process a JSON file
        
        Args:
            filepath: Path to JSON file
            **kwargs: Additional arguments for process_dataframe
            
        Returns:
            Processing results
        """
        print(f"📂 Loading JSON: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("JSON must contain a list or dict")
        
        return self.process_dataframe(df, **kwargs)
    
    def process_jsonl(self,
                     filepath: str,
                     **kwargs) -> Dict:
        """
        Process a JSONL file (JSON Lines)
        
        Args:
            filepath: Path to JSONL file
            **kwargs: Additional arguments for process_dataframe
            
        Returns:
            Processing results
        """
        print(f"📂 Loading JSONL: {filepath}")
        df = pd.read_json(filepath, lines=True)
        return self.process_dataframe(df, **kwargs)
    
    def process_list(self,
                    data: List[Dict],
                    **kwargs) -> Dict:
        """
        Process a list of dictionaries
        
        Args:
            data: List of dictionaries
            **kwargs: Additional arguments for process_dataframe
            
        Returns:
            Processing results
        """
        df = pd.DataFrame(data)
        return self.process_dataframe(df, **kwargs)
    
    def save_results(self,
                    results: Dict,
                    output_path: str,
                    format: str = 'csv'):
        """
        Save cleaned dataset to file
        
        Args:
            results: Results from process_dataframe
            output_path: Output file path
            format: Output format ('csv', 'json', 'jsonl')
        """
        cleaned_df = results['final_df']
        output_path = Path(output_path)
        
        if format == 'csv':
            cleaned_df.to_csv(output_path, index=False)
        elif format == 'json':
            cleaned_df.to_json(output_path, orient='records', indent=2)
        elif format == 'jsonl':
            cleaned_df.to_json(output_path, orient='records', lines=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"💾 Saved cleaned dataset to: {output_path}")
        
        # Save statistics
        stats_path = output_path.parent / f"{output_path.stem}_stats.json"
        stats = {k: v for k, v in results.items() if k != 'final_df'}
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"📊 Saved statistics to: {stats_path}")


if __name__ == "__main__":
    print("\n🎯 DATASET PROCESSOR TEST\n")
    
    # Create sample data
    sample_data = [
        {"id": 1, "text": "Scientific research shows climate effects.", "source": "nature.com"},
        {"id": 2, "text": "SHOCKING!!! GUARANTEED results!!!", "source": "spam.com"},
        {"id": 3, "text": "Quarterly earnings up 15%.", "source": "company.com"},
        {"id": 4, "text": "Scientific research shows climate effects.", "source": "science.org"},
        {"id": 5, "text": "bad", "source": ""},
    ]
    
    df = pd.DataFrame(sample_data)
    
    # Process
    processor = DatasetProcessor()
    results = processor.process_dataframe(
        df,
        text_column='text',
        source_column='source',
        quality_threshold=0.5,
        remove_high_risk=True,
        remove_duplicates=True
    )
    
    print("\n✅ Cleaned dataset:")
    print(results['final_df'])
