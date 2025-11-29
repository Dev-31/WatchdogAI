#!/usr/bin/env python3
"""
Watchdog AI - One-Click Execution Script
Runs the complete pipeline with sample data
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("🛡️  WATCHDOG AI - ONE-CLICK DEMO")
print("="*70 + "\n")

# Step 1: Check environment
print("📋 Step 1: Checking environment...")
try:
    from src.misinformation_detector import MisinformationDetector
    from src.quality_scorer import DataQualityScorer
    from src.redundancy_detector import RedundancyDetector
    from src.sustainability_tracker import SustainabilityTracker
    from src.dataset_processor import DatasetProcessor
    print("   ✓ All modules imported successfully\n")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    print("   Please ensure all files are in the correct directories.\n")
    sys.exit(1)

# Step 2: Create sample dataset
print("📋 Step 2: Creating sample dataset...")
sample_data = pd.DataFrame([
    {
        "id": 1,
        "text": "Scientific research published in Nature shows significant climate change evidence backed by peer-reviewed studies.",
        "source": "nature.com",
        "category": "science"
    },
    {
        "id": 2,
        "text": "SHOCKING!!! Doctors HATE this ONE WEIRD TRICK that GUARANTEES weight loss!!!",
        "source": "spam-blog.com",
        "category": "health"
    },
    {
        "id": 3,
        "text": "The quarterly earnings report indicates a 15% increase in revenue year-over-year.",
        "source": "company.com",
        "category": "business"
    },
    {
        "id": 4,
        "text": "Scientific research published in Nature shows significant climate change evidence backed by peer-reviewed studies.",
        "source": "science.org",
        "category": "science"
    },  # Duplicate
    {
        "id": 5,
        "text": "bad stuff",
        "source": "",
        "category": "unknown"
    },  # Low quality
    {
        "id": 6,
        "text": "New AI algorithms improve medical diagnosis accuracy by 23% according to Stanford study.",
        "source": "stanford.edu",
        "category": "technology"
    },
    {
        "id": 7,
        "text": "MIRACLE CURE!!! 100% GUARANTEED to work EVERY TIME!!!",
        "source": "unknown.com",
        "category": "health"
    },  # Misinformation
    {
        "id": 8,
        "text": "Local community center announces summer programs for youth education and recreation.",
        "source": "local-news.com",
        "category": "community"
    },
])

print(f"   ✓ Created dataset with {len(sample_data)} rows\n")

# Step 3: Individual Component Tests
print("📋 Step 3: Testing individual components...\n")

# Test 3.1: Misinformation Detection
print("   3.1 Misinformation Detection")
detector = MisinformationDetector()
test_text = "SHOCKING!!! Doctors HATE this trick!!!"
result = detector.analyze_text(test_text, "spam.com")
print(f"       Text: {test_text[:50]}...")
print(f"       Risk Level: {result['risk_level'].upper()}")
print(f"       Score: {result['misinformation_score']:.3f}\n")

# Test 3.2: Quality Scoring
print("   3.2 Quality Scoring")
scorer = DataQualityScorer()
test_data = {"text": "AI has revolutionized healthcare. Studies show improvement.", "title": "AI Article"}
result = scorer.score_data(test_data)
print(f"       Quality Level: {result['quality_level'].upper()}")
print(f"       Score: {result['overall_score']:.3f}\n")

# Test 3.3: Redundancy Detection
print("   3.3 Redundancy Detection")
redundancy = RedundancyDetector()
test_texts = [
    "Climate change is urgent.",
    "Weather was sunny.",
    "Climate change is urgent.",  # Duplicate
]
result = redundancy.find_duplicates(test_texts)
print(f"       Total items: {result['total_items']}")
print(f"       Unique items: {result['unique_count']}")
print(f"       Duplicates: {result['duplicate_count']}\n")

# Test 3.4: Sustainability Tracking
print("   3.4 Sustainability Tracking")
tracker = SustainabilityTracker()
savings = tracker.calculate_savings(100, 65)
print(f"       Data reduction: 35%")
print(f"       Carbon saved: {savings['immediate_savings']['carbon_kg']:.6f} kg CO₂\n")

# Step 4: Process Full Dataset
print("📋 Step 4: Processing complete dataset through pipeline...\n")
processor = DatasetProcessor()

results = processor.process_dataframe(
    sample_data,
    text_column='text',
    source_column='source',
    quality_threshold=0.5,
    remove_high_risk=True,
    remove_duplicates=True,
    verbose=True
)

# Step 5: Save Results
print("\n📋 Step 5: Saving results...")

# Create output directory
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# Save cleaned data
output_csv = output_dir / "cleaned_demo_data.csv"
results['final_df'].to_csv(output_csv, index=False)
print(f"   ✓ Cleaned data saved to: {output_csv}")

# Save statistics
import json
stats_file = output_dir / "demo_stats.json"
stats = {k: v for k, v in results.items() if k != 'final_df'}
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2, default=str)
print(f"   ✓ Statistics saved to: {stats_file}\n")

# Step 6: Display Final Results
print("📋 Step 6: Final Results\n")
print("   Cleaned Dataset Preview:")
print("   " + "-"*66)
for _, row in results['final_df'].head(3).iterrows():
    print(f"   ID {row['id']}: {row['text'][:50]}... [{row['category']}]")
print("   " + "-"*66 + "\n")

# Step 7: Run Tests (optional)
print("📋 Step 7: Running unit tests...\n")
try:
    import pytest
    test_dir = project_root / "tests"
    if test_dir.exists():
        print("   Running tests...")
        result = pytest.main([str(test_dir), "-v", "--tb=short"])
        if result == 0:
            print("\n   ✓ All tests passed!\n")
        else:
            print("\n   ⚠️  Some tests failed (see above)\n")
    else:
        print("   ⚠️  Test directory not found, skipping tests\n")
except ImportError:
    print("   ⚠️  pytest not installed, skipping tests")
    print("   Install with: pip install pytest\n")

# Step 8: API Demo (optional)
print("📋 Step 8: API Information\n")
print("   To start the REST API server:")
print("   $ python api/app.py")
print("   ")
print("   The API will be available at: http://localhost:5000")
print("   ")
print("   Example API usage:")
print("   $ curl -X POST http://localhost:5000/analyze \\")
print("        -H 'Content-Type: application/json' \\")
print("        -d '{\"text\": \"Your text here\"}'\n")

# Summary
print("="*70)
print("✅ WATCHDOG AI DEMO COMPLETE")
print("="*70)
print(f"""
Summary:
  • Original dataset: {results['original_count']} rows
  • Cleaned dataset: {results['final_count']} rows
  • Items removed: {results['total_removed']} ({100 - results['retention_rate']:.1f}%)
  • Processing time: {results['processing_time']:.2f} seconds
  
  • Data reduction: {results['sustainability']['immediate_savings']['reduction_percentage']:.1f}%
  • Carbon saved: {results['sustainability']['immediate_savings']['carbon_kg']:.6f} kg CO₂

Next Steps:
  1. Process your own data: python main.py process --input your_data.csv --output clean.csv
  2. Start the API: python api/app.py
  3. Run tests: pytest tests/
  4. View documentation: Check README.md and SETUP_GUIDE.md

""")

print("="*70 + "\n")
