"""
Watchdog AI - MVP Verification Checkpoint Script
Runs all 11 checkpoints and reports PASS/FAIL for each.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from misinformation_detector import MisinformationDetector
from quality_scorer import DataQualityScorer
from redundancy_detector import RedundancyDetector
from sustainability_tracker import SustainabilityTracker
import requests
import json

# Test Data
HIGH_QUALITY_TEXT = "A peer-reviewed study in Nature Climate Change confirms that global renewable energy capacity increased by 50% over the last decade."
LOW_QUALITY_TEXT = "weather good today. sun shine sky blue. i like data. data good. buy coin now gogogo!!!!!!!!!"
MISINFO_TEXT = "DOCTORS ARE HIDING THIS!! One weird fruit cures ALL diseases instantly. Big Pharma banned it! Click here NOW! 100% GUARANTEED MIRACLE."
CREDIBLE_TEXT = HIGH_QUALITY_TEXT
TOXIC_TEXT = "You are absolutely pathetic and nobody listens to you. This entire project is a disaster and everyone knows you're failing."

# Duplicate test data
TEXT_A = "Climate change is accelerating faster than scientists predicted."
TEXT_B = TEXT_A  # Exact duplicate
TEXT_C = "Scientists say global warming is speeding up beyond expectations."  # Semantic duplicate

API_URL = "http://localhost:5000"

def print_result(checkpoint, description, passed, details=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n{'='*60}")
    print(f"CP-{checkpoint}: {description}")
    print(f"Result: {status}")
    if details:
        print(f"Details: {details}")
    print(f"{'='*60}")
    return passed

def run_checkpoints():
    results = []
    
    # Initialize modules
    quality_scorer = DataQualityScorer()
    misinfo_detector = MisinformationDetector()
    redundancy_detector = RedundancyDetector(similarity_threshold=0.7)
    sustainability_tracker = SustainabilityTracker()
    
    print("\n" + "="*60)
    print("WATCHDOG AI - MVP VERIFICATION CHECKPOINTS")
    print("="*60)
    
    # ========== CP-1: Quality Scorer - High Quality ==========
    score_result = quality_scorer.score_data({'text': HIGH_QUALITY_TEXT})
    score = score_result.get('overall_score', 0)
    level = score_result.get('quality_level', 'unknown')
    passed = score >= 0.80 and level.lower() == 'high'
    results.append(print_result(1, "Quality Scorer - High Quality Text", passed, 
                                f"Score: {score:.2f}, Level: {level} (Expected: >=0.80, HIGH)"))
    
    # ========== CP-2: Quality Scorer - Low Quality ==========
    score_result = quality_scorer.score_data({'text': LOW_QUALITY_TEXT})
    score = score_result.get('overall_score', 0)
    level = score_result.get('quality_level', 'unknown')
    passed = score < 0.50 or level.lower() in ['low', 'medium']
    results.append(print_result(2, "Quality Scorer - Low Quality/Spam Text", passed, 
                                f"Score: {score:.2f}, Level: {level} (Expected: <0.50 or LOW/MEDIUM)"))
    
    # ========== CP-3: Misinfo Detector - Clickbait ==========
    misinfo_result = misinfo_detector.analyze_text(MISINFO_TEXT)
    risk = misinfo_result.get('risk_level', 'unknown')
    flags = misinfo_result.get('flags', [])
    passed = risk.lower() == 'high' and len(flags) >= 2
    results.append(print_result(3, "Misinfo Detector - Clickbait/Spam", passed, 
                                f"Risk: {risk}, Flags: {flags} (Expected: HIGH, >=2 flags)"))
    
    # ========== CP-4: Misinfo Detector - Credible ==========
    misinfo_result = misinfo_detector.analyze_text(CREDIBLE_TEXT)
    risk = misinfo_result.get('risk_level', 'unknown')
    flags = misinfo_result.get('flags', [])
    passed = risk.lower() == 'low' and len(flags) == 0
    results.append(print_result(4, "Misinfo Detector - Credible Text", passed, 
                                f"Risk: {risk}, Flags: {flags} (Expected: LOW, 0 flags)"))
    
    # ========== CP-5: Toxicity Detection ==========
    toxic_result = misinfo_detector.analyze_text(TOXIC_TEXT)
    risk = toxic_result.get('risk_level', 'unknown')
    flags = toxic_result.get('flags', [])
    passed = risk.lower() == 'high' and 'toxicity' in flags
    results.append(print_result(5, "Toxicity Detection", passed, 
                                f"Risk: {risk}, Flags: {flags} (Expected: HIGH, toxicity flag)"))
    
    # ========== CP-6: Duplicate Detector - Exact Match ==========
    dup_result = redundancy_detector.find_duplicates([TEXT_A, TEXT_B], method='exact')
    exact_dups = dup_result.get('exact_duplicates', [])
    passed = len(exact_dups) > 0
    results.append(print_result(6, "Duplicate Detector - Exact Match", passed, 
                                f"Exact duplicates found: {len(exact_dups)} (Expected: >0)"))
    
    # ========== CP-7: Duplicate Detector - Semantic Match ==========
    dup_result = redundancy_detector.find_duplicates([TEXT_A, TEXT_C], method='semantic', threshold=0.6)
    semantic_dups = dup_result.get('semantic_duplicates', [])
    passed = len(semantic_dups) > 0
    results.append(print_result(7, "Duplicate Detector - Semantic Match", passed, 
                                f"Semantic duplicates found: {len(semantic_dups)} (Expected: >0)"))
    
    # ========== CP-8: Sustainability Tracker ==========
    savings = sustainability_tracker.calculate_savings(original_data_mb=10.0, optimized_data_mb=6.0)
    carbon = savings.get('carbon_kg', 0)
    energy = savings.get('energy_kwh', 0)
    passed = carbon > 0 and energy > 0
    results.append(print_result(8, "Sustainability Tracker - Savings Calculation", passed, 
                                f"Carbon: {carbon:.6f} kg, Energy: {energy:.6f} kWh (Expected: >0)"))
    
    # ========== CP-9: API /analyze ==========
    try:
        response = requests.post(f"{API_URL}/analyze", json={"text": HIGH_QUALITY_TEXT}, timeout=5)
        passed = response.status_code == 200 and 'risk_level' in response.json()
        details = f"Status: {response.status_code}, Has risk_level: {'risk_level' in response.json()}"
    except Exception as e:
        passed = False
        details = f"Error: {str(e)}"
    results.append(print_result(9, "API /analyze Endpoint", passed, details))
    
    # ========== CP-10: API /process ==========
    try:
        test_data = {"data": [{"text": "Sample text 1"}, {"text": "Sample text 2"}, {"text": "Sample text 1"}]}
        response = requests.post(f"{API_URL}/process", json=test_data, timeout=10)
        result = response.json()
        passed = response.status_code == 200 and 'retention_rate' in result
        details = f"Status: {response.status_code}, Retention Rate: {result.get('retention_rate', 'N/A')}"
    except Exception as e:
        passed = False
        details = f"Error: {str(e)}"
    results.append(print_result(10, "API /process Endpoint", passed, details))
    
    # ========== CP-11: UI Integration (Manual) ==========
    print(f"\n{'='*60}")
    print("CP-11: UI Integration")
    print("Result: [MANUAL CHECK REQUIRED]")
    print("Details: Open frontend.html, test all 5 samples, verify charts update.")
    print(f"{'='*60}")
    
    # ========== SUMMARY ==========
    passed_count = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Passed: {passed_count}/{total} checkpoints")
    print(f"Status: {'ALL CORE CHECKS PASSED!' if passed_count == total else 'SOME CHECKS FAILED'}")
    print("="*60)
    
    return passed_count == total

if __name__ == "__main__":
    success = run_checkpoints()
    sys.exit(0 if success else 1)
