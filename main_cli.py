"""
Watchdog AI - Main CLI Entry Point
Command-line interface for processing datasets
"""

import argparse
import sys
from pathlib import Path

from src.dataset_processor import DatasetProcessor


def process_command(args):
    """Process dataset through pipeline"""
    print(f"\n🛡️  WATCHDOG AI - Processing Dataset")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}\n")
    
    # Initialize processor
    processor = DatasetProcessor(
        similarity_threshold=args.similarity,
        region=args.region
    )
    
    # Determine input format
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Process based on file extension
    if input_path.suffix == '.csv':
        results = processor.process_csv(
            args.input,
            text_column=args.text_column,
            source_column=args.source_column,
            quality_threshold=args.quality_threshold,
            remove_high_risk=not args.keep_high_risk,
            remove_duplicates=not args.keep_duplicates
        )
    elif input_path.suffix == '.json':
        results = processor.process_json(
            args.input,
            text_column=args.text_column,
            source_column=args.source_column,
            quality_threshold=args.quality_threshold,
            remove_high_risk=not args.keep_high_risk,
            remove_duplicates=not args.keep_duplicates
        )
    elif input_path.suffix == '.jsonl':
        results = processor.process_jsonl(
            args.input,
            text_column=args.text_column,
            source_column=args.source_column,
            quality_threshold=args.quality_threshold,
            remove_high_risk=not args.keep_high_risk,
            remove_duplicates=not args.keep_duplicates
        )
    else:
        print(f"❌ Error: Unsupported file format: {input_path.suffix}")
        print("Supported formats: .csv, .json, .jsonl")
        sys.exit(1)
    
    # Save results
    output_path = Path(args.output)
    output_format = output_path.suffix[1:]  # Remove the dot
    
    processor.save_results(results, args.output, format=output_format)
    
    print("\n✅ Processing complete!")


def analyze_command(args):
    """Analyze a single text"""
    from src.misinformation_detector import MisinformationDetector
    from src.quality_scorer import DataQualityScorer
    
    print(f"\n🛡️  WATCHDOG AI - Text Analysis")
    print(f"Text: {args.text[:100]}...")
    
    # Misinformation detection
    detector = MisinformationDetector()
    misinfo_result = detector.analyze_text(args.text, args.source)
    
    print(f"\n📊 Misinformation Analysis:")
    print(f"   Risk Level: {misinfo_result['risk_level'].upper()}")
    print(f"   Score: {misinfo_result['misinformation_score']:.3f}")
    print(f"   Confidence: {misinfo_result['confidence']:.3f}")
    if misinfo_result['flags']:
        print(f"   Flags: {', '.join(misinfo_result['flags'])}")
    if misinfo_result['explanations']:
        print(f"   Explanations:")
        for exp in misinfo_result['explanations']:
            print(f"      - {exp}")
    
    # Quality scoring
    scorer = DataQualityScorer()
    quality_result = scorer.score_data({'text': args.text})
    
    print(f"\n📊 Quality Analysis:")
    print(f"   Quality Level: {quality_result['quality_level'].upper()}")
    print(f"   Overall Score: {quality_result['overall_score']:.3f}")
    print(f"   Individual Scores:")
    for key, value in quality_result['individual_scores'].items():
        print(f"      - {key}: {value:.3f}")


def stats_command(args):
    """Show statistics for a processed dataset"""
    import json
    
    stats_path = Path(args.stats_file)
    if not stats_path.exists():
        print(f"❌ Error: Stats file not found: {args.stats_file}")
        sys.exit(1)
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    print(f"\n📊 DATASET STATISTICS")
    print(f"{'='*70}")
    print(f"Original count: {stats['original_count']:,}")
    print(f"Final count: {stats['final_count']:,}")
    print(f"Total removed: {stats['total_removed']:,}")
    print(f"Retention rate: {stats['retention_rate']:.1f}%")
    
    if 'steps' in stats:
        print(f"\n📌 Step-by-Step Results:")
        for step, data in stats['steps'].items():
            print(f"   {step.title()}:")
            print(f"      Removed: {data['removed']}")
            print(f"      Remaining: {data['remaining']}")
    
    if 'sustainability' in stats:
        sust = stats['sustainability']
        print(f"\n🌍 Sustainability Impact:")
        print(f"   Data saved: {sust['immediate_savings']['data_mb']:.2f} MB")
        print(f"   Energy saved: {sust['immediate_savings']['energy_kwh']:.6f} kWh")
        print(f"   Carbon saved: {sust['immediate_savings']['carbon_kg']:.6f} kg CO₂")


def main():
    parser = argparse.ArgumentParser(
        description='Watchdog AI - Data Quality & Misinformation Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a CSV file
  python main.py process --input data.csv --output cleaned.csv
  
  # Process with custom settings
  python main.py process --input data.csv --output cleaned.csv \\
      --text-column content --quality-threshold 0.6
  
  # Analyze a single text
  python main.py analyze --text "Your text here" --source "example.com"
  
  # View statistics
  python main.py stats --stats-file cleaned_stats.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a dataset')
    process_parser.add_argument('--input', '-i', required=True, help='Input file path')
    process_parser.add_argument('--output', '-o', required=True, help='Output file path')
    process_parser.add_argument('--text-column', default='text', help='Name of text column')
    process_parser.add_argument('--source-column', default=None, help='Name of source column')
    process_parser.add_argument('--quality-threshold', type=float, default=0.5, 
                               help='Minimum quality score (0-1)')
    process_parser.add_argument('--similarity', type=float, default=0.85,
                               help='Similarity threshold for duplicates (0-1)')
    process_parser.add_argument('--region', default='global_average',
                               choices=['global_average', 'us_average', 'eu_average', 'renewable'],
                               help='Energy grid region')
    process_parser.add_argument('--keep-high-risk', action='store_true',
                               help='Keep high-risk misinformation items')
    process_parser.add_argument('--keep-duplicates', action='store_true',
                               help='Keep duplicate items')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single text')
    analyze_parser.add_argument('--text', '-t', required=True, help='Text to analyze')
    analyze_parser.add_argument('--source', '-s', default=None, help='Source of the text')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument('--stats-file', required=True, help='Path to stats JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        process_command(args)
    elif args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'stats':
        stats_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
