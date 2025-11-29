"""
Watchdog AI - Flask REST API
Provides REST endpoints for data quality and misinformation detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.misinformation_detector import MisinformationDetector
from src.quality_scorer import DataQualityScorer
from src.redundancy_detector import RedundancyDetector
from src.sustainability_tracker import SustainabilityTracker
from src.dataset_processor import DatasetProcessor

app = Flask(__name__)
CORS(app)

# Initialize components
detector = MisinformationDetector()
scorer = DataQualityScorer()
redundancy = RedundancyDetector()
tracker = SustainabilityTracker()
processor = DatasetProcessor()


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'name': 'Watchdog AI API',
        'version': '1.0.0',
        'description': 'AI-powered data quality and misinformation detection',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/analyze': 'Analyze single text (POST)',
            '/analyze/batch': 'Analyze multiple texts (POST)',
            '/quality': 'Score data quality (POST)',
            '/duplicates': 'Find duplicates (POST)',
            '/process': 'Process full dataset (POST)',
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Watchdog AI is running'})


@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Analyze a single text for misinformation
    
    Request body:
    {
        "text": "text to analyze",
        "source": "optional source"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data['text']
        source = data.get('source')
        
        result = detector.analyze_text(text, source)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple texts
    
    Request body:
    {
        "texts": ["text1", "text2", ...],
        "sources": ["source1", "source2", ...] (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing required field: texts'}), 400
        
        texts = data['texts']
        sources = data.get('sources')
        
        results = detector.batch_analyze(texts, sources)
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/quality', methods=['POST'])
def score_quality():
    """
    Score data quality
    
    Request body:
    {
        "data": {"text": "...", "other_fields": "..."},
        "text_field": "text" (optional)
    }
    """
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'Missing required field: data'}), 400
        
        data = request_data['data']
        text_field = request_data.get('text_field', 'text')
        
        result = scorer.score_data(data, text_field)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/duplicates', methods=['POST'])
def find_duplicates():
    """
    Find duplicates in text list
    
    Request body:
    {
        "texts": ["text1", "text2", ...],
        "similarity_threshold": 0.85 (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing required field: texts'}), 400
        
        texts = data['texts']
        threshold = data.get('similarity_threshold', 0.85)
        
        # Create new detector with custom threshold
        dup_detector = RedundancyDetector(similarity_threshold=threshold)
        result = dup_detector.find_duplicates(texts)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
def process_dataset():
    """
    Process a full dataset
    
    Request body:
    {
        "data": [{"text": "...", ...}, ...],
        "text_column": "text" (optional),
        "source_column": "source" (optional),
        "quality_threshold": 0.5 (optional),
        "remove_high_risk": true (optional),
        "remove_duplicates": true (optional)
    }
    """
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'Missing required field: data'}), 400
        
        data = request_data['data']
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Process
        results = processor.process_dataframe(
            df,
            text_column=request_data.get('text_column', 'text'),
            source_column=request_data.get('source_column'),
            quality_threshold=request_data.get('quality_threshold', 0.5),
            remove_high_risk=request_data.get('remove_high_risk', True),
            remove_duplicates=request_data.get('remove_duplicates', True),
            verbose=False
        )
        
        # Convert DataFrame to dict for JSON response
        cleaned_data = results['final_df'].to_dict(orient='records')
        
        # Remove DataFrame from results
        response = {k: v for k, v in results.items() if k != 'final_df'}
        response['cleaned_data'] = cleaned_data
        
        return jsonify({
            'success': True,
            'result': response
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/sustainability', methods=['POST'])
def calculate_sustainability():
    """
    Calculate sustainability impact
    
    Request body:
    {
        "original_size_mb": 100,
        "optimized_size_mb": 65
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'original_size_mb' not in data or 'optimized_size_mb' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        original = data['original_size_mb']
        optimized = data['optimized_size_mb']
        
        result = tracker.calculate_savings(original, optimized)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🛡️  WATCHDOG AI - API SERVER")
    print("="*70)
    print("\nAPI Documentation: http://localhost:5000/")
    print("\nAvailable endpoints:")
    print("  - POST /analyze         - Analyze single text")
    print("  - POST /analyze/batch   - Analyze multiple texts")
    print("  - POST /quality         - Score data quality")
    print("  - POST /duplicates      - Find duplicates")
    print("  - POST /process         - Process full dataset")
    print("  - POST /sustainability  - Calculate impact")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)