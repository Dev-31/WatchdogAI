"""
Watchdog AI - Flask REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.misinformation_detector import MisinformationDetector
from src.quality_scorer import DataQualityScorer
from src.redundancy_detector import RedundancyDetector
from src.sustainability_tracker import SustainabilityTracker
from src.dataset_processor import DatasetProcessor

app = Flask(__name__)
CORS(app)

detector = MisinformationDetector()
scorer = DataQualityScorer()
redundancy = RedundancyDetector()
tracker = SustainabilityTracker()
processor = DatasetProcessor()


@app.route('/')
def home():
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
    return jsonify({'status': 'healthy', 'message': 'Watchdog AI is running'})


@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        
        print(f"[DEBUG] /analyze received: {data}")
        
        text = data.get('text', '')
        source = data.get('source', None)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"[DEBUG] Analyzing: {text[:100]}...")
        
        result = detector.analyze_text(text, source)
        
        print(f"[DEBUG] Result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"[ERROR] /analyze: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    try:
        data = request.json
        
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
        print(f"[ERROR] /analyze/batch: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/quality', methods=['POST'])
def quality_check():
    try:
        data = request.json
        
        print(f"[DEBUG] /quality received: {data}")
        
        if 'data' in data:
            item_to_score = data['data']
        elif 'text' in data:
            item_to_score = {'text': data['text']}
            if 'source' in data:
                item_to_score['source'] = data['source']
        else:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        print(f"[DEBUG] Scoring: {item_to_score.get('text', '')[:100]}...")
        
        result = scorer.score_data(item_to_score)
        
        print(f"[DEBUG] Quality result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"[ERROR] /quality: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/duplicates', methods=['POST'])
def find_duplicates():
    try:
        data = request.json
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing required field: texts'}), 400
        
        texts = data['texts']
        threshold = data.get('similarity_threshold', 0.85)
        
        dup_detector = RedundancyDetector(similarity_threshold=threshold)
        result = dup_detector.find_duplicates(texts)
        
        return jsonify({'success': True, 'result': result})
    
    except Exception as e:
        print(f"[ERROR] /duplicates: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
def process_dataset():
    try:
        request_data = request.json
        
        print(f"[DEBUG] /process received {len(request_data.get('data', []))} items")
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'Missing required field: data'}), 400
        
        data = request_data['data']
        df = pd.DataFrame(data)
        
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        
        results = processor.process_dataframe(
            df,
            text_column=request_data.get('text_column', 'text'),
            source_column=request_data.get('source_column'),
            quality_threshold=request_data.get('quality_threshold', 0.5),
            remove_high_risk=request_data.get('remove_high_risk', True),
            remove_duplicates=request_data.get('remove_duplicates', True),
            verbose=False
        )
        
        print(f"[DEBUG] Removed: {results.get('total_removed')}")
        
        cleaned_data = results['final_df'].to_dict(orient='records')
        
        response = {
            'original_count': results['original_count'],
            'final_count': results['final_count'],
            'total_removed': results['total_removed'],
            'retention_rate': results['retention_rate'],
            'steps': results.get('steps', {}),
            'sustainability': results.get('sustainability', {}),
            'cleaned_data': cleaned_data
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"[ERROR] /process: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/sustainability', methods=['POST'])
def calculate_sustainability():
    try:
        data = request.json
        
        if not data or 'original_size_mb' not in data or 'optimized_size_mb' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        original = data['original_size_mb']
        optimized = data['optimized_size_mb']
        
        result = tracker.calculate_savings(original, optimized)
        
        return jsonify({'success': True, 'result': result})
    
    except Exception as e:
        print(f"[ERROR] /sustainability: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🛡️  WATCHDOG AI - API SERVER")
    print("="*70)
    print("\nAPI: http://localhost:5000/")
    print("\nEndpoints:")
    print("  POST /analyze")
    print("  POST /analyze/batch")
    print("  POST /quality")
    print("  POST /duplicates")
    print("  POST /process")
    print("  POST /sustainability")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)