"""
Image Segmentation Flask API
=============================
REST API for K-means image segmentation with MLflow tracking.

Endpoints:
- GET  /                    - Web UI
- POST /api/segment         - Segment an image
- POST /api/find-optimal-k  - Find optimal K value
- GET  /api/sample-images   - List sample images
- POST /api/models/save     - Save model
- POST /api/models/load     - Load model
- GET  /api/models          - List saved models
- GET  /api/mlflow/runs     - Get MLflow experiment runs
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os
import io # images  convert to bytes
import base64 # encoding and decoding 
from datetime import datetime

from segmentation_engine import (
    ImageSegmenter,
    find_optimal_k,
    create_visualization,
    image_to_base64,
    fig_to_base64,
    get_sklearn_image,
    resize_image,
    MLFLOW_AVAILABLE
)

# Try to import MLflow for API endpoints
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['SAMPLE_FOLDER'] = 'sample_images'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'}

# Ensure directories exist
for folder in ['uploads', 'models', 'sample_images', 'mlruns', 'artifacts']:
    os.makedirs(folder, exist_ok=True)

# Store current segmenter in memory
current_segmenter = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_base64_image(base64_string):
    """Decode base64 image to PIL Image."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


# =============================================================================
# WEB ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/sample_images/<filename>')
def serve_sample_image(filename):
    """Serve sample images."""
    return send_from_directory(app.config['SAMPLE_FOLDER'], filename)


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/api/segment', methods=['POST'])
def segment_image():
    """
    Segment an image using K-means.
    
    Request JSON:
        - image: base64 encoded image
        - n_clusters: number of segments (default: 4)
        - preprocessing: 'none', 'enhance', 'normalize' (default: 'enhance')
        - track_mlflow: track with MLflow (default: true)
        - max_size: max image dimension (default: 500)
    
    Response:
        - success: boolean
        - original_image: base64
        - segmented_image: base64
        - visualization: base64 (full analysis plot)
        - metrics: clustering metrics
        - cluster_colors: RGB colors for each cluster
        - mlflow_run_id: MLflow run ID (if tracking enabled)
        - mlflow_available: whether MLflow is installed
    """
    global current_segmenter
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get parameters
        n_clusters = int(data.get('n_clusters', 4))
        preprocessing = data.get('preprocessing', 'enhance')
        max_size = int(data.get('max_size', 500))
        
        # Handle track_mlflow - ensure it's a boolean
        track_mlflow_param = data.get('track_mlflow', True)
        if isinstance(track_mlflow_param, str):
            track_mlflow = track_mlflow_param.lower() in ('true', '1', 'yes')
        else:
            track_mlflow = bool(track_mlflow_param)
        
        # Only track if MLflow is available AND user wants tracking
        should_track = track_mlflow and MLFLOW_AVAILABLE
        
        print(f"Segmentation request: n_clusters={n_clusters}, preprocessing={preprocessing}, track_mlflow={track_mlflow}, mlflow_available={MLFLOW_AVAILABLE}")
        
        # Images are usually converted into Base64 strings (a long string of text) for transport.
        # Decode image
        image = decode_base64_image(data['image'])
        # Images come in many formats (PNG, JPEG, GIF). 
        #Some have an "Alpha" channel for transparency (RGBA), 
        # 
        # while others might be Grayscale. 
        # The Solution: K-means for color segmentation expects 3 features per pixel: Red, Green, and Blue.
        image_array = np.array(image.convert('RGB'))
        
        # Resize if needed
        image_array = resize_image(image_array, max_size)
        
        # Create segmenter and fit
        segmenter = ImageSegmenter(
            n_clusters=n_clusters,
            preprocessing=preprocessing
        )
        
        result = segmenter.fit(image_array, track_mlflow=should_track)
        current_segmenter = segmenter
        
        # Create visualization
        fig = create_visualization(
            image_array,
            result['segmented_image'],
            result['labels'],
            result['metrics'],
            n_clusters
        )
        
        # Prepare response
        response = {
            'success': True,
            'original_image': image_to_base64(image_array),
            'segmented_image': image_to_base64(result['segmented_image']),
            'visualization': fig_to_base64(fig),
            'metrics': {
                'silhouette_score': round(result['metrics']['silhouette_score'], 4),
                'davies_bouldin_index': round(result['metrics']['davies_bouldin_index'], 4),
                'calinski_harabasz_score': round(result['metrics']['calinski_harabasz_score'], 2),
                'inertia': round(result['metrics']['inertia'], 2),
                'segment_percentages': result['metrics']['segment_percentages']
            },
            'cluster_colors': result['cluster_colors'].tolist(),
            'n_clusters': n_clusters,
            'image_size': {
                'width': image_array.shape[1],
                'height': image_array.shape[0]
            },
            'mlflow_available': MLFLOW_AVAILABLE,
            'mlflow_tracked': should_track and segmenter.mlflow_run_id is not None
        }
        
        # Add MLflow run ID if available
        if segmenter.mlflow_run_id:
            response['mlflow_run_id'] = segmenter.mlflow_run_id
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/find-optimal-k', methods=['POST'])
def api_find_optimal_k():
    """
    Find optimal number of clusters.
    
    Request JSON:
        - image: base64 encoded image
        - k_min: minimum K (default: 2)
        - k_max: maximum K (default: 8)
    
    Response:
        - success: boolean
        - best_k: optimal K value
        - results: metrics for each K
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image = decode_base64_image(data['image'])
        image_array = resize_image(np.array(image.convert('RGB')), 300)
        
        k_min = int(data.get('k_min', 2))
        k_max = int(data.get('k_max', 8))
        
        result = find_optimal_k(image_array, k_range=range(k_min, k_max + 1))
        
        # Format results
        formatted = {}
        for k, metrics in result['results'].items():
            formatted[str(k)] = {
                'silhouette': round(metrics['silhouette'], 4),
                'davies_bouldin': round(metrics['davies_bouldin'], 4),
                'calinski_harabasz': round(metrics['calinski_harabasz'], 2),
                'inertia': round(metrics['inertia'], 2)
            }
        
        return jsonify({
            'success': True,
            'best_k': result['best_k'],
            'results': formatted
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample-images', methods=['GET'])
def list_sample_images():
    """List available sample images including sklearn built-in."""
    try:
        images = []
        
        # Add sklearn built-in images
        images.append({
            'name': 'china',
            'filename': 'sklearn_china',
            'type': 'sklearn',
            'description': 'Chinese Temple (sklearn built-in)'
        })
        images.append({
            'name': 'flower',
            'filename': 'sklearn_flower',
            'type': 'sklearn',
            'description': 'Flower (sklearn built-in)'
        })
        
        # Add local sample images
        if os.path.exists(app.config['SAMPLE_FOLDER']):
            for filename in os.listdir(app.config['SAMPLE_FOLDER']):
                if allowed_file(filename):
                    images.append({
                        'name': os.path.splitext(filename)[0],
                        'filename': filename,
                        'type': 'local',
                        'url': f'/sample_images/{filename}'
                    })
        
        return jsonify({'success': True, 'images': images})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample-images/<name>', methods=['GET'])
def get_sample_image(name):
    """Get a sample image as base64."""
    try:
        if name in ['china', 'flower', 'sklearn_china', 'sklearn_flower']:
            # Get sklearn image
            img_name = name.replace('sklearn_', '')
            image = get_sklearn_image(img_name)
            if image is not None:
                return jsonify({
                    'success': True,
                    'image': image_to_base64(image),
                    'name': img_name
                })
            else:
                return jsonify({'error': 'Could not load sklearn image'}), 500
        else:
            # Try local file
            filepath = os.path.join(app.config['SAMPLE_FOLDER'], name)
            if os.path.exists(filepath):
                image = Image.open(filepath)
                return jsonify({
                    'success': True,
                    'image': image_to_base64(np.array(image.convert('RGB'))),
                    'name': name
                })
            return jsonify({'error': 'Image not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/save', methods=['POST'])
def save_model():
    """Save the current model."""
    global current_segmenter
    
    if current_segmenter is None or current_segmenter.model is None:
        return jsonify({'error': 'No model to save. Run segmentation first.'}), 400
    
    try:
        data = request.get_json() or {}
        model_name = data.get('name', f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        filename = f"{secure_filename(model_name)}.joblib"
        filepath = os.path.join(app.config['MODELS_FOLDER'], filename)
        
        current_segmenter.save_model(filepath)
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'filename': filename,
            'path': filepath
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/load', methods=['POST'])
def load_model():
    """Load a saved model."""
    global current_segmenter
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join(app.config['MODELS_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Model not found'}), 404
        
        current_segmenter = ImageSegmenter.load_model(filepath)
        
        return jsonify({
            'success': True,
            'model_info': {
                'n_clusters': current_segmenter.n_clusters,
                'preprocessing': current_segmenter.preprocessing,
                'metrics': current_segmenter.metrics
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    """List all saved models."""
    try:
        models = []
        models_dir = app.config['MODELS_FOLDER']
        
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.joblib'):
                    filepath = os.path.join(models_dir, filename)
                    stat = os.stat(filepath)
                    models.append({
                        'filename': filename,
                        'name': os.path.splitext(filename)[0],
                        'size_kb': round(stat.st_size / 1024, 2),
                        'created': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return jsonify({'success': True, 'models': models})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<filename>', methods=['DELETE'])
def delete_model(filename):
    """Delete a saved model."""
    try:
        filepath = os.path.join(app.config['MODELS_FOLDER'], secure_filename(filename))
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': f'Deleted {filename}'})
        else:
            return jsonify({'error': 'Model not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_mlflow_tracking_uri():
    """Get properly formatted MLflow tracking URI."""
    mlruns_path = os.path.abspath("mlruns")
    os.makedirs(mlruns_path, exist_ok=True)
    
    if os.name == 'nt':  # Windows
        return f"file:///{mlruns_path.replace(os.sep, '/')}"
    else:  # Linux/Mac
        return f"file://{mlruns_path}"


@app.route('/api/mlflow/status', methods=['GET'])
def mlflow_status():
    """Check MLflow availability and status."""
    tracking_uri = get_mlflow_tracking_uri() if MLFLOW_AVAILABLE else None
    return jsonify({
        'success': True,
        'mlflow_available': MLFLOW_AVAILABLE,
        'mlflow_installed': mlflow is not None,
        'tracking_uri': tracking_uri
    })


@app.route('/api/mlflow/runs', methods=['GET'])
def mlflow_runs():
    """Get list of MLflow experiment runs."""
    if not MLFLOW_AVAILABLE or mlflow is None:
        return jsonify({
            'success': False,
            'error': 'MLflow not available',
            'runs': []
        })
    
    try:
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        client = MlflowClient()
        
        # Get all experiments
        experiments = client.search_experiments()
        
        all_runs = []
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=50
            )
            
            for run in runs:
                all_runs.append({
                    'run_id': run.info.run_id,
                    'experiment_name': exp.name,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'params': dict(run.data.params),
                    'metrics': {k: round(v, 4) for k, v in run.data.metrics.items()}
                })
        
        return jsonify({
            'success': True,
            'runs': all_runs,
            'total_runs': len(all_runs)
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'runs': []
        })


@app.route('/api/mlflow/runs/<run_id>', methods=['GET'])
def mlflow_run_details(run_id):
    """Get details of a specific MLflow run."""
    if not MLFLOW_AVAILABLE or mlflow is None:
        return jsonify({'success': False, 'error': 'MLflow not available'})
    
    try:
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        client = MlflowClient()
        
        run = client.get_run(run_id)
        
        return jsonify({
            'success': True,
            'run': {
                'run_id': run.info.run_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'params': dict(run.data.params),
                'metrics': {k: round(v, 4) for k, v in run.data.metrics.items()},
                'tags': dict(run.data.tags)
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'mlflow_available': MLFLOW_AVAILABLE
    })


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Image Segmentation API with MLflow")
    print("=" * 60)
    print("\n  Starting server...")
    print("  Open http://localhost:5000 in your browser")
    print("\n  API Endpoints:")
    print("    POST /api/segment          - Segment an image")
    print("    POST /api/find-optimal-k   - Find optimal K")
    print("    GET  /api/sample-images    - List sample images")
    print("    GET  /api/models           - List saved models")
    print("    POST /api/models/save      - Save current model")
    print("    POST /api/models/load      - Load a model")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)