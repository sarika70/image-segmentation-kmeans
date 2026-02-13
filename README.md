# Image Segmentation Web App

A complete web application for K-means image segmentation with Flask API, modern UI, and MLflow tracking.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-orange.svg)

## Features

- **K-Means Segmentation**: Segment images into K distinct regions
- **Modern Web UI**: Drag-and-drop interface with real-time results
- **MLflow Tracking**: Automatic experiment tracking and model logging
- **Model Management**: Save, load, and delete trained models
- **Optimal K Detection**: Automatically find the best number of clusters
- **Multiple Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz scores
- **Sample Images**: Built-in sklearn sample images for testing

## Project Structure

```
image_segmentation_app/
├── app.py                  # Flask API application
├── segmentation_engine.py  # K-means segmentation logic
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Web UI (HTML/CSS/JS)
├── models/                # Saved models directory
├── uploads/               # Uploaded images
├── sample_images/         # Sample images
├── mlruns/                # MLflow tracking data
└── artifacts/             # Generated artifacts
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/api/segment` | Segment an image |
| `POST` | `/api/find-optimal-k` | Find optimal K value |
| `GET` | `/api/sample-images` | List sample images |
| `GET` | `/api/sample-images/<name>` | Get sample image |
| `GET` | `/api/models` | List saved models |
| `POST` | `/api/models/save` | Save current model |
| `POST` | `/api/models/load` | Load a model |
| `DELETE` | `/api/models/<filename>` | Delete a model |
| `GET` | `/api/health` | Health check |

## API Usage Examples

### Segment an Image

```python
import requests
import base64

# Read image as base64
with open('image.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post('http://localhost:5000/api/segment', json={
    'image': image_b64,
    'n_clusters': 5,
    'preprocessing': 'enhance',
    'track_mlflow': True
})

result = response.json()
print(f"Silhouette Score: {result['metrics']['silhouette_score']}")
```

### Find Optimal K

```python
response = requests.post('http://localhost:5000/api/find-optimal-k', json={
    'image': image_b64,
    'k_min': 2,
    'k_max': 10
})

result = response.json()
print(f"Best K: {result['best_k']}")
```

### Save and Load Models

```python
# Save model
requests.post('http://localhost:5000/api/models/save', json={
    'name': 'my_model'
})

# List models
models = requests.get('http://localhost:5000/api/models').json()

# Load model
requests.post('http://localhost:5000/api/models/load', json={
    'filename': 'my_model.joblib'
})
```

## Segmentation Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | required | Base64 encoded image |
| `n_clusters` | int | 4 | Number of segments (2-12) |
| `preprocessing` | string | "enhance" | Preprocessing: "none", "enhance", "normalize" |
| `track_mlflow` | bool | true | Track experiment with MLflow |
| `max_size` | int | 500 | Max image dimension (for resizing) |

## Evaluation Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | -1 to 1 | Higher = better cluster separation |
| **Davies-Bouldin Index** | 0 to ∞ | Lower = better cluster separation |
| **Calinski-Harabasz** | 0 to ∞ | Higher = denser, well-separated clusters |
| **Inertia** | 0 to ∞ | Lower = tighter clusters |

## MLflow Tracking

View MLflow experiments:

```bash
cd image_segmentation_app
mlflow ui --port 5001
```

Then open: **http://localhost:5001**

### What Gets Tracked

**Parameters:**
- n_clusters
- preprocessing
- image_height, image_width
- total_pixels

**Metrics:**
- silhouette_score
- davies_bouldin_index
- calinski_harabasz_score
- inertia

**Artifacts:**
- kmeans_model (MLflow model format)

## Web UI Features

1. **Drag & Drop Upload**: Drop images directly onto the upload zone
2. **Sample Images**: Click sklearn sample images for quick testing
3. **Adjustable K**: Use slider to set number of segments (2-12)
4. **Preprocessing Options**: Enhance contrast, normalize, or none
5. **Real-time Metrics**: See clustering quality metrics instantly
6. **Visualization**: Full analysis plot with distribution charts
7. **Color Palette**: View extracted segment colors
8. **Model Management**: Save and load trained models

## Recommended K Values

| Image Type | Recommended K |
|------------|---------------|
| Simple objects | 2-4 |
| Portraits | 4-6 |
| Landscapes | 5-8 |
| Complex scenes | 8-12 |

## Tips for Best Results

1. **Use "Find Optimal K"** before segmentation to determine best cluster count
2. **Enable preprocessing** ("enhance") for better contrast in images
3. **Start with lower K** and increase if needed
4. **Save good models** for reuse on similar images
5. **Check silhouette score** - values > 0.5 indicate good clustering

## Troubleshooting

**Image too large:**
- Images are automatically resized to max 500px
- For larger images, processing may be slower

**Poor segmentation:**
- Try different K values
- Use "Find Optimal K" feature
- Try different preprocessing options

**MLflow not tracking:**
- Ensure `track_mlflow` is True
- Check `mlruns/` directory exists


# First, start the Flask server
python app.py

# In another terminal, run all tests
python test_api.py

# Run quick test only
python test_api.py --quick

# Run specific test class
python test_api.py TestSegmentationEndpoint

# Run specific test
python test_api.py TestSegmentationEndpoint.test_segment_image_basic

# Run with verbose output
python test_api.py -v
```

### Test Coverage

| Endpoint | Tests |
|----------|-------|
| `GET /api/health` | ✓ |
| `GET /api/mlflow/status` | ✓ |
| `GET /api/mlflow/runs` | ✓ |
| `POST /api/segment` | ✓ (8 scenarios) |
| `POST /api/find-optimal-k` | ✓ |
| `GET /api/sample-images` | ✓ |
| `GET /api/sample-images/<name>` | ✓ |
| `GET /api/models` | ✓ |
| `POST /api/models/save` | ✓ |
| `POST /api/models/load` | ✓ |
| `DELETE /api/models/<filename>` | ✓ |

### Expected Output
```
IMAGE SEGMENTATION API TEST SUITE
============================================================

test_health_check ... ✓ Health check passed. MLflow available: True
test_mlflow_status ... ✓ MLflow status: available=True
test_segment_image_basic ... ✓ Basic segmentation: silhouette=0.7182
test_segment_image_with_mlflow ... ✓ MLflow tracking: run_id=abc123
...

----------------------------------------------------------------------
Ran 29 tests in 45.123s

OK

## License

MIT License - Feel free to use and modify.
