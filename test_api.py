"""
API Test Cases for Image Segmentation Application
==================================================
Comprehensive test suite using requests library.

Usage:
    # First start the Flask app in another terminal:
    python app.py
    
    # Then run tests:
    python test_api.py
    
    # Run specific test:
    python test_api.py TestSegmentationAPI.test_segment_image
    
    # Run with verbose output:
    python test_api.py -v
"""

import requests
import base64
import unittest
import json
import os
import time
import numpy as np
from PIL import Image
from io import BytesIO


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "http://localhost:5000"
API_URL = f"{BASE_URL}/api"

# Timeout for requests (seconds)
TIMEOUT = 30


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_test_image(width=100, height=100, color_blocks=True):
    """Create a test image and return as base64."""
    if color_blocks:
        # Create image with distinct color regions (good for segmentation)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        half_h, half_w = height // 2, width // 2
        
        img[0:half_h, 0:half_w] = [255, 0, 0]       # Red
        img[0:half_h, half_w:width] = [0, 255, 0]   # Green
        img[half_h:height, 0:half_w] = [0, 0, 255]  # Blue
        img[half_h:height, half_w:width] = [255, 255, 0]  # Yellow
    else:
        # Random noise image
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Convert to base64
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_grayscale_test_image(width=100, height=100):
    """Create a grayscale test image."""
    img = np.zeros((height, width), dtype=np.uint8)
    img[0:height//2, :] = 50
    img[height//2:, :] = 200
    
    pil_img = Image.fromarray(img, mode='L')
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def is_server_running():
    """Check if the Flask server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# =============================================================================
# TEST CASES
# =============================================================================

class TestHealthEndpoint(unittest.TestCase):
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test GET /api/health returns healthy status."""
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
        self.assertIn('mlflow_available', data)
        
        print(f"✓ Health check passed. MLflow available: {data['mlflow_available']}")


class TestMLflowEndpoints(unittest.TestCase):
    """Test MLflow-related endpoints."""
    
    def test_mlflow_status(self):
        """Test GET /api/mlflow/status returns MLflow status."""
        response = requests.get(f"{API_URL}/mlflow/status", timeout=TIMEOUT)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('mlflow_available', data)
        self.assertIn('mlflow_installed', data)
        
        print(f"✓ MLflow status: available={data['mlflow_available']}")
    
    def test_mlflow_runs_empty(self):
        """Test GET /api/mlflow/runs returns runs list."""
        response = requests.get(f"{API_URL}/mlflow/runs", timeout=TIMEOUT)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('runs', data)
        
        if data['success']:
            print(f"✓ MLflow runs: {data.get('total_runs', 0)} runs found")
        else:
            print(f"✓ MLflow runs endpoint working (MLflow may not be installed)")


class TestSegmentationEndpoint(unittest.TestCase):
    """Test image segmentation endpoint."""
    
    def test_segment_image_basic(self):
        """Test POST /api/segment with basic parameters."""
        image_b64 = create_test_image(100, 100)
        
        payload = {
            'image': image_b64,
            'n_clusters': 4,
            'preprocessing': 'none',
            'track_mlflow': False
        }
        
        response = requests.post(
            f"{API_URL}/segment",
            json=payload,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('segmented_image', data)
        self.assertIn('original_image', data)
        self.assertIn('visualization', data)
        self.assertIn('metrics', data)
        self.assertIn('cluster_colors', data)
        self.assertEqual(data['n_clusters'], 4)
        
        # Check metrics
        metrics = data['metrics']
        self.assertIn('silhouette_score', metrics)
        self.assertIn('davies_bouldin_index', metrics)
        self.assertIn('calinski_harabasz_score', metrics)
        self.assertIn('inertia', metrics)
        self.assertIn('segment_percentages', metrics)
        
        print(f"✓ Basic segmentation: silhouette={metrics['silhouette_score']}")
    
    def test_segment_image_with_mlflow(self):
        """Test POST /api/segment with MLflow tracking enabled."""
        image_b64 = create_test_image(100, 100)
        
        payload = {
            'image': image_b64,
            'n_clusters': 3,
            'preprocessing': 'enhance',
            'track_mlflow': True
        }
        
        response = requests.post(
            f"{API_URL}/segment",
            json=payload,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('mlflow_available', data)
        self.assertIn('mlflow_tracked', data)
        
        if data['mlflow_tracked']:
            self.assertIn('mlflow_run_id', data)
            print(f"✓ MLflow tracking: run_id={data['mlflow_run_id']}")
        else:
            print(f"✓ Segmentation successful (MLflow not tracked)")
    
    def test_segment_different_k_values(self):
        """Test segmentation with different K values."""
        image_b64 = create_test_image(100, 100)
        
        k_values = [2, 4, 6, 8]
        results = {}
        
        for k in k_values:
            payload = {
                'image': image_b64,
                'n_clusters': k,
                'preprocessing': 'none',
                'track_mlflow': False
            }
            
            response = requests.post(
                f"{API_URL}/segment",
                json=payload,
                timeout=TIMEOUT
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data['success'])
            self.assertEqual(data['n_clusters'], k)
            self.assertEqual(len(data['cluster_colors']), k)
            
            results[k] = data['metrics']['silhouette_score']
        
        print(f"✓ Different K values tested: {results}")
    
    def test_segment_preprocessing_options(self):
        """Test different preprocessing options."""
        image_b64 = create_test_image(100, 100)
        
        preprocessing_options = ['none', 'enhance', 'normalize']
        
        for preprocess in preprocessing_options:
            payload = {
                'image': image_b64,
                'n_clusters': 4,
                'preprocessing': preprocess,
                'track_mlflow': False
            }
            
            response = requests.post(
                f"{API_URL}/segment",
                json=payload,
                timeout=TIMEOUT
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data['success'])
            
            print(f"✓ Preprocessing '{preprocess}': silhouette={data['metrics']['silhouette_score']}")
    
    def test_segment_larger_image(self):
        """Test segmentation with larger image (tests resize)."""
        image_b64 = create_test_image(800, 600)
        
        payload = {
            'image': image_b64,
            'n_clusters': 4,
            'preprocessing': 'none',
            'track_mlflow': False,
            'max_size': 300
        }
        
        response = requests.post(
            f"{API_URL}/segment",
            json=payload,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        
        # Check that image was resized
        self.assertLessEqual(data['image_size']['width'], 300)
        self.assertLessEqual(data['image_size']['height'], 300)
        
        print(f"✓ Large image resized to {data['image_size']}")
    
    def test_segment_no_image_error(self):
        """Test error handling when no image provided."""
        payload = {
            'n_clusters': 4
        }
        
        response = requests.post(
            f"{API_URL}/segment",
            json=payload,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)
        
        print(f"✓ No image error handled correctly")
    
    def test_segment_invalid_image_error(self):
        """Test error handling with invalid base64 image."""
        payload = {
            'image': 'not_valid_base64!!!',
            'n_clusters': 4
        }
        
        response = requests.post(
            f"{API_URL}/segment",
            json=payload,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn('error', data)
        
        print(f"✓ Invalid image error handled correctly")


class TestFindOptimalKEndpoint(unittest.TestCase):
    """Test find optimal K endpoint."""
    
    def test_find_optimal_k_basic(self):
        """Test POST /api/find-optimal-k with basic parameters."""
        image_b64 = create_test_image(100, 100)
        
        payload = {
            'image': image_b64,
            'k_min': 2,
            'k_max': 6
        }
        
        response = requests.post(
            f"{API_URL}/find-optimal-k",
            json=payload,
            timeout=TIMEOUT * 2  # This takes longer
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('best_k', data)
        self.assertIn('results', data)
        
        # Check results for each K
        for k in range(2, 7):
            k_str = str(k)
            self.assertIn(k_str, data['results'])
            self.assertIn('silhouette', data['results'][k_str])
            self.assertIn('davies_bouldin', data['results'][k_str])
        
        print(f"✓ Optimal K found: {data['best_k']}")
    
    def test_find_optimal_k_custom_range(self):
        """Test find optimal K with custom range."""
        image_b64 = create_test_image(100, 100)
        
        payload = {
            'image': image_b64,
            'k_min': 3,
            'k_max': 5
        }
        
        response = requests.post(
            f"{API_URL}/find-optimal-k",
            json=payload,
            timeout=TIMEOUT * 2
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Should only have K=3,4,5
        self.assertEqual(len(data['results']), 3)
        self.assertIn('3', data['results'])
        self.assertIn('4', data['results'])
        self.assertIn('5', data['results'])
        
        print(f"✓ Custom K range (3-5) tested, best K: {data['best_k']}")


class TestSampleImagesEndpoint(unittest.TestCase):
    """Test sample images endpoints."""
    
    def test_list_sample_images(self):
        """Test GET /api/sample-images returns image list."""
        response = requests.get(f"{API_URL}/sample-images", timeout=TIMEOUT)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('images', data)
        self.assertIsInstance(data['images'], list)
        
        # Should have at least sklearn images
        sklearn_images = [img for img in data['images'] if img.get('type') == 'sklearn']
        self.assertGreaterEqual(len(sklearn_images), 2)
        
        print(f"✓ Sample images listed: {len(data['images'])} images")
    
    def test_get_sklearn_china_image(self):
        """Test GET /api/sample-images/china returns sklearn image."""
        response = requests.get(f"{API_URL}/sample-images/china", timeout=TIMEOUT)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('image', data)
        self.assertEqual(data['name'], 'china')
        
        # Verify it's valid base64
        try:
            image_data = base64.b64decode(data['image'])
            self.assertGreater(len(image_data), 0)
        except Exception as e:
            self.fail(f"Invalid base64 image: {e}")
        
        print(f"✓ sklearn china image retrieved")
    
    def test_get_sklearn_flower_image(self):
        """Test GET /api/sample-images/flower returns sklearn image."""
        response = requests.get(f"{API_URL}/sample-images/flower", timeout=TIMEOUT)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('image', data)
        
        print(f"✓ sklearn flower image retrieved")


class TestModelManagementEndpoints(unittest.TestCase):
    """Test model save/load/delete endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Run segmentation first to create a model."""
        image_b64 = create_test_image(100, 100)
        
        payload = {
            'image': image_b64,
            'n_clusters': 4,
            'preprocessing': 'none',
            'track_mlflow': False
        }
        
        response = requests.post(
            f"{API_URL}/segment",
            json=payload,
            timeout=TIMEOUT
        )
        
        if response.status_code != 200:
            raise Exception("Failed to create model for testing")
    
    def test_01_save_model(self):
        """Test POST /api/models/save saves model."""
        payload = {
            'name': 'test_model_api'
        }
        
        response = requests.post(
            f"{API_URL}/models/save",
            json=payload,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['model_name'], 'test_model_api')
        self.assertIn('filename', data)
        
        print(f"✓ Model saved: {data['filename']}")
    
    def test_02_list_models(self):
        """Test GET /api/models returns model list."""
        response = requests.get(f"{API_URL}/models", timeout=TIMEOUT)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('models', data)
        self.assertIsInstance(data['models'], list)
        
        # Should have at least our test model
        model_names = [m['name'] for m in data['models']]
        self.assertIn('test_model_api', model_names)
        
        print(f"✓ Models listed: {len(data['models'])} models")
    
    def test_03_load_model(self):
        """Test POST /api/models/load loads model."""
        payload = {
            'filename': 'test_model_api.joblib'
        }
        
        response = requests.post(
            f"{API_URL}/models/load",
            json=payload,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('model_info', data)
        self.assertEqual(data['model_info']['n_clusters'], 4)
        
        print(f"✓ Model loaded: n_clusters={data['model_info']['n_clusters']}")
    
    def test_04_load_nonexistent_model(self):
        """Test loading nonexistent model returns error."""
        payload = {
            'filename': 'nonexistent_model.joblib'
        }
        
        response = requests.post(
            f"{API_URL}/models/load",
            json=payload,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn('error', data)
        
        print(f"✓ Nonexistent model error handled correctly")
    
    def test_05_delete_model(self):
        """Test DELETE /api/models/<filename> deletes model."""
        response = requests.delete(
            f"{API_URL}/models/test_model_api.joblib",
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        
        # Verify model is deleted
        response = requests.get(f"{API_URL}/models", timeout=TIMEOUT)
        data = response.json()
        model_names = [m['name'] for m in data['models']]
        self.assertNotIn('test_model_api', model_names)
        
        print(f"✓ Model deleted successfully")


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self):
        """Test complete workflow: find optimal K -> segment -> save -> load."""
        image_b64 = create_test_image(150, 150)
        
        # Step 1: Find optimal K
        print("\n--- Step 1: Finding optimal K ---")
        response = requests.post(
            f"{API_URL}/find-optimal-k",
            json={'image': image_b64, 'k_min': 2, 'k_max': 6},
            timeout=TIMEOUT * 2
        )
        self.assertEqual(response.status_code, 200)
        optimal_k = response.json()['best_k']
        print(f"Optimal K: {optimal_k}")
        
        # Step 2: Segment with optimal K
        print("\n--- Step 2: Segmenting with optimal K ---")
        response = requests.post(
            f"{API_URL}/segment",
            json={
                'image': image_b64,
                'n_clusters': optimal_k,
                'preprocessing': 'enhance',
                'track_mlflow': True
            },
            timeout=TIMEOUT
        )
        self.assertEqual(response.status_code, 200)
        segment_result = response.json()
        print(f"Silhouette: {segment_result['metrics']['silhouette_score']}")
        
        # Step 3: Save model
        print("\n--- Step 3: Saving model ---")
        response = requests.post(
            f"{API_URL}/models/save",
            json={'name': 'integration_test_model'},
            timeout=TIMEOUT
        )
        self.assertEqual(response.status_code, 200)
        print(f"Model saved: {response.json()['filename']}")
        
        # Step 4: Load model
        print("\n--- Step 4: Loading model ---")
        response = requests.post(
            f"{API_URL}/models/load",
            json={'filename': 'integration_test_model.joblib'},
            timeout=TIMEOUT
        )
        self.assertEqual(response.status_code, 200)
        loaded_info = response.json()['model_info']
        self.assertEqual(loaded_info['n_clusters'], optimal_k)
        print(f"Model loaded: n_clusters={loaded_info['n_clusters']}")
        
        # Cleanup: Delete model
        requests.delete(f"{API_URL}/models/integration_test_model.joblib", timeout=TIMEOUT)
        
        print("\n✓ Complete workflow test passed!")
    
    def test_batch_segmentation(self):
        """Test batch processing multiple images."""
        print("\n--- Batch Segmentation Test ---")
        
        images = [
            create_test_image(100, 100, color_blocks=True),
            create_test_image(120, 80, color_blocks=True),
            create_test_image(80, 120, color_blocks=False),
        ]
        
        results = []
        
        for i, img in enumerate(images):
            response = requests.post(
                f"{API_URL}/segment",
                json={
                    'image': img,
                    'n_clusters': 4,
                    'preprocessing': 'none',
                    'track_mlflow': False
                },
                timeout=TIMEOUT
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            results.append({
                'image': i + 1,
                'silhouette': data['metrics']['silhouette_score'],
                'size': data['image_size']
            })
            print(f"Image {i+1}: silhouette={data['metrics']['silhouette_score']:.4f}")
        
        self.assertEqual(len(results), 3)
        print(f"\n✓ Batch segmentation: {len(results)} images processed")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_minimum_k_value(self):
        """Test segmentation with K=2 (minimum)."""
        image_b64 = create_test_image(100, 100)
        
        response = requests.post(
            f"{API_URL}/segment",
            json={'image': image_b64, 'n_clusters': 2, 'track_mlflow': False},
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['n_clusters'], 2)
        self.assertEqual(len(data['cluster_colors']), 2)
        
        print(f"✓ Minimum K=2 works correctly")
    
    def test_high_k_value(self):
        """Test segmentation with high K value."""
        image_b64 = create_test_image(100, 100)
        
        response = requests.post(
            f"{API_URL}/segment",
            json={'image': image_b64, 'n_clusters': 10, 'track_mlflow': False},
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['n_clusters'], 10)
        
        print(f"✓ High K=10 works correctly")
    
    def test_small_image(self):
        """Test segmentation with very small image."""
        image_b64 = create_test_image(20, 20)
        
        response = requests.post(
            f"{API_URL}/segment",
            json={'image': image_b64, 'n_clusters': 4, 'track_mlflow': False},
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        
        print(f"✓ Small image (20x20) works correctly")
    
    def test_grayscale_image(self):
        """Test segmentation with grayscale image."""
        image_b64 = create_grayscale_test_image(100, 100)
        
        response = requests.post(
            f"{API_URL}/segment",
            json={'image': image_b64, 'n_clusters': 3, 'track_mlflow': False},
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        
        print(f"✓ Grayscale image works correctly")


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance(unittest.TestCase):
    """Performance and timing tests."""
    
    def test_segmentation_timing(self):
        """Measure segmentation response time."""
        image_b64 = create_test_image(200, 200)
        
        start_time = time.time()
        
        response = requests.post(
            f"{API_URL}/segment",
            json={'image': image_b64, 'n_clusters': 4, 'track_mlflow': False},
            timeout=TIMEOUT
        )
        
        elapsed = time.time() - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed, 10)  # Should complete within 10 seconds
        
        print(f"✓ Segmentation completed in {elapsed:.2f}s")
    
    def test_concurrent_requests(self):
        """Test handling of multiple concurrent requests."""
        import concurrent.futures
        
        image_b64 = create_test_image(100, 100)
        
        def make_request():
            response = requests.post(
                f"{API_URL}/segment",
                json={'image': image_b64, 'n_clusters': 4, 'track_mlflow': False},
                timeout=TIMEOUT
            )
            return response.status_code == 200
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        self.assertTrue(all(results))
        print(f"✓ {len(results)} concurrent requests handled successfully")


# =============================================================================
# MAIN
# =============================================================================

def run_quick_test():
    """Run a quick test to verify API is working."""
    print("\n" + "="*60)
    print("QUICK API TEST")
    print("="*60)
    
    # Check if server is running
    if not is_server_running():
        print("\n❌ ERROR: Flask server is not running!")
        print("Please start the server first: python app.py")
        return False
    
    print("✓ Server is running\n")
    
    # Test health endpoint
    response = requests.get(f"{API_URL}/health")
    print(f"Health check: {response.json()}")
    
    # Test segmentation
    image_b64 = create_test_image(100, 100)
    response = requests.post(
        f"{API_URL}/segment",
        json={'image': image_b64, 'n_clusters': 4, 'track_mlflow': True}
    )
    data = response.json()
    
    if data.get('success'):
        print(f"\n✓ Segmentation successful!")
        print(f"  - Silhouette score: {data['metrics']['silhouette_score']}")
        print(f"  - MLflow tracked: {data.get('mlflow_tracked', False)}")
        if data.get('mlflow_run_id'):
            print(f"  - MLflow run ID: {data['mlflow_run_id']}")
    else:
        print(f"\n❌ Segmentation failed: {data.get('error')}")
        return False
    
    print("\n" + "="*60)
    print("QUICK TEST PASSED!")
    print("="*60)
    return True


if __name__ == '__main__':
    import sys
    
    # Check if server is running
    if not is_server_running():
        print("\n" + "="*60)
        print("ERROR: Flask server is not running!")
        print("="*60)
        print("\nPlease start the server first:")
        print("  python app.py")
        print("\nThen run tests in another terminal:")
        print("  python test_api.py")
        print("="*60 + "\n")
        sys.exit(1)
    
    # Run quick test if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    # Run full test suite
    print("\n" + "="*60)
    print("IMAGE SEGMENTATION API TEST SUITE")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print("="*60 + "\n")
    
    # Run tests
    unittest.main(verbosity=2)