"""
MLflow Image Segmentation with Real Image Data
================================================
Uses real images from various sources for K-means segmentation.

Image Sources:
1. sklearn sample images (built-in)
2. PIL sample images
3. URL-based images (when network available)
4. Local file images

Usage:
    python main.py --image china --n_clusters 5
    python main.py --image flower --k_range 2,4,6,8
    python main.py --image_path /path/to/your/image.jpg
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import load_sample_image
import mlflow
import mlflow.sklearn
import argparse
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import PIL for additional image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import urllib for URL-based images
try:
    import urllib.request
    import io
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False


# =============================================================================
# IMAGE DATA SOURCES
# =============================================================================

def get_sklearn_sample_image(name='china'):
    """
    Load built-in sklearn sample images.
    
    Available images:
    - 'china': Chinese temple/building (427x640)
    - 'flower': Red/orange flower (427x640)
    
    These are REAL photographs included with scikit-learn.
    """
    try:
        if name.lower() == 'china':
            image = load_sample_image('china.jpg')
        elif name.lower() == 'flower':
            image = load_sample_image('flower.jpg')
        else:
            print(f"Unknown sklearn image: {name}. Using 'china'.")
            image = load_sample_image('china.jpg')
        
        print(f"Loaded sklearn sample image: {name}")
        print(f"Image shape: {image.shape}")
        return image
    except Exception as e:
        print(f"Error loading sklearn image: {e}")
        return None


def get_url_image(url):
    """Download image from URL."""
    if not URLLIB_AVAILABLE:
        print("urllib not available")
        return None
    
    try:
        print(f"Downloading image from: {url[:50]}...")
        with urllib.request.urlopen(url, timeout=15) as response:
            image_data = response.read()
        
        if PIL_AVAILABLE:
            img = Image.open(io.BytesIO(image_data))
            img = img.convert('RGB')
            return np.array(img)
        else:
            print("PIL not available for image processing")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def get_local_image(path):
    """Load image from local file path."""
    if not PIL_AVAILABLE:
        print("PIL not available")
        return None
    
    try:
        img = Image.open(path)
        img = img.convert('RGB')
        print(f"Loaded local image: {path}")
        return np.array(img)
    except Exception as e:
        print(f"Error loading local image: {e}")
        return None


def resize_image(image, max_size=300):
    """Resize image if too large (for faster processing)."""
    h, w = image.shape[:2]
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if PIL_AVAILABLE:
            img = Image.fromarray(image)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            image = np.array(img)
            print(f"Resized image from ({h}, {w}) to ({new_h}, {new_w})")
        else:
            # Simple numpy resize (lower quality but works)
            indices_h = np.linspace(0, h-1, new_h).astype(int)
            indices_w = np.linspace(0, w-1, new_w).astype(int)
            image = image[indices_h][:, indices_w]
            print(f"Resized image from ({h}, {w}) to ({new_h}, {new_w})")
    
    return image


# Sample real-world image URLs (Creative Commons / Public Domain)
SAMPLE_IMAGE_URLS = {
    'landscape': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Sunrise_over_the_sea.jpg/320px-Sunrise_over_the_sea.jpg',
    'city': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabyte.jpg/320px-New_york_times_square-terabyte.jpg',
    'nature': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bigar_Waterfall.jpg/320px-Bigar_Waterfall.jpg',
    'animal': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/220px-Cat_November_2010-1a.jpg',
    'food': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/300px-Good_Food_Display_-_NCI_Visuals_Online.jpg',
    'medical': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Cell_colony.jpg/220px-Cell_colony.jpg',
    'satellite': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/240px-The_Earth_seen_from_Apollo_17.jpg',
}


def get_image(source='china', image_path=None, max_size=300):
    """
    Get image from various sources.
    
    Parameters:
    -----------
    source : str
        Image source - 'china', 'flower' (sklearn), or URL key
    image_path : str
        Path to local image file
    max_size : int
        Maximum dimension for resizing
    
    Returns:
    --------
    numpy array : RGB image
    """
    image = None
    source_name = source
    
    # Priority 1: Local file path
    if image_path and os.path.exists(image_path):
        image = get_local_image(image_path)
        source_name = os.path.basename(image_path)
    
    # Priority 2: sklearn built-in images
    elif source.lower() in ['china', 'flower']:
        image = get_sklearn_sample_image(source)
        source_name = f"sklearn_{source}"
    
    # Priority 3: URL-based images
    elif source.lower() in SAMPLE_IMAGE_URLS:
        image = get_url_image(SAMPLE_IMAGE_URLS[source.lower()])
        source_name = source
    
    # Priority 4: Direct URL
    elif source.startswith('http'):
        image = get_url_image(source)
        source_name = "custom_url"
    
    # Fallback: sklearn china image
    if image is None:
        print("Falling back to sklearn 'china' image...")
        image = get_sklearn_sample_image('china')
        source_name = "sklearn_china"
    
    # Resize if needed
    if image is not None:
        image = resize_image(image, max_size)
    
    return image, source_name


# =============================================================================
# K-MEANS SEGMENTATION
# =============================================================================

def segment_image(image, n_clusters):
    """Segment image using K-means clustering."""
    original_shape = image.shape
    pixels = image.reshape(-1, 3).astype(np.float64)
    
    print(f"Clustering {len(pixels)} pixels into {n_clusters} segments...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(pixels)
    
    cluster_colors = kmeans.cluster_centers_.astype(np.uint8)
    segmented = cluster_colors[labels].reshape(original_shape)
    
    print("Cluster colors (RGB):")
    for i, color in enumerate(cluster_colors):
        print(f"  Segment {i}: RGB({color[0]:3d}, {color[1]:3d}, {color[2]:3d})")
    
    return segmented, labels, kmeans, pixels


def calculate_metrics(pixels, labels, kmeans, sample_size=10000):
    """Calculate clustering evaluation metrics."""
    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels, sample_labels = pixels[idx], labels[idx]
    else:
        sample_pixels, sample_labels = pixels, labels
    
    return {
        'silhouette': silhouette_score(sample_pixels, sample_labels),
        'davies_bouldin': davies_bouldin_score(sample_pixels, sample_labels),
        'calinski_harabasz': calinski_harabasz_score(sample_pixels, sample_labels),
        'inertia': kmeans.inertia_
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_result_plot(image, segmented, labels, kmeans, metrics, source_name):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'Original Image\n({source_name})')
    axes[0, 0].axis('off')
    
    # Segmented image
    axes[0, 1].imshow(segmented)
    axes[0, 1].set_title(f'Segmented (K={kmeans.n_clusters})')
    axes[0, 1].axis('off')
    
    # Segment labels
    label_img = labels.reshape(image.shape[:2])
    axes[0, 2].imshow(label_img, cmap='tab10')
    axes[0, 2].set_title('Segment Labels')
    axes[0, 2].axis('off')
    
    # Color palette
    n = kmeans.n_clusters
    colors = kmeans.cluster_centers_ / 255
    axes[1, 0].barh(range(n), [1]*n, color=colors)
    axes[1, 0].set_yticks(range(n))
    axes[1, 0].set_yticklabels([f'Seg {i}' for i in range(n)])
    axes[1, 0].set_title('Extracted Color Palette')
    axes[1, 0].set_xlim(0, 1)
    
    # Pixel distribution per segment
    unique, counts = np.unique(labels, return_counts=True)
    axes[1, 1].pie(counts, labels=[f'Seg {i}' for i in unique], 
                   colors=colors, autopct='%1.1f%%')
    axes[1, 1].set_title('Pixel Distribution')
    
    # Metrics
    metrics_text = f"""
    EVALUATION METRICS
    ==================
    
    Silhouette Score:     {metrics['silhouette']:.4f}
    (Range: -1 to 1, higher = better)
    
    Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}
    (Lower = better)
    
    Calinski-Harabasz:    {metrics['calinski_harabasz']:.1f}
    (Higher = better)
    
    Inertia (WCSS):       {metrics['inertia']:.1f}
    (Lower = better)
    """
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=11, 
                    family='monospace', verticalalignment='center')
    axes[1, 2].set_title('Metrics Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


# =============================================================================
# MLFLOW EXPERIMENT
# =============================================================================

def run_experiment(image_source='china', image_path=None, n_clusters=4, 
                   experiment_name='image_segmentation', max_size=300):
    """Run MLflow experiment with real image."""
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        print(f"\n{'='*60}")
        print(f"MLFLOW IMAGE SEGMENTATION EXPERIMENT")
        print(f"{'='*60}")
        
        # Load image
        image, source_name = get_image(image_source, image_path, max_size)
        
        if image is None:
            print("ERROR: Could not load any image!")
            return None, None
        
        print(f"\nImage loaded: {source_name}")
        print(f"Shape: {image.shape}")
        
        # Log parameters
        mlflow.log_param("image_source", source_name)
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("image_height", image.shape[0])
        mlflow.log_param("image_width", image.shape[1])
        mlflow.log_param("total_pixels", image.shape[0] * image.shape[1])
        mlflow.log_param("max_size", max_size)
        
        # Segment image
        print(f"\n--- Segmentation ---")
        segmented, labels, kmeans, pixels = segment_image(image, n_clusters)
        
        # Calculate metrics
        metrics = calculate_metrics(pixels, labels, kmeans)
        
        # Log metrics
        mlflow.log_metric("silhouette", metrics['silhouette'])
        mlflow.log_metric("davies_bouldin", metrics['davies_bouldin'])
        mlflow.log_metric("calinski_harabasz", metrics['calinski_harabasz'])
        mlflow.log_metric("inertia", metrics['inertia'])
        
        print(f"\n--- Metrics ---")
        print(f"Silhouette:        {metrics['silhouette']:.4f}")
        print(f"Davies-Bouldin:    {metrics['davies_bouldin']:.4f}")
        print(f"Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}")
        print(f"Inertia:           {metrics['inertia']:.1f}")
        
        # Save artifacts
        os.makedirs("artifacts", exist_ok=True)
        
        # Save plot
        fig = create_result_plot(image, segmented, labels, kmeans, metrics, source_name)
        plot_path = f"artifacts/{source_name}_k{n_clusters}.png"
        fig.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        mlflow.log_artifact(plot_path)
        
        # Save segmented image
        seg_path = f"artifacts/{source_name}_segmented_k{n_clusters}.npy"
        np.save(seg_path, segmented)
        mlflow.log_artifact(seg_path)
        
        # Save original image
        orig_path = f"artifacts/{source_name}_original.npy"
        np.save(orig_path, image)
        mlflow.log_artifact(orig_path)
        
        # Log model
        mlflow.sklearn.log_model(kmeans, "kmeans_model")
        
        model_path = f"artifacts/kmeans_{source_name}_k{n_clusters}.joblib"
        joblib.dump(kmeans, model_path)
        mlflow.log_artifact(model_path)
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n--- Artifacts Saved ---")
        print(f"Plot: {plot_path}")
        print(f"Model: {model_path}")
        print(f"MLflow Run ID: {run_id}")
        
        return metrics, run_id


def run_k_sweep(image_source='china', image_path=None, k_values=[2,3,4,5,6],
                experiment_name='image_segmentation', max_size=300):
    """Run experiments for multiple K values."""
    
    results = {}
    
    for k in k_values:
        metrics, run_id = run_experiment(
            image_source=image_source,
            image_path=image_path,
            n_clusters=k,
            experiment_name=experiment_name,
            max_size=max_size
        )
        if metrics:
            results[k] = {'metrics': metrics, 'run_id': run_id}
    
    if not results:
        print("No successful experiments!")
        return None, None
    
    # Summary
    best_k = max(results, key=lambda k: results[k]['metrics']['silhouette'])
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'K':<5} {'Silhouette':<12} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<18}")
    print("-" * 55)
    
    for k in sorted(results.keys()):
        m = results[k]['metrics']
        marker = " <-- BEST" if k == best_k else ""
        print(f"{k:<5} {m['silhouette']:<12.4f} {m['davies_bouldin']:<15.4f} {m['calinski_harabasz']:<18.1f}{marker}")
    
    print(f"\nBest K: {best_k} (highest silhouette score)")
    
    return results, best_k


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLflow Image Segmentation with Real Images')
    
    parser.add_argument('--image', type=str, default='china',
                        help='Image source: china, flower, landscape, city, nature, animal, food, or URL')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to local image file')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters (runs single experiment)')
    parser.add_argument('--k_range', type=str, default='2,3,4,5,6',
                        help='K values to test (comma-separated)')
    parser.add_argument('--experiment_name', type=str, default='image_segmentation',
                        help='MLflow experiment name')
    parser.add_argument('--max_size', type=int, default=300,
                        help='Max image dimension (for faster processing)')
    
    args = parser.parse_args()
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("mlruns")
    
    print("\n" + "="*60)
    print("AVAILABLE IMAGE SOURCES")
    print("="*60)
    print("Built-in (sklearn): china, flower")
    print("URL-based: landscape, city, nature, animal, food, medical, satellite")
    print("Custom: --image_path /path/to/image.jpg")
    print("="*60)
    
    if args.n_clusters:
        # Single experiment
        run_experiment(
            image_source=args.image,
            image_path=args.image_path,
            n_clusters=args.n_clusters,
            experiment_name=args.experiment_name,
            max_size=args.max_size
        )
    else:
        # K sweep
        k_values = [int(k) for k in args.k_range.split(',')]
        run_k_sweep(
            image_source=args.image,
            image_path=args.image_path,
            k_values=k_values,
            experiment_name=args.experiment_name,
            max_size=args.max_size
        )
    
    print(f"\n{'='*60}")
    print("View results: mlflow ui")
    print("Open: http://localhost:5000")
    print(f"{'='*60}")# -*- coding: utf-8 -*-

