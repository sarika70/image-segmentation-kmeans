"""
Image Segmentation Engine
==========================
K-means based image segmentation with MLflow tracking.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import load_sample_image
from PIL import Image, ImageEnhance, ImageFilter
import joblib
import os
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import MLflow - handle if not installed
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not installed. Experiment tracking disabled.")


class ImageSegmenter:
    """
    Image Segmentation using K-means clustering with MLflow tracking.
    """
    
    def __init__(self, n_clusters=4, preprocessing='enhance'):
        self.n_clusters = n_clusters
        self.preprocessing = preprocessing
        self.model = None
        self.metrics = {}
        self.cluster_colors = None
        self.original_shape = None
        self.mlflow_run_id = None
        
    def preprocess_image(self, image):
        """Apply image preprocessing."""
        img = image.copy() # deep copy and shallow copy
        
        if self.preprocessing == 'enhance':
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            img = img.filter(ImageFilter.SHARPEN)
        elif self.preprocessing == 'normalize':
            img_array = np.array(img).astype(np.float32)
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
            #Normalization stretches that range to 0–255.
            img_array = (img_array * 255).astype(np.uint8) #full 8-bit color depth
            img = Image.fromarray(img_array)
            
        return img
    
    def fit(self, image, track_mlflow=True, experiment_name="image_segmentation"):
        """
        Fit the segmentation model on an image.
        The Segment Distribution Analysis in the metrics section tells you 
        how much of a color exists, 
        while Color Extraction tells you what that color actually is.
        Together, they allow the engine to reconstruct the simplified image 
        and provide the data for the pie charts and visual summaries.
        """
        # Convert to PIL Image if needed
        # often used by OpenCV or Scikit-Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Ensure RGB
        # Also called as channel standardization. 
        #  Forces the image into a 3-channel (Red, Green, Blue) format.
        image = image.convert('RGB')
        
        # Preprocess
        processed_image = self.preprocess_image(image)
        image_array = np.array(processed_image)
        
        # Store original shape # RGB channels
        self.original_shape = image_array.shape
        
        # Reshape for clustering
        # Collapses the 2D grid of pixels into a long 1D list of 3D coordinates
        # A 10x10 image is a grid (10, 10, 3)
        # list of 100 points (100, 3).
        pixels = image_array.reshape(-1, 3).astype(np.float64)
        
        # Fit K-means
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        labels = self.model.fit_predict(pixels)
        
        # Calculate metrics
        self._calculate_metrics(pixels, labels)
        
        # Get cluster colors
        #Color Extraction Logic #Once the clusters are identified, 
        # this method determines the visual "identity" of each segment by calculating its representative color.
        # 
        self.cluster_colors = self._get_cluster_colors(image_array, labels)
        
        # Create segmented image
        segmented = self.cluster_colors[labels].reshape(self.original_shape)
        
        # Track with MLflow
        if track_mlflow:
            self._track_mlflow(experiment_name)
        
        return {
            'segmented_image': segmented,
            'labels': labels.reshape(self.original_shape[:2]),
            'metrics': self.metrics,
            'cluster_colors': self.cluster_colors,
            'n_clusters': self.n_clusters
        }
    
    def _get_cluster_colors(self, image_array, labels):
        """Get representative color for each cluster."""
        # 2D array to 1D 
        pixels = image_array.reshape(-1, 3)
        #Initialize an empty array to store the final RGB color for each of the K clusters
        colors = np.zeros((self.n_clusters, 3), dtype=np.uint8)
        
        for i in range(self.n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                colors[i] = np.mean(pixels[mask], axis=0).astype(np.uint8)
        
        return colors
    
    # calculating these scores for every single pixel in a high-resolution image 
    # is computationally expensive, we use a sampling strategy.
    def _calculate_metrics(self, pixels, labels):
        """Calculate clustering evaluation metrics."""
        # Sample for large images
        sample_size = 10000
        if len(pixels) > sample_size:
            # Randomly select 10,000 unique indices from the flattened pixel array.
            idx = np.random.choice(len(pixels), sample_size, replace=False)
            sample_pixels = pixels[idx]
            sample_labels = labels[idx]
        else:
            sample_pixels = pixels
            sample_labels = labels
            
        # Silhouette Score: Measures how similar a pixel is to its own cluster 
        # compared to other clusters. Range: -1 to 1 (Higher is better).
        #'silhouette_score': float(silhouette_score(sample_pixels, sample_labels)),
        
        # Davies-Bouldin Index: The average 'similarity' between clusters. 
        # Lower scores mean the clusters are better separated (Lower is better).
        #'davies_bouldin_index': float(davies_bouldin_score(sample_pixels, sample_labels)),
        
        # Calinski-Harabasz Score: The ratio of the sum of between-clusters scatter 
        # and of within-cluster scatter (Higher is better).
        #'calinski_harabasz_score': float(calinski_harabasz_score(sample_pixels, sample_labels)),
        
        # Inertia (WCSS): The sum of squared distances of samples to their closest cluster center. 
        # It measures how internally coherent the clusters are.
        #'inertia': float(self.model.inertia_)    
        
        self.metrics = {
            'silhouette_score': float(silhouette_score(sample_pixels, sample_labels)),
            'davies_bouldin_index': float(davies_bouldin_score(sample_pixels, sample_labels)),
            'calinski_harabasz_score': float(calinski_harabasz_score(sample_pixels, sample_labels)),
            'inertia': float(self.model.inertia_)
        }
        
        # Calculate segment percentages
        #  SEGMENT DISTRIBUTION ANALYSIS
        # Identify unique cluster IDs and count how many pixels were assigned to each.
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        self.metrics['segment_percentages'] = {
            f'segment_{i}': round(float(counts[i] / total * 100), 2)
            for i in range(len(unique))
        }
    
    def _track_mlflow(self, experiment_name):
        """Track experiment with MLflow."""
        if not MLFLOW_AVAILABLE:
            print("MLflow not available - skipping tracking")
            self.mlflow_run_id = None
            return
        
        try:
            # Set tracking URI with proper file:// scheme for cross-platform support
            mlruns_path = os.path.abspath("mlruns")
            
            # Ensure directory exists
            os.makedirs(mlruns_path, exist_ok=True)
            
            # Use file:// URI scheme (required for Windows and recommended for all platforms)
            # On Windows: file:///C:/path/to/mlruns
            # On Linux/Mac: file:///path/to/mlruns
            if os.name == 'nt':  # Windows
                tracking_uri = f"file:///{mlruns_path.replace(os.sep, '/')}"
            else:  # Linux/Mac
                tracking_uri = f"file://{mlruns_path}"
            
            mlflow.set_tracking_uri(tracking_uri)
            print(f"MLflow tracking URI: {tracking_uri}")
            
            # Set or create experiment
            mlflow.set_experiment(experiment_name)
            
            # Start run and log everything
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_param("n_clusters", self.n_clusters)
                mlflow.log_param("preprocessing", self.preprocessing)
                mlflow.log_param("image_height", self.original_shape[0])
                mlflow.log_param("image_width", self.original_shape[1])
                mlflow.log_param("total_pixels", self.original_shape[0] * self.original_shape[1])
                mlflow.log_param("timestamp", datetime.now().isoformat())
                
                # Log metrics
                mlflow.log_metric("silhouette_score", self.metrics['silhouette_score'])
                mlflow.log_metric("davies_bouldin_index", self.metrics['davies_bouldin_index'])
                mlflow.log_metric("calinski_harabasz_score", self.metrics['calinski_harabasz_score'])
                mlflow.log_metric("inertia", self.metrics['inertia'])
                
                # Log segment percentages as metrics
                for seg, pct in self.metrics.get('segment_percentages', {}).items():
                    mlflow.log_metric(f"pct_{seg}", pct)
                
                # Log model WITHOUT registering (model registry requires a database backend)
                mlflow.sklearn.log_model(
                    self.model, 
                    "kmeans_model",
                    registered_model_name=None  # Don't register - just log as artifact
                )
                
                # Store run ID
                self.mlflow_run_id = run.info.run_id
                print(f"MLflow tracking successful. Run ID: {self.mlflow_run_id}")
                
        except Exception as e:
            print(f"MLflow tracking error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.mlflow_run_id = None
    
    def predict(self, image):
        """Segment a new image using the fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        image = image.convert('RGB')
        processed_image = self.preprocess_image(image)
        image_array = np.array(processed_image)
        
        pixels = image_array.reshape(-1, 3).astype(np.float64)
        labels = self.model.predict(pixels)
        
        cluster_colors = self._get_cluster_colors(image_array, labels)
        segmented = cluster_colors[labels].reshape(image_array.shape)
        
        return {
            'segmented_image': segmented,
            'labels': labels.reshape(image_array.shape[:2])
        }
    
    def save_model(self, path):
        """Save the model to disk."""
        model_data = {
            'model': self.model,
            'n_clusters': self.n_clusters,
            'preprocessing': self.preprocessing,
            'cluster_colors': self.cluster_colors,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, path)
        return path
    
    @classmethod
    def load_model(cls, path):
        """Load a model from disk."""
        model_data = joblib.load(path)
        
        segmenter = cls(
            n_clusters=model_data['n_clusters'],
            preprocessing=model_data['preprocessing']
        )
        segmenter.model = model_data['model']
        segmenter.cluster_colors = model_data['cluster_colors']
        segmenter.metrics = model_data.get('metrics', {})
        
        return segmenter


def find_optimal_k(image, k_range=range(2, 9)):
    """Find optimal number of clusters."""
    results = {}
    
    for k in k_range:
        segmenter = ImageSegmenter(n_clusters=k)
        result = segmenter.fit(image, track_mlflow=False)
        results[k] = {
            'silhouette': result['metrics']['silhouette_score'],
            'davies_bouldin': result['metrics']['davies_bouldin_index'],
            'calinski_harabasz': result['metrics']['calinski_harabasz_score'],
            'inertia': result['metrics']['inertia']
        }
    
    best_k = max(results, key=lambda k: results[k]['silhouette'])
    return {'results': results, 'best_k': best_k}


def get_sklearn_image(name='china'):
    """Load sklearn sample images."""
    try:
        if name.lower() == 'flower':
            return load_sample_image('flower.jpg')
        else:
            return load_sample_image('china.jpg')
    except Exception as e:
        print(f"Error loading sklearn image: {e}")
        return None


def create_visualization(original, segmented, labels, metrics, n_clusters):
    """Create visualization of segmentation results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Segmented image
    axes[0, 1].imshow(segmented)
    axes[0, 1].set_title(f'Segmented (K={n_clusters})', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Segment labels
    im = axes[0, 2].imshow(labels, cmap='tab10')
    axes[0, 2].set_title('Segment Labels', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Segment distribution
    unique, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
    axes[1, 0].pie(counts, labels=[f'Seg {i}' for i in unique],
                   colors=colors, autopct='%1.1f%%')
    axes[1, 0].set_title('Segment Distribution', fontsize=12, fontweight='bold')
    
    # Metrics bar chart
    metric_names = ['Silhouette', 'Davies-Bouldin']
    metric_values = [metrics['silhouette_score'], metrics['davies_bouldin_index']]
    bar_colors = ['#3498db', '#e74c3c']
    axes[1, 1].bar(metric_names, metric_values, color=bar_colors)
    axes[1, 1].set_title('Clustering Metrics', fontsize=12, fontweight='bold')
    for i, v in enumerate(metric_values):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Metrics summary
    metrics_text = f"""
EVALUATION METRICS
══════════════════════

Silhouette Score:     {metrics['silhouette_score']:.4f}
  (Range: -1 to 1, higher = better)

Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}
  (Lower = better)

Calinski-Harabasz:    {metrics['calinski_harabasz_score']:.1f}
  (Higher = better)

Inertia (WCSS):       {metrics['inertia']:.1f}

══════════════════════
Segments: {n_clusters}
    """
    axes[1, 2].text(0.05, 0.95, metrics_text, fontsize=10, family='monospace',
                    verticalalignment='top', transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].set_title('Summary', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


'''
These utility functions act as the bridge between the Machine Learning 
world (NumPy arrays and Matplotlib plots) and 
the Web world (Strings/HTML). 
They prepare the final visual results to be sent over an API 
or displayed in a user interface.

'''

def image_to_base64(image_array):
    """Convert numpy array to base64 string."""
    img = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def image_to_base64_ct(image_array):
    """Convert numpy array to base64 string."""
    
    # 1. DATA CONVERSION
    # Converts the raw NumPy grid into a PIL (Python Imaging Library) object.
    # .astype(np.uint8) is critical: it ensures numbers are integers (0-255).
    img = Image.fromarray(image_array.astype(np.uint8))
    
    # 2. MEMORY MANAGEMENT
    # Creates an in-memory binary stream called a 'buffer'. 
    # This acts like a "virtual file" so we don't have to save to the hard drive.
    buffer = io.BytesIO()
    
    # 3. COMPRESSION/FORMATTING
    # Saves the PIL image into our virtual file using the PNG format.
    # This compresses the data so it travels faster over the network.
    img.save(buffer, format='PNG')
    
    # 4. STREAM HANDLING
    # Moves the 'read pointer' back to the very start of the virtual file.
    # Think of this as rewinding a tape so you can read it from the beginning.
    buffer.seek(0)
    
    # 5. ENCODING
    # .getvalue(): Gets the raw binary data (0s and 1s) from the buffer.
    # base64.b64encode(): Converts binary to a text-safe characters (A-Z, 0-9).
    # .decode('utf-8'): Turns the encoded bytes into a standard Python string.
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def fig_to_base64(fig):
    """Convert matplotlib figure to base64."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def fig_to_base64_ct(fig):
    """Convert matplotlib figure to base64."""
    """Convert matplotlib figure (plots/charts) to base64."""
    
    # 1. BUFFER INITIALIZATION
    # Similar to the image function, we use BytesIO to avoid writing to disk.
    buffer = io.BytesIO()
    
    # 2. RENDERING THE PLOT
    # Saves the chart (e.g., the Elbow plot or Color Pie Chart) into the buffer.
    # dpi=100 sets the resolution; bbox_inches='tight' removes unnecessary margins.
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    
    # 3. STREAM RESET
    # Moves the pointer back to zero so the encoder reads from the start.
    buffer.seek(0)
    
    # 4. MEMORY CLEANUP
    # Explicitly closes the figure. This is important to prevent memory leaks 
    # in web servers where many plots are generated over time.
    plt.close(fig)
    
    # 5. ENCODING
    # Converts the rendered plot image into a Base64 string for the frontend.
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def resize_image(image, max_size=500):
    """Resize image if too large."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    return np.array(image)


def resize_image_ct(image, max_size=500):
    """Resize image if too large."""
    # 1. Check if input is a NumPy array; if so, convert to PIL Image to use its resize tools
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    
    # 2. Get current width and height
    w, h = image.size
    
    # 3. Check if either dimension exceeds our safety limit (max_size)
    if max(w, h) > max_size:
        # Calculate the scale factor to maintain the original Aspect Ratio
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        
        # 4. Perform the resize using LANCZOS filtering (high-quality downsampling)
        # This significantly reduces the number of pixels (N) for K-means to process
        image = image.resize(new_size, Image.LANCZOS)
    
    # 5. Convert back to a NumPy array so it's ready for the KMeans .fit() method
    return np.array(image)