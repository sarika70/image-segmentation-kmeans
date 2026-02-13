"""
Download Sample Images for Offline Use (v2)
=============================================
Uses multiple reliable image sources to avoid rate limiting.

Sources:
- scikit-learn built-in images
- Picsum.photos (Lorem Picsum - free, no rate limits)
- Unsplash Source API (free, reliable)
- Generated synthetic images

Usage:
    python download_sample_images_v2.py
"""

import os
import urllib.request
import numpy as np
from PIL import Image
import io
import time

# Create images directory
IMAGES_DIR = "sample_images"
os.makedirs(IMAGES_DIR, exist_ok=True)


# =============================================================================
# RELIABLE IMAGE SOURCES (No Rate Limiting)
# =============================================================================

# Picsum Photos - Free, reliable, no rate limits
# Format: https://picsum.photos/seed/{name}/{width}/{height}
PICSUM_IMAGES = {
    'landscape': 'https://picsum.photos/seed/landscape123/640/480',
    'city': 'https://picsum.photos/seed/cityscape456/640/480',
    'nature': 'https://picsum.photos/seed/nature789/640/480',
    'portrait': 'https://picsum.photos/seed/portrait321/480/640',
    'abstract': 'https://picsum.photos/seed/abstract654/640/480',
    'building': 'https://picsum.photos/seed/architecture987/640/480',
    'forest': 'https://picsum.photos/seed/forest111/640/480',
    'ocean': 'https://picsum.photos/seed/ocean222/640/480',
    'mountain': 'https://picsum.photos/seed/mountain333/640/480',
    'flower': 'https://picsum.photos/seed/flower444/640/480',
}

# Alternative sources (public domain / CC0)
ALTERNATIVE_SOURCES = {
    'colorful1': 'https://www.w3schools.com/css/img_5terre.jpg',
    'colorful2': 'https://www.w3schools.com/css/img_forest.jpg',
    'lights': 'https://www.w3schools.com/css/img_lights.jpg',
}


def download_image(name, url, save_dir=IMAGES_DIR, delay=0.5):
    """Download and save an image with retry logic."""
    filepath = os.path.join(save_dir, f"{name}.jpg")
    
    if os.path.exists(filepath):
        print(f"  [SKIP] {name} already exists")
        return filepath
    
    try:
        print(f"  [DOWNLOADING] {name}...", end=" ", flush=True)
        
        # Add delay to be polite to servers
        time.sleep(delay)
        
        # Create request with proper headers
        request = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            }
        )
        
        with urllib.request.urlopen(request, timeout=30) as response:
            image_data = response.read()
        
        # Open and convert to RGB
        img = Image.open(io.BytesIO(image_data))
        img = img.convert('RGB')
        
        # Save as JPEG
        img.save(filepath, 'JPEG', quality=95)
        print(f"OK ({img.size[0]}x{img.size[1]})")
        
        return filepath
    
    except Exception as e:
        print(f"FAILED ({str(e)[:50]})")
        return None


def create_synthetic_images(save_dir=IMAGES_DIR):
    """Create synthetic images for testing (no download needed)."""
    
    synthetic_images = {
        'synthetic_colors': create_color_blocks(),
        'synthetic_gradient': create_gradient(),
        'synthetic_circles': create_circles(),
        'synthetic_stripes': create_stripes(),
        'synthetic_checker': create_checkerboard(),
    }
    
    print("\n--- Creating Synthetic Images ---")
    
    for name, img_array in synthetic_images.items():
        filepath = os.path.join(save_dir, f"{name}.jpg")
        if os.path.exists(filepath):
            print(f"  [SKIP] {name} already exists")
            continue
        
        img = Image.fromarray(img_array)
        img.save(filepath, 'JPEG', quality=95)
        print(f"  [CREATED] {name} ({img_array.shape[1]}x{img_array.shape[0]})")
    
    return list(synthetic_images.keys())


def create_color_blocks():
    """Create image with distinct color blocks."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 4 quadrants with different colors
    img[0:200, 0:200] = [220, 60, 60]      # Red
    img[0:200, 200:400] = [60, 180, 60]    # Green
    img[200:400, 0:200] = [60, 60, 200]    # Blue
    img[200:400, 200:400] = [230, 220, 60] # Yellow
    
    # Add noise
    noise = np.random.randint(-20, 20, img.shape)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Orange circle in center
    y, x = np.ogrid[:400, :400]
    mask = (x - 200)**2 + (y - 200)**2 <= 60**2
    img[mask] = [255, 140, 0]
    
    return img


def create_gradient():
    """Create RGB gradient image."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    for i in range(400):
        for j in range(400):
            img[i, j] = [
                int(255 * i / 400),      # R increases top to bottom
                int(255 * j / 400),      # G increases left to right
                int(255 * (400-i) / 400) # B decreases top to bottom
            ]
    
    return img


def create_circles():
    """Create image with colored circles."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240  # Light gray background
    
    circles = [
        ((100, 100), 60, [255, 0, 0]),    # Red
        ((300, 100), 60, [0, 255, 0]),    # Green
        ((100, 300), 60, [0, 0, 255]),    # Blue
        ((300, 300), 60, [255, 255, 0]),  # Yellow
        ((200, 200), 80, [255, 128, 0]),  # Orange (center)
    ]
    
    y, x = np.ogrid[:400, :400]
    
    for (cx, cy), radius, color in circles:
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        img[mask] = color
    
    return img


def create_stripes():
    """Create striped pattern image."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    colors = [
        [255, 0, 0],    # Red
        [255, 165, 0],  # Orange
        [255, 255, 0],  # Yellow
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [128, 0, 128],  # Purple
    ]
    
    stripe_height = 400 // len(colors)
    
    for i, color in enumerate(colors):
        start = i * stripe_height
        end = (i + 1) * stripe_height
        img[start:end, :] = color
    
    # Add some noise
    noise = np.random.randint(-15, 15, img.shape)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def create_checkerboard():
    """Create checkerboard pattern with multiple colors."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    colors = [
        [200, 50, 50],   # Dark red
        [50, 150, 50],   # Dark green
        [50, 50, 180],   # Dark blue
        [180, 180, 50],  # Dark yellow
    ]
    
    block_size = 50
    
    for i in range(0, 400, block_size):
        for j in range(0, 400, block_size):
            color_idx = ((i // block_size) + (j // block_size)) % len(colors)
            img[i:i+block_size, j:j+block_size] = colors[color_idx]
    
    return img


def get_sklearn_images(save_dir=IMAGES_DIR):
    """Save sklearn built-in sample images."""
    print("\n--- Saving sklearn Built-in Images ---")
    
    try:
        from sklearn.datasets import load_sample_image
        
        for name in ['china', 'flower']:
            filepath = os.path.join(save_dir, f"sklearn_{name}.jpg")
            if os.path.exists(filepath):
                print(f"  [SKIP] sklearn_{name} already exists")
                continue
            
            img_array = load_sample_image(f'{name}.jpg')
            img = Image.fromarray(img_array)
            img.save(filepath, 'JPEG', quality=95)
            print(f"  [SAVED] sklearn_{name} ({img_array.shape[1]}x{img_array.shape[0]})")
        
        return ['sklearn_china', 'sklearn_flower']
    
    except Exception as e:
        print(f"  [ERROR] Could not load sklearn images: {e}")
        return []


def main():
    print("=" * 60)
    print("DOWNLOADING SAMPLE IMAGES FOR SEGMENTATION (v2)")
    print("=" * 60)
    print(f"\nSaving to: {os.path.abspath(IMAGES_DIR)}/")
    
    successful = []
    failed = []
    
    # 1. Save sklearn built-in images (always works)
    sklearn_images = get_sklearn_images()
    successful.extend(sklearn_images)
    
    # 2. Create synthetic images (always works)
    synthetic_images = create_synthetic_images()
    successful.extend(synthetic_images)
    
    # 3. Download from Picsum (reliable, no rate limits)
    print("\n--- Downloading from Picsum Photos ---")
    for name, url in PICSUM_IMAGES.items():
        result = download_image(name, url, delay=0.3)
        if result:
            successful.append(name)
        else:
            failed.append(name)
    
    # 4. Download from alternative sources
    print("\n--- Downloading from Alternative Sources ---")
    for name, url in ALTERNATIVE_SOURCES.items():
        result = download_image(name, url, delay=0.3)
        if result:
            successful.append(name)
        else:
            failed.append(name)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed downloads: {', '.join(failed)}")
    
    # List all available images
    print(f"\n--- Available Images ---")
    all_images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')])
    
    for f in all_images:
        path = os.path.join(IMAGES_DIR, f)
        img = Image.open(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {f:<30} {img.size[0]:>4}x{img.size[1]:<4}  ({size_kb:.1f} KB)")
    
    print(f"\n--- Usage Examples ---")
    print(f"python main_real_images.py --image_path sample_images/sklearn_china.jpg --n_clusters 5")
    print(f"python main_real_images.py --image_path sample_images/synthetic_colors.jpg --k_range 2,4,6")
    print(f"python main_real_images.py --image_path sample_images/landscape.jpg --n_clusters 6")


if __name__ == "__main__":
    main()