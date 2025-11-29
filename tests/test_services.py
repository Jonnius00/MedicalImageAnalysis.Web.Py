import numpy as np
import cv2
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.otsu import apply_otsu_thresholding
from services.kmeans import apply_kmeans_clustering
from services.pca import apply_pca
from services.watershed import apply_watershed
from services.region_growing import apply_region_growing


def create_test_image():
    """Create a simple test image for validation."""
    # Create a test image with distinct regions
    image = np.zeros((200, 200), dtype=np.uint8)
    
    # Add some shapes
    cv2.rectangle(image, (20, 20), (80, 80), 100, -1)
    cv2.circle(image, (150, 50), 30, 150, -1)
    cv2.rectangle(image, (100, 120), (180, 180), 200, -1)
    
    return image


def test_services():
    """Test all implemented services."""
    print("Creating test image...")
    test_image = create_test_image()
    
    print("Testing Otsu thresholding...")
    try:
        otsu_result = apply_otsu_thresholding(test_image)
        print(f"✓ Otsu thresholding successful. Threshold: {otsu_result['metrics']['threshold_value']}")
    except Exception as e:
        print(f"✗ Otsu thresholding failed: {e}")
    
    print("Testing K-means clustering...")
    try:
        kmeans_result = apply_kmeans_clustering(test_image, k=3)
        print(f"✓ K-means clustering successful. Clusters: {kmeans_result['metrics']['num_clusters']}")
    except Exception as e:
        print(f"✗ K-means clustering failed: {e}")
    
    print("Testing PCA...")
    try:
        pca_result = apply_pca(test_image, num_components=3)
        print(f"✓ PCA successful. Explained variance: {pca_result['metrics']['explained_variance_ratio']:.4f}")
    except Exception as e:
        print(f"✗ PCA failed: {e}")
    
    print("Testing Watershed...")
    try:
        watershed_result = apply_watershed(test_image)
        print(f"✓ Watershed successful. Regions: {watershed_result['metrics']['num_regions']}")
    except Exception as e:
        print(f"✗ Watershed failed: {e}")
    
    print("Testing Region Growing...")
    try:
        region_growing_result = apply_region_growing(test_image, seed_point=(50, 50), tolerance=20)
        print(f"✓ Region Growing successful. Region size: {region_growing_result['metrics']['region_pixels']} pixels")
    except Exception as e:
        print(f"✗ Region Growing failed: {e}")


if __name__ == "__main__":
    test_services()