import numpy as np
import cv2
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.comparison import calculate_image_metrics, compare_algorithms, similarity_score


def create_test_images():
    """Create test images for validation."""
    # Create a test image with distinct regions
    image1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image1, (20, 20), (80, 80), 100, -1)
    
    # Create a different test image
    image2 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(image2, (50, 50), 30, 150, -1)
    
    # Create a similar image to image1
    image3 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image3, (20, 20), (80, 80), 105, -1)  # Slightly different intensity
    
    return image1, image2, image3


def test_metrics():
    """Test image metrics calculation."""
    print("Testing image metrics calculation...")
    image1, image2, image3 = create_test_images()
    
    metrics1 = calculate_image_metrics(image1)
    metrics2 = calculate_image_metrics(image2)
    metrics3 = calculate_image_metrics(image3)
    
    print(f"Image 1 metrics: {metrics1}")
    print(f"Image 2 metrics: {metrics2}")
    print(f"Image 3 metrics: {metrics3}")
    
    # Check that all metrics are calculated
    required_metrics = ["mean_intensity", "std_intensity", "min_intensity", "max_intensity", "entropy", "edge_density"]
    for metric in required_metrics:
        assert metric in metrics1, f"Missing metric: {metric}"
        assert metric in metrics2, f"Missing metric: {metric}"
        assert metric in metrics3, f"Missing metric: {metric}"
    
    print("✓ Image metrics calculation successful")


def test_comparison():
    """Test algorithm comparison."""
    print("\nTesting algorithm comparison...")
    image1, image2, image3 = create_test_images()
    
    # Mock algorithm results
    results = {
        "Algorithm1": {"display_image": image1},
        "Algorithm2": {"display_image": image2},
        "Algorithm3": {"display_image": image3}
    }
    
    comparison = compare_algorithms(results)
    
    print(f"Comparison results keys: {comparison.keys()}")
    
    # Check that comparison contains expected keys
    assert "algorithms" in comparison, "Missing 'algorithms' key in comparison"
    assert "comparison_matrix" in comparison, "Missing 'comparison_matrix' key in comparison"
    assert "summary" in comparison, "Missing 'summary' key in comparison"
    
    print("✓ Algorithm comparison successful")


def test_similarity():
    """Test image similarity calculation."""
    print("\nTesting image similarity calculation...")
    image1, image2, image3 = create_test_images()
    
    # Test similarity between identical images (should be high)
    similarity_same = similarity_score(image1, image1)
    print(f"Similarity (same image): {similarity_same}")
    
    # Test similarity between similar images (should be moderate)
    similarity_similar = similarity_score(image1, image3)
    print(f"Similarity (similar images): {similarity_similar}")
    
    # Test similarity between different images (should be low)
    similarity_different = similarity_score(image1, image2)
    print(f"Similarity (different images): {similarity_different}")
    
    # Check that similarity values are in valid range
    assert 0 <= similarity_same <= 1, "Similarity should be between 0 and 1"
    assert 0 <= similarity_similar <= 1, "Similarity should be between 0 and 1"
    assert 0 <= similarity_different <= 1, "Similarity should be between 0 and 1"
    
    print("✓ Image similarity calculation successful")


if __name__ == "__main__":
    test_metrics()
    test_comparison()
    test_similarity()
    print("\n✓ All comparison tests passed!")