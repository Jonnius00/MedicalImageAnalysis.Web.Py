import cv2
import numpy as np
from utils.image_io import save_image


def apply_kmeans_clustering(image: np.ndarray, k: int = 3, guid: str = None) -> dict:
    """
    Apply K-means clustering to the input image.
    
    Args:
        image (np.ndarray): Input grayscale image
        k (int): Number of clusters
        guid (str): GUID for the image set
        
    Returns:
        dict: Dictionary containing the result image path, display image, and metrics
    """
    # Set random seed for reproducibility
    cv2.setRNGSeed(42)
    
    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to uint8
    centers = np.uint8(centers)
    
    # Map labels to centers
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    # Create a colorized version for better visualization
    colored_segmentation = create_colored_segmentation(labels, image.shape, k)
    
    # Save the result
    result_path = save_image(colored_segmentation, f"kmeans_k{k}", guid)
    
    # Calculate metrics
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    
    metrics = {
        "num_clusters": k,
        "cluster_centers": centers.flatten().tolist(),
        "label_counts": label_counts
    }
    
    return {
        "image_path": result_path,
        "display_image": colored_segmentation,
        "metrics": metrics
    }


def create_colored_segmentation(labels: np.ndarray, image_shape: tuple, k: int) -> np.ndarray:
    """
    Create a colorized segmentation image for better visualization.
    
    Args:
        labels (np.ndarray): Cluster labels for each pixel
        image_shape (tuple): Shape of the original image
        k (int): Number of clusters
        
    Returns:
        np.ndarray: Colorized segmentation image
    """
    # Set random seed for consistent colors
    np.random.seed(42)
    
    # Create a color palette
    colors = np.random.randint(0, 255, (k, 3), dtype=np.uint8)
    
    # Map labels to colors
    colored_image = colors[labels.flatten()]
    colored_image = colored_image.reshape((*image_shape, 3))
    
    return colored_image