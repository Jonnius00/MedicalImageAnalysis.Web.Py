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


def apply_kmeans_to_stack(image_stack: np.ndarray, k: int = 3, progress_callback=None) -> dict:
    """
    Apply K-means clustering to each slice in a 3D image stack.
    
    Args:
        image_stack (np.ndarray): 3D array of shape (num_slices, height, width)
        k (int): Number of clusters
        progress_callback: Optional callback function(current, total) for progress updates
        
    Returns:
        dict: Dictionary containing:
            - labels_stack: 3D array of cluster labels (num_slices, height, width)
            - colored_stack: 4D array of colored images (num_slices, height, width, 3)
            - centers: Cluster centers (computed from all slices combined)
            - metrics: Processing metrics
    """
    # Set random seed for reproducibility
    cv2.setRNGSeed(42)
    np.random.seed(42)
    
    num_slices, height, width = image_stack.shape
    
    # Pre-allocate output arrays
    labels_stack = np.zeros((num_slices, height, width), dtype=np.int32)
    colored_stack = np.zeros((num_slices, height, width, 3), dtype=np.uint8)
    
    # Create consistent color palette for all slices
    colors = np.random.randint(0, 255, (k, 3), dtype=np.uint8)
    
    # K-means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Option 1: Apply K-means to entire volume for consistent clustering
    # Flatten all slices into one array for global clustering
    all_pixels = image_stack.reshape((-1, 1)).astype(np.float32)
    
    # Apply K-means to all pixels
    _, all_labels, centers = cv2.kmeans(
        all_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Reshape labels back to stack shape
    labels_stack = all_labels.reshape((num_slices, height, width))
    
    # Create colored visualization for each slice
    for i in range(num_slices):
        slice_labels = labels_stack[i].flatten()
        colored_slice = colors[slice_labels]
        colored_stack[i] = colored_slice.reshape((height, width, 3))
        
        # Report progress
        if progress_callback:
            progress_callback(i + 1, num_slices)
    
    # Calculate metrics
    unique, counts = np.unique(labels_stack, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    
    # Calculate total volume for each cluster (as percentage)
    total_voxels = num_slices * height * width
    cluster_volumes = {
        f"cluster_{label}": round(count / total_voxels * 100, 2) 
        for label, count in zip(unique.tolist(), counts.tolist())
    }
    
    metrics = {
        "num_clusters": k,
        "num_slices": num_slices,
        "slice_dimensions": f"{height} x {width}",
        "total_voxels": total_voxels,
        "cluster_centers": centers.flatten().tolist(),
        "cluster_volumes_percent": cluster_volumes,
        "label_counts": label_counts
    }
    
    return {
        "labels_stack": labels_stack,
        "colored_stack": colored_stack,
        "centers": centers,
        "colors": colors,
        "metrics": metrics
    }