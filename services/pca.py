import cv2
import numpy as np
from utils.image_io import save_image


def apply_pca(image: np.ndarray, num_components: int = 3, guid: str = None) -> dict:
    """
    Apply PCA to the input image.
    
    Args:
        image (np.ndarray): Input grayscale image
        num_components (int): Number of principal components to keep
        guid (str): GUID for the image set
        
    Returns:
        dict: Dictionary containing the result image path, display image, and metrics
    """
    # Get image dimensions
    h, w = image.shape
    
    # Reshape image to row vector format (each row is a pixel)
    data = image.astype(np.float32)
    
    # Calculate mean
    mean = np.mean(data, axis=0)
    
    # Center the data
    data_centered = data - mean
    
    # Apply PCA using OpenCV
    mean_vector, eigenvectors = cv2.PCACompute(data_centered, mean=None, maxComponents=num_components)
    
    # Project the data onto the principal components
    projected = np.dot(data_centered, eigenvectors.T)
    
    # Reconstruct the image
    reconstructed = np.dot(projected, eigenvectors) + mean
    
    # Clip values to valid range and convert to uint8
    reconstructed = np.clip(reconstructed, 0, 255)
    reconstructed = reconstructed.astype(np.uint8)
    
    # Save the result
    result_path = save_image(reconstructed, f"pca_{num_components}", guid)
    
    # Calculate explained variance ratio
    # First compute total variance
    total_variance = np.sum(np.var(data, axis=0))
    
    # Then compute variance of projected data
    projected_variance = np.sum(np.var(projected, axis=0))
    
    # Explained variance ratio
    explained_variance_ratio = projected_variance / total_variance if total_variance > 0 else 0
    
    # Calculate metrics
    metrics = {
        "num_components": num_components,
        "explained_variance_ratio": float(explained_variance_ratio),
        "total_variance": float(total_variance),
        "projected_variance": float(projected_variance)
    }
    
    return {
        "image_path": result_path,
        "display_image": reconstructed,
        "metrics": metrics
    }