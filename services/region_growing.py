import cv2
import numpy as np
from utils.image_io import save_image
from collections import deque


def apply_region_growing(image: np.ndarray, seed_point: tuple = None, tolerance: int = 10, guid: str = None) -> dict:
    """
    Apply region growing segmentation to the input image.
    
    Args:
        image (np.ndarray): Input grayscale image
        seed_point (tuple): Seed point coordinates (y, x) for region growing
        tolerance (int): Tolerance value for region growing
        guid (str): GUID for the image set
        
    Returns:
        dict: Dictionary containing the result image path, display image, and metrics
    """
    # If no seed point is provided, use the center of the image
    if seed_point is None:
        seed_point = (image.shape[0] // 2, image.shape[1] // 2)
    
    # Apply region growing
    binary_mask = region_growing(image, seed_point, tolerance)
    
    # Save the result
    result_path = save_image(binary_mask, f"region_growing_t{tolerance}", guid)
    
    # Calculate metrics
    region_pixels = np.sum(binary_mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    region_percentage = (region_pixels / total_pixels) * 100
    
    metrics = {
        "seed_point": seed_point,
        "tolerance": tolerance,
        "region_pixels": int(region_pixels),
        "region_percentage": float(region_percentage)
    }
    
    return {
        "image_path": result_path,
        "display_image": binary_mask,
        "metrics": metrics
    }


def region_growing(image: np.ndarray, seed: tuple, tolerance: int) -> np.ndarray:
    """
    Perform region growing segmentation using BFS.
    
    Args:
        image (np.ndarray): Input grayscale image
        seed (tuple): Seed point coordinates (y, x)
        tolerance (int): Tolerance value for region growing
        
    Returns:
        np.ndarray: Binary mask of the segmented region
    """
    # Get image dimensions
    rows, cols = image.shape
    
    # Create output mask
    mask = np.zeros((rows, cols), dtype=np.uint8)
    
    # Get seed intensity
    seed_intensity = image[seed]
    
    # Initialize queue with seed point
    queue = deque([seed])
    mask[seed] = 255
    
    # 8-connectivity neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    # BFS region growing
    while queue:
        current_point = queue.popleft()
        current_y, current_x = current_point
        
        # Check all 8 neighbors
        for dy, dx in neighbors:
            neighbor_y, neighbor_x = current_y + dy, current_x + dx
            
            # Check if neighbor is within image bounds
            if 0 <= neighbor_y < rows and 0 <= neighbor_x < cols:
                # Check if neighbor has already been processed
                if mask[neighbor_y, neighbor_x] == 0:
                    # Check if neighbor intensity is within tolerance
                    neighbor_intensity = image[neighbor_y, neighbor_x]
                    if abs(int(neighbor_intensity) - int(seed_intensity)) <= tolerance:
                        mask[neighbor_y, neighbor_x] = 255
                        queue.append((neighbor_y, neighbor_x))
    
    return mask