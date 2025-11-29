import cv2
import numpy as np
from utils.image_io import save_image


def apply_watershed(image: np.ndarray, guid: str = None) -> dict:
    """
    Apply watershed segmentation to the input image.
    
    Args:
        image (np.ndarray): Input grayscale image
        guid (str): GUID for the image set
        
    Returns:
        dict: Dictionary containing the result image path, display image, and metrics
    """
    # Apply Otsu thresholding to get binary image
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Sure foreground area
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
    
    # Create colorized watershed result
    colored_result = create_colored_watershed(image, markers)
    
    # Save the result
    result_path = save_image(colored_result, "watershed", guid)
    
    # Calculate metrics
    unique_labels = np.unique(markers)
    num_regions = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)  # Exclude boundary label (0)
    
    metrics = {
        "num_regions": int(num_regions),
        "boundary_pixels": int(np.sum(markers == 0)),
        "unique_labels": unique_labels.tolist()
    }
    
    return {
        "image_path": result_path,
        "display_image": colored_result,
        "metrics": metrics
    }


def create_colored_watershed(image: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    Create a colorized version of the watershed result.
    
    Args:
        image (np.ndarray): Original grayscale image
        markers (np.ndarray): Watershed markers
        
    Returns:
        np.ndarray: Colorized watershed image
    """
    # Create a colorized image
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Set random seed for consistent colors
    np.random.seed(42)
    
    # Colorize the regions
    unique_labels = np.unique(markers)
    
    # Create a color map for each label
    for label in unique_labels:
        if label == 0:  # Boundary
            colored_image[markers == label] = [255, 255, 255]  # White boundary
        else:
            # Random color for each region
            color = np.random.randint(0, 255, 3, dtype=np.uint8)
            colored_image[markers == label] = color
    
    return colored_image