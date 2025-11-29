import cv2
import numpy as np
from utils.image_io import save_image


def apply_otsu_thresholding(image: np.ndarray, guid: str = None) -> dict:
    """
    Apply Otsu's thresholding to the input image.
    
    Args:
        image (np.ndarray): Input grayscale image
        guid (str): GUID for the image set
        
    Returns:
        dict: Dictionary containing the result image path, display image, and metrics
    """
    # Apply Otsu's thresholding
    threshold_value, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert to uint8 if needed
    binary_mask = binary_mask.astype(np.uint8)
    
    # Save the result
    result_path = save_image(binary_mask, "otsu", guid)
    
    # Calculate metrics
    metrics = {
        "threshold_value": float(threshold_value),
        "foreground_pixels": int(np.sum(binary_mask > 0)),
        "background_pixels": int(np.sum(binary_mask == 0))
    }
    
    return {
        "image_path": result_path,
        "display_image": binary_mask,
        "metrics": metrics
    }