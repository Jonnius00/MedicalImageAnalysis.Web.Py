import cv2
import numpy as np
import os
from typing import Union
from uuid import uuid4


def save_image(image: np.ndarray, suffix: str = "", guid: str = None) -> str:
    """
    Save processed image to static/images directory with GUID naming.
    
    Args:
        image (np.ndarray): Image array to save
        suffix (str): Suffix to add to filename
        guid (str): GUID for the image set (generated if None)
        
    Returns:
        str: Path to saved image relative to static directory
    """
    # Generate GUID if not provided
    if guid is None:
        guid = str(uuid4())
    
    # Ensure static/images directory exists
    images_dir = os.path.join("static", "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create filename
    filename = f"{guid}_{suffix}.png" if suffix else f"{guid}.png"
    file_path = os.path.join(images_dir, filename)
    
    # Save image
    success = cv2.imwrite(file_path, image)
    
    if not success:
        raise IOError(f"Failed to save image to {file_path}")
    
    # Return relative path for web display
    return os.path.join("static", "images", filename)


def load_saved_image(image_path: str) -> np.ndarray:
    """
    Load a previously saved image.
    
    Args:
        image_path (str): Path to the saved image
        
    Returns:
        np.ndarray: Loaded image array
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise IOError(f"Failed to load image from {image_path}")
    
    return image


def create_guid() -> str:
    """
    Create a new GUID for an image processing session.
    
    Returns:
        str: New GUID
    """
    return str(uuid4())