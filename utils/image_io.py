import cv2
import numpy as np
import os
import re
from typing import Union, List, Tuple
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


def natural_sort_key(filename: str) -> List:
    """
    Generate a key for natural sorting of filenames.
    Handles both numeric and alphabetic parts correctly.
    
    Example: ['img1', 'img2', 'img10'] sorts correctly instead of ['img1', 'img10', 'img2']
    
    Args:
        filename (str): Filename to generate sort key for
        
    Returns:
        List: Sort key with mixed int/str elements
    """
    return [int(part) if part.isdigit() else part.lower() 
            for part in re.split(r'(\d+)', filename)]


def load_image_stack(folder_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load multiple images from a folder as a 3D stack.
    
    Images are sorted naturally by filename and loaded as grayscale.
    All images must have the same dimensions.
    
    Args:
        folder_path (str): Path to folder containing image slices
        
    Returns:
        Tuple containing:
            - np.ndarray: 3D array of shape (num_slices, height, width)
            - List[str]: List of loaded filenames in order
            
    Raises:
        ValueError: If folder is empty or images have inconsistent dimensions
        IOError: If images cannot be loaded
    """
    # Supported image extensions
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    # Get list of image files
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files 
                   if os.path.splitext(f.lower())[1] in valid_extensions]
    
    if not image_files:
        raise ValueError(f"No valid image files found in {folder_path}")
    
    # Sort files naturally (so img2 comes before img10)
    image_files = sorted(image_files, key=natural_sort_key)
    
    # Load first image to get dimensions
    first_image_path = os.path.join(folder_path, image_files[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    
    if first_image is None:
        raise IOError(f"Failed to load image: {first_image_path}")
    
    height, width = first_image.shape
    num_slices = len(image_files)
    
    # Pre-allocate 3D array
    image_stack = np.zeros((num_slices, height, width), dtype=np.uint8)
    image_stack[0] = first_image
    
    # Load remaining images
    for i, filename in enumerate(image_files[1:], start=1):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")
        
        # Check dimensions match
        if image.shape != (height, width):
            raise ValueError(
                f"Image {filename} has dimensions {image.shape}, "
                f"expected ({height}, {width})"
            )
        
        image_stack[i] = image
    
    return image_stack, image_files


def load_image_stack_from_files(file_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Load multiple images from a list of file paths as a 3D stack.
    
    Useful for uploaded files where paths are already known.
    
    Args:
        file_paths (List[str]): List of paths to image files
        
    Returns:
        Tuple containing:
            - np.ndarray: 3D array of shape (num_slices, height, width)
            - List[str]: List of filenames in order
            
    Raises:
        ValueError: If list is empty or images have inconsistent dimensions
        IOError: If images cannot be loaded
    """
    if not file_paths:
        raise ValueError("No file paths provided")
    
    # Sort paths naturally
    file_paths = sorted(file_paths, key=lambda p: natural_sort_key(os.path.basename(p)))
    
    # Load first image to get dimensions
    first_image = cv2.imread(file_paths[0], cv2.IMREAD_GRAYSCALE)
    
    if first_image is None:
        raise IOError(f"Failed to load image: {file_paths[0]}")
    
    height, width = first_image.shape
    num_slices = len(file_paths)
    
    # Pre-allocate 3D array
    image_stack = np.zeros((num_slices, height, width), dtype=np.uint8)
    image_stack[0] = first_image
    
    # Extract filenames for return
    filenames = [os.path.basename(file_paths[0])]
    
    # Load remaining images
    for i, file_path in enumerate(file_paths[1:], start=1):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise IOError(f"Failed to load image: {file_path}")
        
        # Check dimensions match
        if image.shape != (height, width):
            raise ValueError(
                f"Image {os.path.basename(file_path)} has dimensions {image.shape}, "
                f"expected ({height}, {width})"
            )
        
        image_stack[i] = image
        filenames.append(os.path.basename(file_path))
    
    return image_stack, filenames