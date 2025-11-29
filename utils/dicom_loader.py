import pydicom
import numpy as np
import cv2
from typing import Tuple, Dict, Any


def load_dicom_image(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a DICOM image and extract metadata.
    
    Args:
        file_path (str): Path to the DICOM file
        
    Returns:
        tuple: (image_array, metadata_dict)
    """
    # Read DICOM file
    dicom_data = pydicom.dcmread(file_path)
    
    # Extract pixel data
    pixel_array = dicom_data.pixel_array
    
    # Normalize to uint8 [0, 255]
    normalized_image = normalize_image(pixel_array)
    
    # Extract metadata
    metadata = {
        'patient_id': str(dicom_data.get('PatientID', 'Unknown')),
        'study_date': str(dicom_data.get('StudyDate', 'Unknown')),
        'modality': str(dicom_data.get('Modality', 'Unknown')),
        'body_part': str(dicom_data.get('BodyPartExamined', 'Unknown')),
        'image_shape': pixel_array.shape
    }
    
    return normalized_image, metadata


def load_image(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load standard image formats (PNG, JPG) and prepare metadata.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        tuple: (image_array, metadata_dict)
    """
    # Load image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image from {file_path}")
    
    # Ensure the image is in the correct format
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize to uint8 [0, 255]
    normalized_image = normalize_image(image)
    
    # Create basic metadata
    metadata = {
        'patient_id': 'N/A',
        'study_date': 'N/A',
        'modality': 'N/A',
        'body_part': 'N/A',
        'image_shape': image.shape
    }
    
    return normalized_image, metadata


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to uint8 [0, 255] using min-max stretching.
    
    Args:
        image (np.ndarray): Input image array
        
    Returns:
        np.ndarray: Normalized uint8 image array
    """
    # Convert to float to prevent overflow
    img_float = image.astype(np.float32)
    
    # Min-max normalization
    img_min = np.min(img_float)
    img_max = np.max(img_float)
    
    if img_max > img_min:
        normalized = (img_float - img_min) / (img_max - img_min) * 255
    else:
        normalized = np.zeros_like(img_float)
    
    # Convert to uint8
    return normalized.astype(np.uint8)


def load_medical_image(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load medical image (DICOM or standard formats) based on file extension.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        tuple: (image_array, metadata_dict)
    """
    if file_path.lower().endswith('.dcm'):
        return load_dicom_image(file_path)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return load_image(file_path)
    else:
        raise ValueError(f"Unsupported file format for {file_path}")