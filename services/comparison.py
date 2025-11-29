import numpy as np
import cv2
from typing import Dict, List, Any


def calculate_image_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate various metrics for an image.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Basic statistics
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    min_intensity = float(np.min(gray))
    max_intensity = float(np.max(gray))
    
    # Image entropy (measure of information content)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist = hist[hist > 0]  # Remove zero bins
    hist = hist / hist.sum()  # Normalize
    entropy = -np.sum(hist * np.log2(hist))
    
    # Edge density (measure of edges in the image)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    
    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "entropy": float(entropy),
        "edge_density": float(edge_density)
    }


def compare_algorithms(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compare results from different algorithms.
    
    Args:
        results (dict): Dictionary of algorithm results
        
    Returns:
        dict: Comparison results
    """
    if not results:
        return {}
    
    comparison = {
        "algorithms": {},
        "comparison_matrix": {},
        "summary": {}
    }
    
    # Calculate metrics for each algorithm result
    for algo_name, result in results.items():
        if "display_image" in result:
            metrics = calculate_image_metrics(result["display_image"])
            comparison["algorithms"][algo_name] = metrics
    
    # Create comparison matrix
    algo_names = list(comparison["algorithms"].keys())
    metric_names = ["mean_intensity", "std_intensity", "entropy", "edge_density"]
    
    # Initialize comparison matrix
    comparison_matrix = {}
    for metric in metric_names:
        comparison_matrix[metric] = {}
        for algo1 in algo_names:
            comparison_matrix[metric][algo1] = {}
            for algo2 in algo_names:
                if algo1 in comparison["algorithms"] and algo2 in comparison["algorithms"]:
                    val1 = comparison["algorithms"][algo1][metric]
                    val2 = comparison["algorithms"][algo2][metric]
                    # Calculate relative difference
                    if val1 != 0:
                        diff = abs(val1 - val2) / val1 * 100
                    else:
                        diff = 0 if val2 == 0 else float('inf')
                    comparison_matrix[metric][algo1][algo2] = round(diff, 2)
    
    comparison["comparison_matrix"] = comparison_matrix
    
    # Create summary statistics
    summary = {}
    for metric in metric_names:
        values = [comparison["algorithms"][algo][metric] for algo in algo_names 
                 if algo in comparison["algorithms"]]
        if values:
            summary[metric] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "range": max(values) - min(values)
            }
    
    comparison["summary"] = summary
    
    return comparison


def similarity_score(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate similarity score between two images.
    
    Args:
        image1 (np.ndarray): First image
        image2 (np.ndarray): Second image
        
    Returns:
        float: Similarity score (0-1, where 1 is identical)
    """
    # Resize images to the same size if needed
    if image1.shape != image2.shape:
        # Resize to the smaller dimensions
        min_height = min(image1.shape[0], image2.shape[0])
        min_width = min(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (min_width, min_height))
        image2 = cv2.resize(image2, (min_width, min_height))
    
    # Convert to grayscale if needed
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Normalize images to [0, 1]
    image1_norm = image1.astype(np.float32) / 255.0
    image2_norm = image2.astype(np.float32) / 255.0
    
    # Calculate structural similarity (simplified)
    # Using mean squared error as a simple similarity measure
    mse = np.mean((image1_norm - image2_norm) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    # Normalize PSNR to 0-1 range (assuming PSNR in 0-50 range)
    similarity = min(psnr / 50.0, 1.0)
    
    return float(similarity)