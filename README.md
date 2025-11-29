# 🏥 Medical Image Analysis Suite

A comprehensive, modular, and cross-platform graphical web application for medical image analysis implemented in Python with Streamlit.

## 🌟 Features

This application provides a suite of medical image analysis tools including:

- **Principal Component Analysis (PCA)** - Dimensionality reduction technique
- **K-means Clustering** - Unsupervised segmentation based on pixel intensity
- **Otsu Thresholding** - Automatic thresholding technique
- **Watershed Segmentation** - Marker-based image segmentation
- **Region Growing** - Seed-based segmentation technique
- **Algorithm Comparison** - Quantitative comparison between algorithms

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MedicalImageAnalysis.Web.Py
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv MedIMgAnalyze.Web
   MedIMgAnalyze.Web\Scripts\activate
   
   # macOS/Linux
   python3 -m venv MedIMgAnalyze.Web
   source MedIMgAnalyze.Web/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

Then open your browser to the URL provided in the terminal (typically http://localhost:8501).

## 📁 Supported File Formats

- DICOM (.dcm)
- PNG (.png)
- JPEG (.jpg, .jpeg)

## 📂 Project Structure

```
MedicalImageAnalysis.Web.Py/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── services/              # Image processing algorithms
│   ├── otsu.py
│   ├── kmeans.py
│   ├── pca.py
│   ├── region_growing.py
│   ├── watershed.py
│   └── comparison.py
├── utils/                 # Utility functions
│   ├── dicom_loader.py
│   └── image_io.py
├── static/                # Static files
│   └── images/            # Processed image outputs
└── tests/                 # Test scripts
    ├── test_services.py
    └── test_comparison.py
```

## 🔬 Algorithms

### Otsu Thresholding
Otsu's method is a clustering-based image thresholding technique that automatically determines the optimal threshold value by maximizing the inter-class variance.

### K-means Clustering
K-means clustering partitions the image pixels into K clusters based on their intensity values. The number of clusters (K) can be adjusted using a slider in the UI.

### Principal Component Analysis (PCA)
PCA is a statistical procedure that transforms the image data to a new coordinate system where the greatest variance lies on the first coordinate. Users can select the number of components to use for reconstruction.

### Watershed Segmentation
Watershed segmentation treats the image as a topographic surface and identifies catchment basins and watershed lines.

### Region Growing
Region growing starts from a seed point and grows a region by appending neighboring pixels that fulfill a certain criteria. The tolerance level can be adjusted using a slider in the UI.

### Algorithm Comparison
Quantitatively compare the results of different algorithms using various image metrics:
- Mean intensity
- Standard deviation of intensity
- Entropy (information content)
- Edge density
- Min/Max intensity values

The comparison tab provides:
1. Individual algorithm metrics displayed in a table format
2. Pairwise comparison matrix showing percentage differences between algorithms for each metric
3. Summary statistics across all algorithms

To perform a comparison, you need to run at least one algorithm first, then navigate to the "Comparison" tab and click "Perform Comparison".

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.