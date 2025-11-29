import streamlit as st
import cv2
import numpy as np
from utils.dicom_loader import load_medical_image
from utils.image_io import create_guid
import os

# Import services
from services.otsu import apply_otsu_thresholding
from services.kmeans import apply_kmeans_clustering
from services.pca import apply_pca
from services.watershed import apply_watershed
from services.region_growing import apply_region_growing
from services.comparison import compare_algorithms

# Set page configuration
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon=":hospital:",
    layout="wide"
)


def main():
    st.title("Medical Image Analysis Suite")
    st.markdown("""
    This application provides a suite of medical image analysis tools including:
    - Principal Component Analysis (PCA)
    - K-means Clustering
    - Otsu Thresholding
    - Watershed Segmentation
    - Region Growing
    """)
    
    # Initialize session state
    if 'current_guid' not in st.session_state:
        st.session_state.current_guid = None
    
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
        
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=['dcm', 'png', 'jpg', 'jpeg'],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Load the medical image
            image, metadata = load_medical_image(temp_file_path)
            
            # Store in session state
            st.session_state.uploaded_image = image
            st.session_state.metadata = metadata
            
            # Create new GUID for this session
            st.session_state.current_guid = create_guid()
            
            # Display image info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, caption="Medical Image", use_column_width=True)
            
            with col2:
                st.subheader("Image Metadata")
                st.json(metadata)
                
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    # Show processing options if image is loaded
    if st.session_state.uploaded_image is not None:
        st.markdown("---")
        st.subheader("Image Processing Algorithms")
        
        # Create tabs for different algorithms
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Otsu Thresholding", 
            "K-means Clustering", 
            "PCA", 
            "Watershed", 
            "Region Growing",
            "Comparison"
        ])
        
        with tab1:
            st.header("Otsu Thresholding")
            st.markdown("""
            Otsu's method is a clustering-based image thresholding technique that 
            automatically determines the optimal threshold value by maximizing the 
            inter-class variance.
            """)
            
            if 'otsu_result' not in st.session_state:
                st.session_state.otsu_result = None
                
            if st.button("Apply Otsu Thresholding"):
                with st.spinner("Processing..."):
                    try:
                        st.session_state.otsu_result = apply_otsu_thresholding(
                            st.session_state.uploaded_image, 
                            st.session_state.current_guid
                        )
                        st.success("Processing complete!")
                    except Exception as e:
                        st.error(f"Error applying Otsu thresholding: {str(e)}")
            
            if st.session_state.otsu_result is not None:
                st.subheader("Otsu Thresholding Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        st.session_state.otsu_result["display_image"], 
                        caption="Binary Mask", 
                        use_column_width=True
                    )
                
                with col2:
                    st.subheader("Metrics")
                    st.json(st.session_state.otsu_result["metrics"])
                    st.download_button(
                        label="Download Result",
                        data=open(st.session_state.otsu_result["image_path"], "rb").read(),
                        file_name=os.path.basename(st.session_state.otsu_result["image_path"]),
                        mime="image/png"
                    )
        
        with tab2:
            st.header("K-means Clustering")
            st.markdown("""
            K-means clustering partitions the image pixels into K clusters based 
            on their intensity values.
            """)
            
            if 'kmeans_result' not in st.session_state:
                st.session_state.kmeans_result = None
                
            k_value = st.slider("Number of clusters (K)", 2, 10, 3)
            
            if st.button("Apply K-means Clustering"):
                with st.spinner("Processing..."):
                    try:
                        st.session_state.kmeans_result = apply_kmeans_clustering(
                            st.session_state.uploaded_image,
                            k_value,
                            st.session_state.current_guid
                        )
                        st.success("Processing complete!")
                    except Exception as e:
                        st.error(f"Error applying K-means clustering: {str(e)}")
            
            if st.session_state.kmeans_result is not None:
                st.subheader("K-means Clustering Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        st.session_state.kmeans_result["display_image"], 
                        caption=f"K-means Clustering (K={k_value})", 
                        use_column_width=True
                    )
                
                with col2:
                    st.subheader("Metrics")
                    st.json(st.session_state.kmeans_result["metrics"])
                    st.download_button(
                        label="Download Result",
                        data=open(st.session_state.kmeans_result["image_path"], "rb").read(),
                        file_name=os.path.basename(st.session_state.kmeans_result["image_path"]),
                        mime="image/png"
                    )
        
        with tab3:
            st.header("Principal Component Analysis (PCA)")
            st.markdown("""
            PCA is a statistical procedure that transforms the image data to a new 
            coordinate system where the greatest variance lies on the first coordinate.
            """)
            
            if 'pca_result' not in st.session_state:
                st.session_state.pca_result = None
                
            components = st.slider("Number of components", 1, 10, 3)
            
            if st.button("Apply PCA"):
                with st.spinner("Processing..."):
                    try:
                        st.session_state.pca_result = apply_pca(
                            st.session_state.uploaded_image,
                            components,
                            st.session_state.current_guid
                        )
                        st.success("Processing complete!")
                    except Exception as e:
                        st.error(f"Error applying PCA: {str(e)}")
            
            if st.session_state.pca_result is not None:
                st.subheader("PCA Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        st.session_state.pca_result["display_image"], 
                        caption=f"Reconstructed Image ({components} components)", 
                        use_column_width=True
                    )
                
                with col2:
                    st.subheader("Metrics")
                    st.json(st.session_state.pca_result["metrics"])
                    st.download_button(
                        label="Download Result",
                        data=open(st.session_state.pca_result["image_path"], "rb").read(),
                        file_name=os.path.basename(st.session_state.pca_result["image_path"]),
                        mime="image/png"
                    )
        
        with tab4:
            st.header("Watershed Segmentation")
            st.markdown("""
            Watershed segmentation treats the image as a topographic surface and 
            identifies catchment basins and watershed lines.
            """)
            
            if 'watershed_result' not in st.session_state:
                st.session_state.watershed_result = None
                
            if st.button("Apply Watershed"):
                with st.spinner("Processing..."):
                    try:
                        st.session_state.watershed_result = apply_watershed(
                            st.session_state.uploaded_image,
                            st.session_state.current_guid
                        )
                        st.success("Processing complete!")
                    except Exception as e:
                        st.error(f"Error applying Watershed: {str(e)}")
            
            if st.session_state.watershed_result is not None:
                st.subheader("Watershed Segmentation Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        st.session_state.watershed_result["display_image"], 
                        caption="Watershed Segmentation", 
                        use_column_width=True
                    )
                
                with col2:
                    st.subheader("Metrics")
                    st.json(st.session_state.watershed_result["metrics"])
                    st.download_button(
                        label="Download Result",
                        data=open(st.session_state.watershed_result["image_path"], "rb").read(),
                        file_name=os.path.basename(st.session_state.watershed_result["image_path"]),
                        mime="image/png"
                    )
        
        with tab5:
            st.header("Region Growing")
            st.markdown("""
            Region growing starts from a seed point and grows a region by appending 
            neighboring pixels that fulfill a certain criteria.
            """)
            
            if 'region_growing_result' not in st.session_state:
                st.session_state.region_growing_result = None
                
            tolerance = st.slider("Tolerance", 1, 50, 10)
            
            if st.button("Apply Region Growing"):
                with st.spinner("Processing..."):
                    try:
                        st.session_state.region_growing_result = apply_region_growing(
                            st.session_state.uploaded_image,
                            tolerance=tolerance,
                            guid=st.session_state.current_guid
                        )
                        st.success("Processing complete!")
                    except Exception as e:
                        st.error(f"Error applying Region Growing: {str(e)}")
            
            if st.session_state.region_growing_result is not None:
                st.subheader("Region Growing Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        st.session_state.region_growing_result["display_image"], 
                        caption=f"Region Growing (Tolerance={tolerance})", 
                        use_column_width=True
                    )
                
                with col2:
                    st.subheader("Metrics")
                    st.json(st.session_state.region_growing_result["metrics"])
                    st.download_button(
                        label="Download Result",
                        data=open(st.session_state.region_growing_result["image_path"], "rb").read(),
                        file_name=os.path.basename(st.session_state.region_growing_result["image_path"]),
                        mime="image/png"
                    )
        
        with tab6:
            st.header("Algorithm Comparison")
            st.markdown("""
            Compare the results of different algorithms quantitatively.
            """)
            
            # Collect all results
            results = {}
            if st.session_state.otsu_result is not None:
                results["Otsu Thresholding"] = st.session_state.otsu_result
            if st.session_state.kmeans_result is not None:
                results["K-means Clustering"] = st.session_state.kmeans_result
            if st.session_state.pca_result is not None:
                results["PCA"] = st.session_state.pca_result
            if st.session_state.watershed_result is not None:
                results["Watershed"] = st.session_state.watershed_result
            if st.session_state.region_growing_result is not None:
                results["Region Growing"] = st.session_state.region_growing_result
            
            if results:
                if st.button("Perform Comparison"):
                    with st.spinner("Comparing algorithms..."):
                        try:
                            comparison_results = compare_algorithms(results)
                            st.session_state.comparison_results = comparison_results
                            st.success("Comparison complete!")
                        except Exception as e:
                            st.error(f"Error performing comparison: {str(e)}")
                
                if 'comparison_results' in st.session_state and st.session_state.comparison_results:
                    comparison = st.session_state.comparison_results
                    
                    # Display individual algorithm metrics
                    st.subheader("Algorithm Metrics")
                    metrics_df = []
                    for algo_name, metrics in comparison["algorithms"].items():
                        row = {"Algorithm": algo_name}
                        row.update(metrics)
                        metrics_df.append(row)
                    
                    st.dataframe(metrics_df)
                    
                    # Display comparison matrix
                    st.subheader("Pairwise Comparison (Percentage Difference)")
                    for metric, comparisons in comparison["comparison_matrix"].items():
                        st.write(f"**{metric.replace('_', ' ').title()}**")
                        # Convert to format suitable for dataframe
                        algo_names = list(comparisons.keys())
                        comparison_data = []
                        for algo1 in algo_names:
                            row = [comparisons[algo1][algo2] for algo2 in algo_names]
                            comparison_data.append(row)
                        
                        # Create dataframe
                        import pandas as pd
                        df = pd.DataFrame(comparison_data, columns=algo_names, index=algo_names)
                        st.dataframe(df)
                    
                    # Display summary statistics
                    st.subheader("Summary Statistics")
                    summary_data = []
                    for metric, stats in comparison["summary"].items():
                        row = {"Metric": metric.replace('_', ' ').title()}
                        row.update({k: round(v, 4) for k, v in stats.items()})
                        summary_data.append(row)
                    
                    import pandas as pd
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df)
            else:
                st.info("Please run at least one algorithm to enable comparison.")


if __name__ == "__main__":
    main()