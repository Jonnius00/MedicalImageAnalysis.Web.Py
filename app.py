import streamlit as st
import cv2
import numpy as np
from utils.dicom_loader import load_medical_image
from utils.image_io import create_guid, load_image_stack
import os

# Import services
from services.otsu import apply_otsu_thresholding
from services.kmeans import apply_kmeans_clustering, apply_kmeans_to_stack
from services.pca import apply_pca
from services.watershed import apply_watershed
from services.region_growing import apply_region_growing
from services.comparison import compare_algorithms
from services.volume3d import generate_mesh_for_cluster, generate_all_cluster_meshes
from services.visualization3d import create_3d_figure, get_cluster_color

# Set page configuration
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon=":hospital:",
    layout="wide"
)


def main():
    st.title("Medical Image Analysis Suite")
    
    # Sidebar for mode selection
    st.sidebar.title("Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Select Mode:",
        ["Single Image Analysis", "3D Model Generation"],
        help="Choose between analyzing a single image or generating a 3D model from image slices"
    )
    
    if analysis_mode == "Single Image Analysis":
        run_single_image_analysis()
    else:
        run_3d_model_generation()


def run_single_image_analysis():
    """Handle single image analysis workflow (existing functionality)."""
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


def run_3d_model_generation():
    """Handle 3D model generation from image stack workflow."""
    st.markdown("""
    ## 3D Model Generation
    
    Generate a 3D model from a stack of medical image slices (CT or MRI scans).
    
    **Workflow:**
    1. Select or upload a dataset of image slices
    2. Preview the slices
    3. Apply K-means clustering to segment the images
    4. Generate and visualize the 3D model
    """)
    
    # Initialize session state for 3D workflow
    if 'image_stack' not in st.session_state:
        st.session_state.image_stack = None
    if 'stack_filenames' not in st.session_state:
        st.session_state.stack_filenames = None
    if 'clustered_stack' not in st.session_state:
        st.session_state.clustered_stack = None
    
    st.markdown("---")
    st.subheader("Step 1: Load Image Stack")
    
    # Dataset selection
    dataset_option = st.radio(
        "Choose data source:",
        ["Sample Dataset", "Upload Custom Images"],
        horizontal=True
    )
    
    if dataset_option == "Sample Dataset":
        # Predefined sample datasets
        sample_datasets = {
            "Heart MRI (11 slices)": "Heart_PNGs",
            "Chest CT (8 slices)": "CT_Breast_chest_PNGs"
        }
        
        selected_dataset = st.selectbox(
            "Select a sample dataset:",
            list(sample_datasets.keys())
        )
        
        folder_path = sample_datasets[selected_dataset]
        
        if st.button("Load Dataset", type="primary"):
            if os.path.exists(folder_path):
                with st.spinner(f"Loading {selected_dataset}..."):
                    try:
                        image_stack, filenames = load_image_stack(folder_path)
                        st.session_state.image_stack = image_stack
                        st.session_state.stack_filenames = filenames
                        st.session_state.clustered_stack = None  # Reset clustered data
                        st.success(f"✅ Loaded {len(filenames)} slices successfully!")
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")
            else:
                st.error(f"Dataset folder not found: {folder_path}")
    
    else:
        # Custom file upload
        uploaded_files = st.file_uploader(
            "Upload image slices (PNG, JPG)",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="stack_uploader"
        )
        
        if uploaded_files and len(uploaded_files) > 1:
            if st.button("Load Uploaded Images", type="primary"):
                with st.spinner("Loading uploaded images..."):
                    try:
                        # Save uploaded files temporarily
                        temp_dir = "temp_upload"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_paths = []
                        
                        for uploaded_file in uploaded_files:
                            temp_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            temp_paths.append(temp_path)
                        
                        # Load as stack
                        from utils.image_io import load_image_stack_from_files
                        image_stack, filenames = load_image_stack_from_files(temp_paths)
                        st.session_state.image_stack = image_stack
                        st.session_state.stack_filenames = filenames
                        st.session_state.clustered_stack = None
                        
                        # Clean up temp files
                        for temp_path in temp_paths:
                            os.remove(temp_path)
                        os.rmdir(temp_dir)
                        
                        st.success(f"✅ Loaded {len(filenames)} slices successfully!")
                    except Exception as e:
                        st.error(f"Error loading images: {str(e)}")
        elif uploaded_files and len(uploaded_files) == 1:
            st.warning("Please upload at least 2 images to create a 3D stack.")
    
    # Display loaded stack info and preview
    if st.session_state.image_stack is not None:
        stack = st.session_state.image_stack
        filenames = st.session_state.stack_filenames
        
        st.markdown("---")
        st.subheader("Step 2: Preview Image Stack")
        
        # Stack information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Slices", stack.shape[0])
        with col2:
            st.metric("Image Height", stack.shape[1])
        with col3:
            st.metric("Image Width", stack.shape[2])
        
        # Slice preview slider
        slice_idx = st.slider(
            "Browse slices:",
            0, stack.shape[0] - 1, 0,
            key="slice_preview_slider"
        )
        
        # Display current slice
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                stack[slice_idx],
                caption=f"Slice {slice_idx + 1}/{stack.shape[0]}: {filenames[slice_idx]}",
                use_column_width=True
            )
        
        with col2:
            # Show thumbnail grid of all slices
            st.write("**All Slices Overview:**")
            # Create a grid of thumbnails
            cols_per_row = 4
            rows_needed = (stack.shape[0] + cols_per_row - 1) // cols_per_row
            
            for row in range(rows_needed):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    slice_num = row * cols_per_row + col_idx
                    if slice_num < stack.shape[0]:
                        with cols[col_idx]:
                            # Resize for thumbnail
                            thumbnail = cv2.resize(stack[slice_num], (100, 100))
                            st.image(thumbnail, caption=f"#{slice_num + 1}", width=100)
        
        st.markdown("---")
        st.subheader("Step 3: Apply K-means Clustering")
        
        st.markdown("""
        K-means clustering will segment each slice into K distinct regions based on pixel intensity.
        This separates different tissue types (e.g., background, soft tissue, bone).
        """)
        
        # K-means parameters
        col1, col2 = st.columns([1, 2])
        with col1:
            k_value = st.slider(
                "Number of clusters (K):",
                min_value=2,
                max_value=6,
                value=3,
                help="Higher K = more segments. Start with 2-3 for basic tissue separation."
            )
        
        with col2:
            st.info(f"� **K={k_value}** will create {k_value} distinct regions in each slice.")
        
        # Process button
        if st.button("🔬 Apply K-means to All Slices", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"Processing slice {current}/{total}...")
            
            with st.spinner("Applying K-means clustering..."):
                try:
                    result = apply_kmeans_to_stack(
                        st.session_state.image_stack,
                        k=k_value,
                        progress_callback=update_progress
                    )
                    st.session_state.clustered_stack = result
                    st.session_state.kmeans_k = k_value
                    progress_bar.progress(1.0)
                    status_text.text("✅ Clustering complete!")
                    st.success(f"Successfully clustered {stack.shape[0]} slices into {k_value} regions!")
                except Exception as e:
                    st.error(f"Error during clustering: {str(e)}")
        
        # Display clustered results
        if st.session_state.clustered_stack is not None:
            result = st.session_state.clustered_stack
            
            st.markdown("### Clustering Results")
            
            # Display metrics
            with st.expander("📊 Clustering Metrics", expanded=True):
                metrics = result["metrics"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clusters", metrics["num_clusters"])
                with col2:
                    st.metric("Total Voxels", f"{metrics['total_voxels']:,}")
                with col3:
                    st.metric("Slices Processed", metrics["num_slices"])
                
                st.write("**Cluster Volumes (% of total):**")
                for cluster, volume in metrics["cluster_volumes_percent"].items():
                    st.write(f"  - {cluster}: {volume}%")
                
                st.write("**Cluster Centers (intensity values):**")
                centers = metrics["cluster_centers"]
                st.write(f"  {centers}")
            
            # Side-by-side comparison slider
            st.markdown("### Compare Original vs Clustered")
            
            compare_slice_idx = st.slider(
                "Select slice to compare:",
                0, stack.shape[0] - 1, 0,
                key="compare_slice_slider"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Slice**")
                st.image(
                    stack[compare_slice_idx],
                    caption=f"Original - Slice {compare_slice_idx + 1}",
                    use_column_width=True
                )
            
            with col2:
                st.write("**Clustered Slice**")
                st.image(
                    result["colored_stack"][compare_slice_idx],
                    caption=f"K-means (K={st.session_state.kmeans_k}) - Slice {compare_slice_idx + 1}",
                    use_column_width=True
                )
            
            # Cluster color legend
            st.write("**Cluster Color Legend:**")
            colors = result["colors"]
            centers = result["centers"].flatten()
            legend_cols = st.columns(len(colors))
            for i, (col, color) in enumerate(zip(legend_cols, colors)):
                with col:
                    # Create a small color swatch
                    swatch = np.full((30, 60, 3), color, dtype=np.uint8)
                    st.image(swatch, caption=f"Cluster {i}\n(intensity: {centers[i]:.0f})", width=60)
        
        st.markdown("---")
        st.subheader("Step 4: Generate 3D Model")
        
        # Initialize 3D mesh session state
        if 'meshes_3d' not in st.session_state:
            st.session_state.meshes_3d = None
        
        if st.session_state.clustered_stack is not None:
            st.markdown("""
            Generate a 3D surface mesh from the clustered volume using the **Marching Cubes** algorithm.
            Each cluster will become a separate 3D object that you can view interactively.
            """)
            
            # Mesh generation options
            with st.expander("⚙️ Mesh Generation Options", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    smooth_mesh = st.checkbox(
                        "Apply smoothing",
                        value=True,
                        help="Gaussian smoothing reduces staircase artifacts"
                    )
                    
                with col2:
                    if smooth_mesh:
                        smooth_sigma = st.slider(
                            "Smoothing intensity",
                            0.5, 3.0, 1.0, 0.5,
                            help="Higher = smoother but less detail"
                        )
                    else:
                        smooth_sigma = 1.0
                
                # Cluster selection
                num_clusters = st.session_state.clustered_stack["metrics"]["num_clusters"]
                
                cluster_options = st.multiselect(
                    "Select clusters to include in 3D model:",
                    options=list(range(num_clusters)),
                    default=list(range(num_clusters)),
                    format_func=lambda x: f"Cluster {x} ({st.session_state.clustered_stack['metrics']['cluster_volumes_percent'].get(f'cluster_{x}', 0)}%)"
                )
            
            # Generate 3D model button
            if st.button("🎲 Generate 3D Model", type="primary"):
                if not cluster_options:
                    st.warning("Please select at least one cluster.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    labels_stack = st.session_state.clustered_stack["labels_stack"]
                    
                    with st.spinner("Generating 3D meshes..."):
                        try:
                            meshes = []
                            for i, cluster_id in enumerate(cluster_options):
                                status_text.text(f"Processing cluster {cluster_id}...")
                                
                                mesh = generate_mesh_for_cluster(
                                    labels_stack,
                                    cluster_id=cluster_id,
                                    smooth=smooth_mesh,
                                    smooth_sigma=smooth_sigma
                                )
                                meshes.append(mesh)
                                progress_bar.progress((i + 1) / len(cluster_options))
                            
                            st.session_state.meshes_3d = meshes
                            status_text.text("✅ 3D model generated!")
                            st.success(f"Generated {len(meshes)} cluster meshes!")
                            
                        except Exception as e:
                            st.error(f"Error generating mesh: {str(e)}")
            
            # Display 3D visualization
            if st.session_state.meshes_3d is not None:
                meshes = st.session_state.meshes_3d
                
                st.markdown("### 🎮 Interactive 3D Visualization")
                st.info("💡 **Tip:** Click and drag to rotate. Scroll to zoom. Right-click to pan.")
                
                # Get colors from K-means result
                kmeans_colors = st.session_state.clustered_stack.get("colors", None)
                
                # Convert to list of tuples for Plotly
                if kmeans_colors is not None:
                    color_list = [tuple(c.tolist()) for c in kmeans_colors]
                else:
                    color_list = None
                
                # Create and display 3D figure
                fig = create_3d_figure(
                    meshes=meshes,
                    colors=color_list,
                    title="3D Reconstructed Model from Medical Image Slices"
                )
                
                st.plotly_chart(fig, use_container_width=True, height=600)
                
                # Mesh statistics
                with st.expander("📊 Mesh Statistics", expanded=False):
                    for mesh in meshes:
                        cluster_id = mesh.get("cluster_id", "?")
                        st.write(f"**Cluster {cluster_id}:**")
                        st.write(f"  - Vertices: {mesh['num_vertices']:,}")
                        st.write(f"  - Triangles: {mesh['num_faces']:,}")
                        st.write(f"  - Volume: {mesh.get('volume_percentage', 0):.2f}% of total")
                        st.write("---")
        else:
            st.info("⬆️ Please complete **Step 3** (K-means Clustering) first to generate a 3D model.")
        
        # Discussion Section
        st.markdown("---")
        st.subheader("📖 Discussion & Interpretation")
        
        with st.expander("**How This Works: The Complete Workflow**", expanded=True):
            st.markdown("""
            ### 1. Image Stack Loading
            Medical imaging techniques like **CT (Computed Tomography)** and **MRI (Magnetic Resonance Imaging)** 
            produce a series of 2D cross-sectional images called "slices." Each slice represents a thin layer 
            of the scanned body part at a specific depth.
            
            When we load these slices in order, they form a **3D volume** - like stacking photographs to 
            recreate a 3D object.
            
            ### 2. K-means Clustering (Segmentation)
            **K-means** is an unsupervised machine learning algorithm that groups pixels by their intensity values:
            
            - **K=2**: Separates the image into 2 regions (e.g., background vs. tissue)
            - **K=3**: Typically separates background, soft tissue, and denser structures
            - **K=4+**: Can distinguish between more tissue types but may over-segment
            
            The algorithm works by:
            1. Randomly placing K "center points" in the intensity space
            2. Assigning each pixel to its nearest center
            3. Moving centers to the average of their assigned pixels
            4. Repeating until stable
            
            ### 3. 3D Surface Generation (Marching Cubes)
            The **Marching Cubes algorithm** converts the clustered volume into a 3D mesh:
            
            1. It examines small cubes (8 voxels at a time) throughout the volume
            2. For each cube, it determines where the surface crosses based on which corners 
               are "inside" vs "outside" the cluster
            3. It generates triangles at these crossing points
            4. All triangles connect to form a continuous 3D surface
            
            **Smoothing** (Gaussian filter) is applied before mesh generation to reduce 
            "staircase" artifacts and create a more natural-looking surface.
            """)
        
        with st.expander("**Interpreting the Results**", expanded=False):
            st.markdown("""
            ### What Do the Clusters Represent?
            
            The clusters are based purely on **pixel intensity** (brightness), not anatomical knowledge. 
            However, different tissues typically have different intensities:
            
            | Cluster (typical) | CT Scan | MRI Scan |
            |-------------------|---------|----------|
            | Darkest (low intensity) | Air, lungs | Bone, air |
            | Medium intensity | Soft tissue, organs | Gray matter, organs |
            | Brightest (high intensity) | Bone, contrast agent | Fat, fluid |
            
            **Note:** The exact mapping depends on the imaging modality, scan parameters, and body region.
            
            ### Reading the 3D Model
            
            - **Separate colors** = Different tissue/density regions
            - **Surface smoothness** = Affected by smoothing parameter
            - **Holes or gaps** = May indicate transition zones between tissues
            - **Volume percentage** = Relative size of each region in the scan
            
            ### Tips for Better Results
            
            1. **Start with K=2 or K=3** for basic tissue separation
            2. **Increase K** if you need finer detail (but may cause over-segmentation)
            3. **Use smoothing** for cleaner visualizations
            4. **Hide background cluster** (usually the largest) to see internal structures
            """)
        
        with st.expander("**Limitations & Considerations**", expanded=False):
            st.markdown("""
            ### Technical Limitations
            
            1. **Intensity-based only**: K-means clusters by brightness, not by anatomical structures. 
               Two different organs with similar density will be grouped together.
            
            2. **No spatial awareness**: The algorithm doesn't consider pixel location or connectivity. 
               Disconnected regions with similar intensity are assigned to the same cluster.
            
            3. **Slice spacing assumption**: The 3D model assumes equal spacing between slices. 
               Real medical scans may have different slice thicknesses.
            
            4. **2D slice artifacts**: The stacking of 2D slices can cause visible layer boundaries 
               in the 3D model, especially with few slices.
            
            ### When to Use This Tool
            
            ✅ **Good for:**
            - Educational visualization of medical scan structure
            - Quick 3D overview of scan contents
            - Understanding clustering algorithms
            - Proof-of-concept segmentation
            
            ❌ **Not suitable for:**
            - Clinical diagnosis
            - Precise anatomical measurements
            - Cases requiring anatomical knowledge
            
            ### Comparison with Professional Tools
            
            Professional medical imaging software uses:
            - **Atlas-based segmentation** (anatomical templates)
            - **Deep learning models** trained on labeled data
            - **Region growing with connectivity** constraints
            - **DICOM metadata** for correct spatial scaling
            
            This tool demonstrates the fundamental concepts using simpler algorithms.
            """)


if __name__ == "__main__":
    main()