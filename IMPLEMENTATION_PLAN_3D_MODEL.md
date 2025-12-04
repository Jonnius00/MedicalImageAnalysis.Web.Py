# Implementation Plan: 3D Model Generation from Medical Image Slices

## Project Overview

**Objective:** Extend the existing Medical Image Analysis web application to support:
1. Loading multiple image slices as a dataset
2. Applying K-means clustering to segmented slices
3. Generating a 3D model from clustered image stack
4. Interactive 3D visualization in the browser

**Technology Stack:**
- Python 3.12
- Streamlit (existing)
- OpenCV (existing)
- NumPy (existing)
- Plotly (new - for 3D visualization)
- scikit-image (existing - for marching cubes algorithm)

---

## Available Test Datasets

| Dataset | Files | Description |
|---------|-------|-------------|
| Heart MRI | 11 slices | `Heart_PNGs/img-00002-00001.png` to `img-00002-00011.png` |
| Chest CT | 8 slices | `CT_Breast_chest_PNGs/DicomExport_97.PNG` to `DicomExport_104.PNG` |

**Recommendation:** Start testing with the Heart MRI dataset (11 slices provide smoother 3D reconstruction).

---

## Implementation Phases

### Phase 1: Multi-Image Stack Loading
**Estimated time: 30-45 minutes**

**Goal:** Allow users to upload or select multiple images as a stack.

**Tasks:**
- [ ] 1.1 Create new utility function `load_image_stack()` in `utils/image_io.py`
  - Accept folder path or multiple uploaded files
  - Sort images by filename (natural sorting)
  - Load all images as grayscale
  - Return 3D NumPy array (slices × height × width)
  
- [ ] 1.2 Add UI component in `app.py` for stack selection
  - Option A: Upload multiple files at once
  - Option B: Select from predefined sample datasets (Heart, Chest)
  - Display thumbnail preview of all slices
  - Show stack dimensions (number of slices, resolution)

**Key Code Locations:**
- `utils/image_io.py` - add `load_image_stack()`
- `app.py` - add new tab "3D Model"

---

### Phase 2: Apply K-means to Image Stack
**Estimated time: 30 minutes**

**Goal:** Apply K-means clustering to each slice in the stack.

**Tasks:**
- [ ] 2.1 Create `apply_kmeans_to_stack()` function in `services/kmeans.py`
  - Loop through each slice
  - Apply existing `apply_kmeans_clustering()` logic
  - Return clustered 3D volume (same dimensions as input)
  - Also return cluster labels volume for 3D model generation

- [ ] 2.2 Add progress indicator in UI
  - Show progress bar during processing
  - Display "Processing slice X of Y"

**Algorithm Notes:**
- Use consistent K value across all slices
- Use same random seed for reproducibility
- Store cluster labels (0, 1, 2, ..., K-1) not colored images

---

### Phase 3: 3D Volume Creation
**Estimated time: 20 minutes**

**Goal:** Stack clustered slices into a 3D volume array.

**Tasks:**
- [ ] 3.1 Create new service file `services/volume3d.py`
  - Function `create_volume_from_stack(clustered_slices)` 
  - Input: List of 2D clustered label arrays
  - Output: 3D NumPy array (Z × Y × X)
  
- [ ] 3.2 Add binary mask extraction for each cluster
  - Function `extract_cluster_mask(volume, cluster_id)`
  - Returns binary 3D array where cluster == cluster_id

**Data Structure:**
```
volume_3d shape: (num_slices, height, width)
  - num_slices = Z axis (depth)
  - height = Y axis
  - width = X axis
```

---

### Phase 4: 3D Surface Mesh Generation
**Estimated time: 45 minutes**

**Goal:** Convert 3D volume to mesh for visualization.

**Tasks:**
- [ ] 4.1 Implement marching cubes algorithm
  - Use `skimage.measure.marching_cubes()` (already in requirements)
  - Input: Binary 3D volume (one cluster)
  - Output: Vertices and faces arrays
  
- [ ] 4.2 Add mesh smoothing (optional but recommended)
  - Simple Gaussian smoothing on volume before marching cubes
  - Improves visual quality

- [ ] 4.3 Handle multiple clusters
  - Generate separate mesh for each cluster
  - Assign different colors to each cluster mesh

**Key Function Signature:**
```python
def generate_mesh(volume_3d: np.ndarray, cluster_id: int = None) -> tuple:
    """
    Returns: (vertices, faces, normals)
    """
```

---

### Phase 5: 3D Visualization with Plotly
**Estimated time: 45-60 minutes**

**Goal:** Display interactive 3D model in Streamlit.

**Tasks:**
- [ ] 5.1 Add Plotly to requirements
  - Add `plotly>=5.18.0` to `requirements.txt`
  - Install in virtual environment

- [ ] 5.2 Create visualization service `services/visualization3d.py`
  - Function `create_3d_figure(vertices, faces, color)`
  - Use `plotly.graph_objects.Mesh3d`
  - Return Plotly figure object

- [ ] 5.3 Integrate with Streamlit UI
  - Use `st.plotly_chart()` to display
  - Add controls:
    - Cluster selector dropdown
    - Color picker (optional)
    - Show/hide individual clusters

**Plotly Mesh3d Example Structure:**
```python
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='red',
        opacity=0.8
    )
])
```

---

### Phase 6: UI Integration & Polish
**Estimated time: 30 minutes**

**Goal:** Create cohesive user experience.

**Tasks:**
- [ ] 6.1 Create new "3D Model" tab in main app
  - Step 1: Load image stack
  - Step 2: Configure K-means parameters
  - Step 3: Process and generate 3D model
  - Step 4: View and interact with 3D visualization

- [ ] 6.2 Add slice preview widget
  - Slider to browse through original slices
  - Side-by-side view: original vs clustered

- [ ] 6.3 Add discussion/explanation section
  - Brief explanation of the workflow
  - Interpretation guide for results

- [ ] 6.4 Add export functionality (optional)
  - Download 3D model as file (e.g., STL format)

---

## New Files to Create

| File | Purpose |
|------|---------|
| `services/volume3d.py` | 3D volume operations |
| `services/visualization3d.py` | Plotly 3D visualization |

## Files to Modify

| File | Changes |
|------|---------|
| `requirements.txt` | Add `plotly>=5.18.0` |
| `utils/image_io.py` | Add `load_image_stack()` function |
| `services/kmeans.py` | Add `apply_kmeans_to_stack()` function |
| `app.py` | Add "3D Model" tab with new workflow |

---

## Detailed File Structure After Implementation

```
MedicalImageAnalysis.Web.Py/
├── app.py                      # Modified - add 3D Model tab
├── requirements.txt            # Modified - add plotly
├── services/
│   ├── __init__.py
│   ├── kmeans.py              # Modified - add stack processing
│   ├── volume3d.py            # NEW - 3D volume operations
│   ├── visualization3d.py     # NEW - Plotly 3D viz
│   ├── otsu.py
│   ├── pca.py
│   ├── region_growing.py
│   ├── watershed.py
│   └── comparison.py
├── utils/
│   ├── __init__.py
│   ├── dicom_loader.py
│   └── image_io.py            # Modified - add stack loading
├── Heart_PNGs/                 # Sample dataset
└── CT_Breast_chest_PNGs/       # Sample dataset
```

---

## Implementation Order (Recommended)

```
Step 1: Update requirements.txt, install plotly
    ↓
Step 2: Implement load_image_stack() in utils/image_io.py
    ↓
Step 3: Implement apply_kmeans_to_stack() in services/kmeans.py
    ↓
Step 4: Create services/volume3d.py with volume operations
    ↓
Step 5: Create services/visualization3d.py with Plotly functions
    ↓
Step 6: Add "3D Model" tab to app.py
    ↓
Step 7: Test with Heart MRI dataset
    ↓
Step 8: Test with Chest CT dataset
    ↓
Step 9: Add discussion section and polish UI
```

---

## Key Dependencies

### Existing (no changes needed):
- `streamlit==1.38.0`
- `opencv-python-headless==4.10.0.84`
- `numpy==1.26.4`
- `scikit-image==0.22.0` (provides marching_cubes)

### New (to be added):
- `plotly>=5.18.0`

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Memory issues with large stacks | Limit max slices, resize images if needed |
| Slow processing | Add progress bars, optimize loops with NumPy |
| Poor 3D model quality | Adjust K value, add volume smoothing |
| Plotly not rendering | Ensure proper Streamlit-Plotly integration |

---

## Testing Checklist

- [ ] Load Heart MRI stack (11 slices)
- [ ] Load Chest CT stack (8 slices)
- [ ] Apply K-means with K=2 (binary segmentation)
- [ ] Apply K-means with K=3 (tissue separation)
- [ ] Generate 3D mesh from cluster 0
- [ ] Generate 3D mesh from cluster 1
- [ ] Verify 3D rotation/zoom works in browser
- [ ] Test with custom uploaded images

---

## Estimated Total Time

| Phase | Time |
|-------|------|
| Phase 1: Stack Loading | 30-45 min |
| Phase 2: K-means Stack | 30 min |
| Phase 3: Volume Creation | 20 min |
| Phase 4: Mesh Generation | 45 min |
| Phase 5: 3D Visualization | 45-60 min |
| Phase 6: UI Polish | 30 min |
| **Total** | **~3.5 - 4 hours** |

---

## Quick Start Commands

```powershell
# Activate virtual environment
.\MedIMgAnalyze.Web\Scripts\activate

# Install new dependency
pip install plotly>=5.18.0

# Run the application
streamlit run app.py
```

---

## Notes for Discussion Section (Phase 6)

When adding the discussion section to your website, consider including:

1. **Workflow Explanation**
   - How slices are loaded and sorted
   - How K-means clustering segments each slice
   - How slices are stacked into 3D volume
   - How marching cubes creates the mesh surface

2. **Parameter Impact**
   - Effect of K (number of clusters) on segmentation quality
   - Why certain K values work better for specific tissues

3. **Interpretation Guide**
   - What different clusters typically represent (background, soft tissue, bone, etc.)
   - How to identify organs/structures in the 3D model

4. **Limitations**
   - Dependence on image quality
   - Assumptions about slice spacing
   - Why results may vary between datasets

---

## Questions to Consider Later

1. Do you want to save the 3D model as a file (STL/OBJ)?
2. Do you need to adjust spacing between slices (some datasets have different slice thickness)?
3. Should we add volume measurements (e.g., cluster volume in mm³)?

---

*Document created: December 4, 2025*
*Project: Medical Image Analysis Web Application*
*Assignment: 1.4 - Clustering Big Data - 3D Model*
