"""
3D Visualization Module

This module provides functions for creating interactive 3D visualizations
using Plotly, designed for display in Streamlit.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Optional, Tuple


# Default color palette for clusters (RGB tuples)
DEFAULT_COLORS = [
    (31, 119, 180),    # Blue
    (255, 127, 14),    # Orange
    (44, 160, 44),     # Green
    (214, 39, 40),     # Red
    (148, 103, 189),   # Purple
    (140, 86, 75),     # Brown
    (227, 119, 194),   # Pink
    (127, 127, 127),   # Gray
    (188, 189, 34),    # Yellow-green
    (23, 190, 207),    # Cyan
]


def rgb_to_plotly_color(rgb: Tuple[int, int, int], opacity: float = 0.8) -> str:
    """
    Convert RGB tuple to Plotly color string.
    
    Args:
        rgb: Tuple of (R, G, B) values (0-255)
        opacity: Opacity value (0-1)
        
    Returns:
        str: Plotly-compatible color string
    """
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"


def create_mesh3d_trace(vertices: np.ndarray,
                        faces: np.ndarray,
                        color: Tuple[int, int, int] = (31, 119, 180),
                        opacity: float = 0.8,
                        name: str = "Mesh") -> go.Mesh3d:
    """
    Create a Plotly Mesh3d trace from vertices and faces.
    
    Args:
        vertices: Nx3 array of vertex coordinates (z, y, x from marching cubes)
        faces: Mx3 array of triangle face indices
        color: RGB tuple for mesh color
        opacity: Mesh opacity (0-1)
        name: Name for the trace (shown in legend)
        
    Returns:
        go.Mesh3d: Plotly Mesh3d trace
    """
    if len(vertices) == 0 or len(faces) == 0:
        # Return empty trace
        return go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], name=name)
    
    # Note: marching cubes returns (z, y, x) coordinates
    # We swap to (x, y, z) for standard 3D visualization
    return go.Mesh3d(
        x=vertices[:, 2],  # X (was width)
        y=vertices[:, 1],  # Y (was height)
        z=vertices[:, 0],  # Z (was depth/slices)
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=rgb_to_plotly_color(color, opacity),
        opacity=opacity,
        name=name,
        showlegend=True,
        flatshading=True,
        lighting=dict(
            ambient=0.4,
            diffuse=0.8,
            specular=0.3,
            roughness=0.5,
        ),
        lightposition=dict(x=100, y=200, z=300)
    )


def create_3d_figure(meshes: List[dict],
                     colors: Optional[List[Tuple[int, int, int]]] = None,
                     title: str = "3D Model",
                     show_axes: bool = True,
                     background_color: str = "rgb(20, 20, 30)") -> go.Figure:
    """
    Create a Plotly figure with multiple 3D meshes.
    
    Args:
        meshes: List of mesh dictionaries (from volume3d.generate_mesh_for_cluster)
        colors: Optional list of RGB tuples for each mesh
        title: Figure title
        show_axes: Whether to show axis labels
        background_color: Background color of the 3D scene
        
    Returns:
        go.Figure: Plotly figure object
    """
    if colors is None:
        colors = DEFAULT_COLORS
    
    traces = []
    
    for i, mesh in enumerate(meshes):
        if mesh["num_vertices"] == 0:
            continue
            
        color = colors[i % len(colors)]
        cluster_id = mesh.get("cluster_id", i)
        volume_pct = mesh.get("volume_percentage", 0)
        
        trace = create_mesh3d_trace(
            vertices=mesh["vertices"],
            faces=mesh["faces"],
            color=color,
            opacity=0.8,
            name=f"Cluster {cluster_id} ({volume_pct:.1f}%)"
        )
        traces.append(trace)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Update layout for 3D visualization
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(
                title="Width" if show_axes else "",
                showbackground=True,
                backgroundcolor=background_color,
                gridcolor="rgb(50, 50, 60)",
                showticklabels=show_axes
            ),
            yaxis=dict(
                title="Height" if show_axes else "",
                showbackground=True,
                backgroundcolor=background_color,
                gridcolor="rgb(50, 50, 60)",
                showticklabels=show_axes
            ),
            zaxis=dict(
                title="Depth (Slices)" if show_axes else "",
                showbackground=True,
                backgroundcolor=background_color,
                gridcolor="rgb(50, 50, 60)",
                showticklabels=show_axes
            ),
            bgcolor=background_color,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='data'  # Preserve actual proportions
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgb(30, 30, 40)",
    )
    
    return fig


def create_single_cluster_figure(mesh: dict,
                                  color: Tuple[int, int, int] = (31, 119, 180),
                                  title: str = "3D Cluster") -> go.Figure:
    """
    Create a Plotly figure for a single cluster mesh.
    
    Args:
        mesh: Mesh dictionary from volume3d
        color: RGB tuple for mesh color
        title: Figure title
        
    Returns:
        go.Figure: Plotly figure object
    """
    return create_3d_figure(
        meshes=[mesh],
        colors=[color],
        title=title
    )


def get_cluster_color(cluster_id: int, 
                      custom_colors: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    """
    Get the color for a specific cluster.
    
    Args:
        cluster_id: The cluster ID
        custom_colors: Optional array of custom RGB colors from K-means
        
    Returns:
        Tuple[int, int, int]: RGB color tuple
    """
    if custom_colors is not None and cluster_id < len(custom_colors):
        return tuple(custom_colors[cluster_id].tolist())
    return DEFAULT_COLORS[cluster_id % len(DEFAULT_COLORS)]
