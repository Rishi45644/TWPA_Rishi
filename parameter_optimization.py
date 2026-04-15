import numpy as np
import pandas as pd
import polars as pl
import scienceplots
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator, NearestNDInterpolator, RegularGridInterpolator
from scipy.optimize import differential_evolution
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import h5py
from scipy.signal import find_peaks
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq, fftshift, ifft, ifftshift
import math
from scipy import integrate
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from whittaker_eilers import WhittakerSmoother
from scipy.optimize import curve_fit
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image, ImageSequence
from scipy.spatial import cKDTree
from itertools import product
from numpy import load

plt.style.use(["science", "no-latex", "grid"])
plt.rcParams.update({'font.size': 12})
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{siunitx}"
plt.rcParams["figure.figsize"] = (14, 7) 

from .base_automation import mls_interpolation

def scatter_plotter(x, y, z, f, grid_size):
    '''
    This function takes the parameter data that we had calculated earlier and applies 
    a MLS interpolater on it. Then we plot the interpolated data on a grid with size 
    N, in a 3D Scatter plot along with the original data. 
    '''
    ## Getting and Interpolating Data
    points = np.stack([x, y, z], axis=-1)
    Xq, Yq, Zq = np.meshgrid(
        np.linspace(min(x), max(x), grid_size),
        np.linspace(min(y), max(y), grid_size),
        np.linspace(min(z), max(z), grid_size),
        indexing="ij"
    )
    query_points = np.stack([Xq.ravel(), Yq.ravel(), Zq.ravel()], axis=-1)
    
    Fq = mls_interpolation(points, f, query_points, radius=0.5, poly_degree=1)
    
    ## Finding a Maximum in the dataset
    print("Any NaNs in f:", np.isnan(Fq).any())
    print("Count of NaNs:", np.isnan(Fq).sum())
    
    mask = ~np.isnan(Fq)
    f_valid = Fq[mask]
    points_valid = query_points[mask]

    idx_max = np.argmax(-f_valid)
    max_value = f_valid[idx_max]
    max_point = points_valid[idx_max]
    
    print(max_point, max_value)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=points_valid[:, 0], 
        y=points_valid[:, 1], 
        z=points_valid[:, 2], 
        mode="markers", 
        marker=dict(size=3, 
                    color=f_valid, 
                    cmin=0, 
                    cmax=1.8, 
                    colorbar=dict(title="Gain Ripples"),
                    opacity=0.5
        )
    ))
    fig.show()
    
    # Plotting Real Original Data
    fig = go.Figure()
    
    # Plotting Interpolated Data
    fig.add_trace(go.Scatter3d(
        x=query_points[:, 0],
        y=query_points[:, 1],
        z=query_points[:, 2],
        mode="markers",
        marker=dict(size=3, 
                    color=Fq, 
                    colorscale="inferno", 
                    cmin = 0, 
                    cmax = 2, 
                    colorbar=dict(title="Gain Ripples"), 
                    opacity=0.5),
        # name="MLS Interpolated"
    ))
  
    fig.update_layout(scene=dict(
        xaxis_title='Pump Power (dBm)', 
        yaxis_title='Pump Frequency (GHz)', 
        zaxis_title='Bias Current (mA)'
    ))
    fig.show()
    
    
    fig = go.Figure()
    
    # Original sparse points
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=5, 
                    color=f, 
                    colorscale="inferno", 
                    cmin = 0, 
                    cmax = 2, 
                    colorbar=dict(title='Gain Ripples')),
        # name="Original Data"
    ))
    
    fig.update_layout(scene=dict(
        xaxis_title='Pump Power (dBm)', 
        yaxis_title='Pump Frequency (GHz)', 
        zaxis_title='Bias Current (mA)'
    ))
    
    fig.show()

def volume_plotter(x, y, z, f, isosurfaces, grid_size, radius=0.5, poly_degree=1):
    '''
    This function once again takes our parameter space data and applies an interpolater
    to it. However, instead of plotting a scatter plot we have to create a mesh grid
    on which the interpolater can be applied in order to create a volume plot. The 
    isosurfaces determine basically how many 'cuts' in the volume plot you want, more
    makes the volume plot more smoother in shape. 
    '''
    # Step 1: Prepare grid
    Xq, Yq, Zq = np.meshgrid(
        np.linspace(np.min(x), np.max(x), grid_size),
        np.linspace(np.min(y), np.max(y), grid_size),
        np.linspace(np.min(z), np.max(z), grid_size),
        indexing="ij"
    )
    query_points = np.stack([Xq.ravel(), Yq.ravel(), Zq.ravel()], axis=-1)

    # Step 2: Interpolate with MLS
    points = np.stack([x, y, z], axis=-1)
    Fq = mls_interpolation(points, f, query_points, radius=radius, poly_degree=poly_degree)
    Fq = np.array(Fq).reshape(Xq.shape)

    # Step 3: Handle NaNs or invalid values
    if np.all(np.isnan(Fq)):
        raise ValueError("All interpolated values are NaN — try increasing radius or lowering grid_size.")
    Fq = np.nan_to_num(Fq, nan=np.nanmean(Fq))

    # Determine visualization bounds
    fmin, fmax = np.percentile(Fq, [5, 95])  # avoid outliers

    # Step 4: Plot
    fig = go.Figure()

    # Volume rendering
    fig.add_trace(go.Volume(
        x=Xq.flatten(),
        y=Yq.flatten(),
        z=Zq.flatten(),
        value=Fq.flatten(),
        isomin=fmin,
        isomax=fmax,
        opacity=0.2,
        surface_count=15,
        colorscale='plasma',
        colorbar=dict(title='Interpolated Value')
    ))
    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X Parameter',
            yaxis_title='Y Parameter',
            zaxis_title='Z Parameter'
        ),
        title='3D MLS Interpolated Volume'
    )

    fig.show()
    
def slice_heatmaps_mls(x, y, z, f, grid_size=100, radius=0.5, poly_degree=1):
    """
    Generates heatmap slices along each axis from 3D MLS-interpolated data.
    Saves each slice as a PNG and compiles them into GIFs for each parameter.

    --- Inputs ---
    x, y, z        : (N,) coordinate arrays
    f              : (N,) function values
    grid_size      : Number of points along each axis for interpolation
    radius         : Search radius for MLS neighbors
    poly_degree    : Polynomial degree for local MLS fit
    """

    # Create directories for frames if they don't exist
    os.makedirs("total_frames/pump_power_ripples", exist_ok=True)
    os.makedirs("total_frames/pump_frequency_ripples", exist_ok=True)
    os.makedirs("total_frames/bias_current_ripples", exist_ok=True)

    # Step 1: 3D Grid
    Xgrid = np.linspace(min(x), max(x), grid_size)
    Ygrid = np.linspace(min(y), max(y), grid_size)
    Zgrid = np.linspace(min(z), max(z), grid_size)
    X, Y, Z = np.meshgrid(Xgrid, Ygrid, Zgrid, indexing='ij')
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # Step 2: Interpolation using MLS
    points = np.stack([x, y, z], axis=-1)
    grid_values = mls_interpolation(points, f, grid_points, radius=radius, poly_degree=poly_degree)
    grid_values = grid_values.reshape((grid_size, grid_size, grid_size))

    # Step 3: Create slices along each axis
    for i in range(grid_size):
        # Pump Power slices
        slice_pp = grid_values[i, :, :]
        fig = go.Figure(data=go.Heatmap(
            x=Y[0, :, 0],
            y=Z[0, 0, :],
            z=slice_pp.T,
            colorscale='inferno', 
            zmin = 0, 
            zmax = 2, 
            colorbar=dict(title='Gain Ripples (dB)')
        ))
        fig.update_layout(
            title=f"Slice at Pump Power = {X[i,0,0]:.2f} dBm",
            xaxis_title="Pump Frequency",
            yaxis_title="Bias Current"
        )
        fig.write_image(f"total_frames/pump_power_ripples/frame_{i:03d}.png")

        # Pump Frequency slices
        slice_pf = grid_values[:, i, :]
        fig = go.Figure(data=go.Heatmap(
            x=X[:, 0, 0],
            y=Z[0, 0, :],
            z=slice_pf.T,
            colorscale='inferno', 
            zmin = 0, 
            zmax = 2, 
            colorbar=dict(title='Gain Ripples (dB)')
        ))
        fig.update_layout(
            title=f"Slice at Pump Frequency = {Y[0,i,0]:.3f} GHz",
            xaxis_title="Pump Power",
            yaxis_title="Bias Current"
        )
        fig.write_image(f"total_frames/pump_frequency_ripples/frame_{i:03d}.png")

        # Bias Current slices
        slice_bc = grid_values[:, :, i]
        fig = go.Figure(data=go.Heatmap(
            x=X[:, 0, 0],
            y=Y[0, :, 0],
            z=slice_bc.T,
            colorscale='inferno',
            zmin = 0, 
            zmax = 2, 
            colorbar=dict(title='Gain Ripples (dB)')
        ))
        fig.update_layout(
            title=f"Slice at Bias Current = {Z[0,0,i]:.2f} mA",
            xaxis_title="Pump Power",
            yaxis_title="Pump Frequency"
        )
        fig.write_image(f"total_frames/bias_current_ripples/frame_{i:02d}.png")

    # Step 4: Compile GIFs
    for param, folder, outname, duration in [
        ("Pump Power", "total_frames/pump_power_ripples", "Pump_Power_Sweep_Ripples.gif", 300),
        ("Pump Frequency", "total_frames/pump_frequency_ripples", "Pump_Frequency_Sweep_Ripples.gif", 100),
        ("Bias Current", "total_frames/bias_current_ripples", "Bias_Current_Sweep_Ripples.gif", 200)
    ]:
        frames = [Image.open(os.path.join(folder, f))
                  for f in sorted(os.listdir(folder)) if f.endswith(".png")]
        if frames:
            frames[0].save(outname,
                           save_all=True,
                           append_images=frames[1:],
                           duration=duration,
                           loop=0)

