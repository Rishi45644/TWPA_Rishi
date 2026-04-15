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

def find_parameters(filename):
    """
    This function takes a file, specifically the hdf5 from which we get our data for the 
    TWPA and takes the specific parameters that we need to know for the optimization code. 
    Specifically, we want the values for the pump power, pump frequency and bias current, 
    and returns these values in a list respectively. 
    """
    file = h5py.File(filename, 'r')
    params_txt = (file['Parameters']['PHYSICAL_PARAMETERS']['Description'][()]).decode("utf-8")

    prev = None
    for i in params_txt.split("_"):
        if i.endswith('V'):
            bias_current = float(i[0:i.find('V')])
        elif i.endswith('GHz'):
            pump_freq = float(i[0:i.find('G')])
        elif i.endswith('dBm'):
            pump_power = float(prev)
        prev = i
        
    return [pump_power, pump_freq, bias_current]

def find_mean_ripple(filename, csv=False, start_band=4000000000.0, stop_band=8000000000.0, quantity_type=False):
    '''
    This function takes the filename, whether or not its a CSV, whether or not you want
    the mean or standard deviation of the gain ripples, and the bandwidth in which you
    want to calculate the quantity. This is one of the possible quantity choices that 
    you could take for optimization of the parameter space. It spits out the mean of 
    the gain ripple amplitudes, but could also return the standard deviation of the gain 
    ripple amplitudes if quanitity_type == True.
    '''
    if (csv == False): 
        file = h5py.File(filename, 'r')
        freq_start = file['Parameters']['VNA_PARAMETERS']['freq_start'][0]
        freq_stop = file['Parameters']['VNA_PARAMETERS']['freq_stop'][0]
        probe_pts = file['Parameters']['VNA_PARAMETERS']['probe_pts'][()]
    
        freqs = np.linspace(freq_start, freq_stop, probe_pts)
        s21 = np.asarray(file['sweep_dataframe']).flatten()    
    else:
        df = pd.read_csv(filename, header=None)
        freqs = df[0].to_numpy()
        s21 = df[1].to_numpy()
        
    mask = (freqs >= start_band) & (freqs <= stop_band)
    freqs_window = freqs[mask]
    s21_window = s21[mask]
    
    # Applying the Savitzky-Golay Smoothing Function
    s21_sg = savgol_filter(s21_window, window_length=200, polyorder=2)
    
    peaks, _ = find_peaks(s21_window, prominence=0.0001)
    troughs, _ = find_peaks(-s21_window, prominence=0.0001)
        
    ripple_amplitudes = []
    for i in range(0, len(peaks)):
        ripple_amplitudes.append(s21_window[peaks[i]] - s21_sg[peaks[i]])
    
    for i in range(0, len(troughs)):
        ripple_amplitudes.append(s21_window[troughs[i]] - s21_sg[troughs[i]])
    
    if (quantity_type == True):
        return np.abs(np.std(ripple_amplitudes))
    else:
        return np.abs(np.mean(ripple_amplitudes))

def find_gain_quantity(filename_pumpon, filename_pumpoff, csv=False, start_band=4000000000.0, stop_band=8000000000.0):
    '''
    This function takes the filenames of the two main files, the pump on and pump off, 
    datasets and used them to calculate a quanitity which we are describing as the overall
    gain over a specific bandwidth specified by the start and stop bands. This 'gain'
    is the integral of the pump on minus the pump off datasets. 
    '''
    if (csv == False): 
        file_on = h5py.File(filename_pumpon, 'r')
        freq_start_on = file_on['Parameters']['VNA_PARAMETERS']['freq_start'][0]
        freq_stop_on = file_on['Parameters']['VNA_PARAMETERS']['freq_stop'][0]
        probe_pts_on = file_on['Parameters']['VNA_PARAMETERS']['probe_pts'][()]
    
        freqs_on = np.linspace(freq_start_on, freq_stop_on, probe_pts_on)
        s21_on = np.asarray(file_on['sweep_dataframe']).flatten()    
        
        file_off = h5py.File(filename_pumpoff, 'r')
        freq_start_off = file_off['Parameters']['VNA_PARAMETERS']['freq_start'][0]
        freq_stop_off = file_off['Parameters']['VNA_PARAMETERS']['freq_stop'][0]
        probe_pts_off = file_off['Parameters']['VNA_PARAMETERS']['probe_pts'][()]
    
        freqs_off = np.linspace(freq_start_off, freq_stop_off, probe_pts_off)
        s21_off = np.asarray(file_off['sweep_dataframe']).flatten()    
    else:
        df_on = pd.read_csv(filename_pumpon, header=None)
        freqs_on = df_on[0].to_numpy()
        s21_on = df_on[1].to_numpy()
        
        df_off = pd.read_csv(filename_pumpoff, header=None)
        freqs_off = df_off[0].to_numpy()
        s21_off = df_off[1].to_numpy()
        
    mask = (freqs_on >= start_band) & (freqs_on <= stop_band)
    freqs_on_window = freqs_on[mask]
    s21_on_window = s21_on[mask]
    freqs_off_window = freqs_off[mask]
    s21_off_window = s21_off[mask]
    
    gain = s21_on_window - s21_off_window 
    return integrate.simpson(gain, x=freqs_on_window)

def get_parameter_data(folder_path):
    '''
    This function takes the folder path in which all of the TWPA data files exist and sweeps
    through them for the correct type of file that we want (Magnitude/MLOG) and with the 
    pump on. It then uses the find_parameters and find_mean_ripple functions to calculate the 
    parameters for each file in the folder and append it to a list of the correct type. Then
    a list of np arrays of each of the parameters and the quantity is returned. 
    '''
    param_1 = []    ## Pump Power
    param_2 = []    ## Pump Frequency 
    param_3 = []    ## Bias Current
    quantity = []   ## Quantity (Gain Ripple Amplitude)
    
    folder_path = Path(folder_path)
    for f in folder_path.glob("*.hdf5"):
        file = h5py.File(f, 'r')
        pump_on = (file['Parameters']['EQUIPMENT_PARAMETERS']['pump_on'][()])
        measurement = ((file['Parameters']['VNA_PARAMETERS']['measurement_type'][()]).decode('utf-8'))        
        if ((pump_on == True) and (measurement == 'MLOG')):
            pp, pf, bc = find_parameters(f)
            param_1.append(pp)
            param_2.append(pf)
            param_3.append(bc)
            quantity.append(find_mean_ripple(f))
    
    return [np.array(param_1), np.array(param_2), np.array(param_3), np.array(quantity)]

def get_parameter_data_gain(folder_path): 
    '''
    This function takes the folder path in which all of the TWPA data files exist and sweeps
    through them for the correct type of file that we want (Magnitude/MLOG) and with the 
    pump on. It then uses the find_parameters and find_mean_ripple functions to calculate the 
    parameters for each file in the folder and append it to a list of the correct type. Then
    a list of np arrays of each of the parameters and the quantity is returned. 
    '''
    param_1 = []    ## Pump Power
    param_2 = []    ## Pump Frequency 
    param_3 = []    ## Bias Current
    quantity = []   ## Quantity (Gain)
    
    folder_path = Path(folder_path)
    files = sorted(folder_path.glob("*.hdf5")) 
    for i, f in enumerate(files):
        file_on = h5py.File(f, 'r')
        pump_on = (file_on['Parameters']['EQUIPMENT_PARAMETERS']['pump_on'][()])
        measurement = ((file_on['Parameters']['VNA_PARAMETERS']['measurement_type'][()]).decode('utf-8'))        
        if ((pump_on == False) and (measurement == 'MLOG')):
            if (i + 2) <= len(files):
                j = files[i+2]
                pp, pf, bc = find_parameters(f)
                param_1.append(pp)
                param_2.append(pf)
                param_3.append(bc)
                quantity.append(find_gain_quantity(j, f))
    
    return [np.array(param_1), np.array(param_2), np.array(param_3), np.array(quantity)]

def mls_interpolation(points, values, query_points, radius=0.5, poly_degree=1):
    """
    This function uses Moving Least Squares (MLS) interpolation.
    It finds the nearest neighbours within a radius and fits a 
    local polynomial (constant, linear, quadratic, or cubic)
    weighted by a Gaussian distance kernel.
    
    ---- Inputs ----
    points         :(N, 3) array of Sample Points (Data)
    values         :(N,) array of Function Values (Data)
    query_points   :(M, 3) array of query locations (Interpolated)
    radius         :Search radius for nearest neighbors
    poly_degree    :Degree of polynomial (0=const, 1=linear, 2=quad, 3=cubic)
    
    ---- Outputs ----
    Fq             :(M,) Returns the values of the interpolated points
    """
    tree = cKDTree(points)
    Fq = np.zeros(len(query_points))
    
    for i, q in enumerate(query_points):
        # find neighbors in radius
        idx = tree.query_ball_point(q, r=radius)
        if len(idx) == 0:
            Fq[i] = np.nan
            continue
        
        X = (points[idx] - q) / radius  # normalize local coordinates
        w = np.exp(-np.sum(X**2, axis=1) / radius**2)
        y = values[idx]

        # polynomial basis up to cubic
        if poly_degree == 0:
            A = np.ones((len(idx), 1))
        elif poly_degree == 1:
            A = np.column_stack([np.ones(len(idx)), X])
        elif poly_degree == 2:
            dx, dy, dz = X[:, 0], X[:, 1], X[:, 2]
            A = np.column_stack([np.ones(len(idx)), dx, dy, dz,
                                dx**2, dy**2, dz**2, dx*dy, dy*dz, dz*dx])
        elif poly_degree == 3:
            dx, dy, dz = X[:, 0], X[:, 1], X[:, 2]
            A = np.column_stack([
                np.ones(len(idx)), dx, dy, dz,
                dx**2, dy**2, dz**2, dx*dy, dy*dz, dz*dx,
                dx**3, dy**3, dz**3,
                dx**2*dy, dx**2*dz, dy**2*dx, dy**2*dz, dz**2*dx, dz**2*dy,
                dx*dy*dz
            ])
        else:
            raise NotImplementedError("Degree > 3 not supported")

        # Weighted least squares with regularization
        W = np.diag(w)
        Aw = W @ A
        yw = W @ y
        lambda_reg = 1e-4
        A_reg = np.vstack([Aw, np.sqrt(lambda_reg)*np.eye(A.shape[1])])
        y_reg = np.hstack([yw, np.zeros(A.shape[1])])
        coeffs, *_ = np.linalg.lstsq(A_reg, y_reg, rcond=None)
        

        # Evaluate at q (local coord = 0) → only the constant term remains
        Fq[i] = coeffs[0]
        Fq = np.clip(Fq, np.nanmin(values), np.nanmax(values))
    return Fq

def combine_gif_figure(gif1, gif2, output):
    '''
    This function just basically takes 2 gifs and plots them side by side to create and save another gif of this.
    '''
    # Paths to your two GIFs
    gif1_path = gif1
    gif2_path = gif2
    output_path = output

    # Open both GIFs
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    # Ensure both have the same number of frames (optional but helpful)
    frames1 = [frame.copy() for frame in ImageSequence.Iterator(gif1)]
    frames2 = [frame.copy() for frame in ImageSequence.Iterator(gif2)]

    # If one is shorter, loop it
    if len(frames1) < len(frames2):
        frames1 *= len(frames2) // len(frames1)
    elif len(frames2) < len(frames1):
        frames2 *= len(frames1) // len(frames2)

    # Combine each pair of frames side by side
    combined_frames = []
    for f1, f2 in zip(frames1, frames2):
        # Ensure same height
        h = max(f1.height, f2.height)
        w = f1.width + f2.width
        new_frame = Image.new("RGBA", (w, h))
        new_frame.paste(f1, (0, 0))
        new_frame.paste(f2, (f1.width, 0))
        combined_frames.append(new_frame)

    # Save the combined frames as a single GIF
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=gif1.info["duration"],  # Keep same speed
        loop=0,                          # Infinite loop
        disposal=2
    )
    print(f"✅ Combined GIF saved to {output_path}")
    
