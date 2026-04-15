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

## This is the Time Domain Reflectometry Code

def get_data_tdr(filename_mag, filename_phase, csv=True, start_band=0, stop_band=1.8e10):
    '''
    This function takes the files necessary to get data that is usable for the time 
    domain reflectometry (TDR) block of code. It provides a start and stop band as 
    well such that the user can choose what frequencies want to be looked at. 
    
    --- Inputs ---
    filename_mag            : Filename of the MLOG Data
    filename_phase          : Filename of the PHAS Data
    csv                     : True, if false then dealing with hdf5 files
    start_band              : 0 GHz, starting frequency of data
    stop_band               : 18 GHz, stopping frequency of data
    
    --- Outputs ---
    Returns a list of 3 np.arrays that contain the Frequency, Magnitude, and Phase data
    respectively of the files that were provided. 
    '''
    if (csv == False): 
        file = h5py.File(filename_mag, 'r')
        freq_start = file['Parameters']['VNA_PARAMETERS']['freq_start'][0]
        freq_stop = file['Parameters']['VNA_PARAMETERS']['freq_stop'][0]
        probe_pts = file['Parameters']['VNA_PARAMETERS']['probe_pts'][()]
    
        freqs = np.linspace(freq_start, freq_stop, probe_pts)
        mag = np.asarray(file['sweep_dataframe']).flatten() 
        
        filep = h5py.File(filename_phase, 'r')
        phase = np.asarray(filep['sweep_dataframe']).flatten()        
        
        mask = (freqs >= start_band) & (freqs <= stop_band)
        freqs_window = freqs[mask]
        mag_window = mag[mask]  
        phase_window = phase[mask] 
        return [freqs_window, mag_window, phase_window]
    else: 
        df = pl.read_csv(filename_mag)    
        dfp = pl.read_csv(filename_phase)
        mask = (df[df.columns[0]].to_numpy() >= start_band) & (df[df.columns[0]].to_numpy() <= stop_band)
        return [((df[df.columns[0]]).to_numpy())[mask], ((df[df.columns[1]]).to_numpy())[mask], ((dfp[dfp.columns[1]]).to_numpy())[mask]]
    
def perform_ifft_tdr(freq, mag, phase, ifft_return=False, plot=False):
    '''
    This function performs and inverse fourier transform on the data provided. 
    It can return the arrays containing the time and time domain data of the 
    ifft as well as potentially plot it as well. 
    
    --- Inputs --- 
    freq                : Frequency data, np.array
    mag                 : MLOG/Magnitude data, np.array
    phase               : PHAS/Phase data, np.array
    ifft_return         : False, if True returns time domain data
    plot                : False, if True plots the IFFT 
    
    --- Outputs --- 
    Returns a list of 2 np.arrays time and full_magdata_time, which are the 
    time (x-axis) values and the time domain magnitude (y-axis) data.     
    '''
    phase = np.pi/180 * phase
    full_magdata = 10**(mag/20) *np.exp(1j*phase)

    full_magdata_time = ifft(full_magdata)
    
    N = len(freq)
    df = freq[1] - freq[0]
    dt = 1 / (N*df)
    time = np.arange(N) * dt
    
    if plot == True:            
        plt.plot(time, full_magdata_time, label='Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Inverse Fourier Transform')
        plt.legend()
        plt.show()

    if ifft_return == True:
        return [time, full_magdata_time]
    
def get_impedence_plot(freq, mag, phase, xlim_low=0, xlim_high=5, limit=False):
    ## The first step is to convert the S11 Magnitude and Phase data into a complex version that is basically 
    ## our reflection coefficient which is needed to do all of this. The reflection coefficient is however
    ## not in dB, so it's on a linear scale so converting to that shouldn't be too hard at all. 
    
    phase = np.pi/180 * phase ## Converting Phase from Degrees to Radians 
    gamma_freq = 10**(mag/20) * np.exp(1j*phase) ## Reflection Coefficient in Frequency Domain
    
    gamma_time = ifft(gamma_freq)   ## Perform the IFFT to get to time domain
    N = len(freq)
    df = freq[1] - freq[0]
    dt = 1 / (N*df)
    time = np.arange(N) * dt
    
    impedence = 50 * ((1 + gamma_time)/(1 - gamma_time))  ## Calculate the Impedence with the formula 
    distance = 0.8*3e8*time/2  ## Calculate the distance travelled based on the velocity of light inside the cable + time data
    filtered = savgol_filter(impedence, window_length=15, polyorder=2)

    plt.plot(distance, impedence, label='Raw Data')
    # plt.plot(distance, filtered, label='Smoothed Data', color='firebrick')
    plt.xlabel('Distance (m)')
    plt.ylabel('Impedence (Ohms)')
    plt.title('Impedence vs Distance')
    plt.legend()
    if limit==True:
        plt.xlim(xlim_low, xlim_high)
    plt.show()    
    
def get_impedence_tdr(freqs, mag, phase):
    '''
    This function takes our frequency, MLOG and PHAS data of a certain file, 
    and converts it into distance and impedence values for the scan. 
    
    --- Inputs --- 
    freqs           : Frequency data, np.array
    mag             : MLOG/Magnitude data, np.array
    phase           : PHAS/Phase data, np.array 
    
    --- Outputs --- 
    Returns a list of the distance and impedence np.arrays respectively
    '''
    phase = np.pi/180 * phase 
    gamma_freq = 10**(mag/20) * np.exp(1j*phase)
    
    gamma_time = ifft(gamma_freq)
    N = len(freqs)
    df = freqs[1] - freqs[0]
    dt = 1 / (N*df)
    
    time = np.arange(N) * dt
    
    impedence = 50 * ((1 + gamma_time)/(1 - gamma_time))
    distance = 0.8*3e8*time/2
    
    return [distance, impedence]
    
def impedence_statistics(distance, impedence, threshold, plot=False):
    '''
    This function takes the distance and impedence arrays calculated to create a plot
    of the landscape. It also takes a threshold above which some impedence mismatches
    should be highlighted and plotted. 
    
    --- Inputs --- 
    distances               : The distance values in the impedence landscape
    impedence               : The impedence values in the impedence landscape
    threshold               : The impedence threshold (above/below 50 Ohms) at which a mismatch is to be identified
    plot                    : False, if True plots the Distance vs Impedence graph
    
    --- Outputs --- 
    Returns a list of the disatncess and impedencess of the mismatches
    that have been identified based on the threshold respectively. 
    
    '''
    peaks, _ = find_peaks(impedence, height = 50 + threshold)
    troughs, _ = find_peaks(-impedence, height = -(50 - threshold))
    
    distancess = [distance[peaks], distance[troughs]]
    impedencess = [np.abs(impedence[peaks]), np.abs(impedence[troughs])]
    
    if plot == True: 
        plt.plot(distance, impedence, label='Impedence Landscape')
        plt.plot(distance[peaks], impedence[peaks], 'o', markersize=2.5, color='firebrick', label='Potential Mismatch Locations')
        plt.plot(distance[troughs], impedence[troughs], 'o', markersize=2.5, color='firebrick')
        plt.xlabel('Distance (m)')
        plt.ylabel('Impedence (Ohms)')
        plt.title('Impedence vs Distance')
        plt.legend()
        plt.show()   
        
    return [distancess, impedencess]

def twpa_impedences(folder_name, mismatch_threshold): 
    '''
    This function takes the folder which contains all of the TWPA measurement files as
    well as a mismatch threshold value and uses it to find the impedence plots for all 
    of the files within the folder. It counts the number of these 'mismatches', their
    respective impedences and the distance through which they ocurr within the line to 
    spit out histograms that portray this data. 
    
    --- Inputs --- 
    folder_name         : Folder in which the data files are kept
    mismatch_threshold  : The impedence threshold (above/below 50 Ohms) at which a mismatch is to be identified
    '''
    distances_total = []
    impedences_total = []
    
    folder_path = Path(folder_name)
    files = sorted(folder_path.glob("*.hdf5"))
    for i, f in enumerate(files):
        file_on = h5py.File(f, 'r')
        pump_on  = (file_on['Parameters']['EQUIPMENT_PARAMETERS']['pump_on'][()])
        measurement = ((file_on['Parameters']['VNA_PARAMETERS']['measurement_type'][()]).decode('utf-8'))        
        if ((pump_on == True) and (measurement == 'MLOG')):
            if (i + 2) <= len(files):
                j = files[i+1]
                freq, mag, phase = get_data_tdr(f, j, csv=False, start_band=4e9, stop_band=8e9)
                distance, impedence = get_impedence_tdr(freq, mag, phase)
                distances, impedences = impedence_statistics(distance, impedence, mismatch_threshold)
                distances_total = distances_total + distances 
                impedences_total = impedences_total + impedences
            
    mismatch_distances = []
    mismatch_impedences = []
    mismatch_impedences_absolute = []
    for i in distances_total:
        for j in i:
            mismatch_distances.append(j)
    for i in impedences_total:
        for j in i:
            mismatch_impedences.append(j)
            mismatch_impedences_absolute.append(np.abs(50-j))
            
    plt.hist(mismatch_distances, bins=30, density=True)
    plt.title('Distance to Potential Impedence Mismatch')
    plt.xlabel('Distance (m)')
    plt.ylabel('Normalized Counts')
    plt.show()

    plt.hist(mismatch_impedences, bins=25, density=True)
    plt.title('Impedences of Mismatch Points')
    plt.xlabel('Impedence (Ohms)')
    plt.ylabel('Normalized Counts')
    plt.show()

    plt.hist(mismatch_impedences_absolute, bins=20)
    plt.title('Absolute Impedences of Mismatch Points')
    plt.xlabel('Impedence (Ohms)')
    plt.ylabel('Counts')
    plt.show()
    
