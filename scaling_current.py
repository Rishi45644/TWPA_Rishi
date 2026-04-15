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

## This is the Scaling Current Code   

def m_to_istar(m):
    return np.sqrt(1/m)

def quadratic_fit(x, a):
    return a*x**2

def istar_expression(file_dep, file_zero, frequency_band, file_dispersion):
    '''
    This function essentially does the backbone of the expression calculation for 
    fitting I_star. It gets the data that is given to convert it to the correct
    'x' and 'y' values that will be used in the quadratic fitting for I_star later
    on. For further explaination of the fitting check the documentation. 
    
    --- Inputs --- 
    file_dep               : This is the file from which the bias current 
                             dependent phase values are extracted
                             
    file_zero              : This is the file from which the zero-bias current 
                             phase values are extracted
                             
    frequency_band         : This is the frequency at which you want to calculate I_star
    
    file_dispersion        : This is the dispersion plots from Milad that are used 
                             to calculate the time of flight of the light inside the TWPA
    
    --- Outputs --- 
    A list of the x and y values are returned. The x value is the 
    bias current value (mA) and the y value is the other side of the 
    equation used to calculate I_star, basically the total phase change. 
    '''
    ## Important Values
    N_sc = 320.0
    L_sc = (35.0*15.0)/10**6

    ## Speed of Light and Traversal Time Calculations 
    data = load(file_dispersion)
    lst = data.files

    frequency_twpa = data[lst[0]]
    k_vector = data[lst[1]]/L_sc  
    omega = 2*np.pi*frequency_twpa*1e9 
    
    phase_velocity = (omega/k_vector)
    tau_twpa = (N_sc*L_sc)/phase_velocity 
    
    ## Original bias-current dependent file
    file = h5py.File(file_dep, 'r')
    freq_start = file['Parameters']['VNA_PARAMETERS']['freq_start'][0]
    freq_stop = file['Parameters']['VNA_PARAMETERS']['freq_stop'][0]
    probe_pts = file['Parameters']['VNA_PARAMETERS']['probe_pts'][()]

    freqs = np.linspace(freq_start, freq_stop, probe_pts)
    phase = np.asarray(file['sweep_dataframe']).flatten() 
        
    params_txt = (file['Parameters']['PHYSICAL_PARAMETERS']['Description'][()]).decode("utf-8")

    for i in params_txt.split("_"):
        if i.endswith('V'):
            bias_current = float(i[0:i.find('V')])
    
    ## Zero bias-current file 
    file_naught = h5py.File(file_zero, 'r')
    phase_naught = np.asarray(file_naught['sweep_dataframe']).flatten() 
    
    phase = phase * (np.pi/180)
    phase_naught = phase_naught * (np.pi/180)
    
    ## Relative phase
    indices_phase = np.where(freqs >= frequency_band)
    indices_twpa = np.where(frequency_twpa*1e9 >= frequency_band)
    index_phase = indices_phase[0][0]
    index_twpa = indices_twpa[0][0]
    
    theta_I = phase[index_phase]
    theta_0 = phase_naught[index_phase]
    theta_r = 2*np.pi*frequency_band*tau_twpa[index_twpa]
    
    ## Expression buildup 
    x = bias_current
    y = -2*((theta_I-theta_0)/(theta_r))
    
    return [x, y]

def calculating_istar(folder_name, frequency_band, file_zero, file_dispersion, plot=False):
    '''
    This function takes the entire folder in which the phase data is kept and employs
    the 'istar_expression' function to calculate i_stars for a certain frequency. The 
    fit can only be performed if we fix a frequency, check the documentation for more
    information on why this is. 
    
    --- Inputs ---
    folder_name         : The folder in which all of the current-dependent phase files are (UPH)
    frequency_band      : The frequency at which the i_star fit is to be performed
    file_zero           : The file in which the zero-bias current phase file is (UPH)
    file_dispersion     : The dispersion plot of the TWPA file obtained from Milad
    plot                : Whether or not the fit and the corresponding data should be plotted
    
    --- Outputs --- 
    A list of the corresponding I_star fit and the respective error on the fit.
    '''
    
    dc_currents = []
    delta_thetas = []
    
    folder_path = Path(folder_name)
    for f in folder_path.glob("*.hdf5"):
        x, y = istar_expression(f,
                                file_zero, 
                                frequency_band, 
                                file_dispersion)
        dc_currents.append(x)
        delta_thetas.append(y)
        
    dc_currents = np.array(dc_currents)
    delta_thetas = np.array(delta_thetas)
            
    param, cov = curve_fit(quadratic_fit, dc_currents, delta_thetas)
    perr = np.sqrt(np.diag(cov))
    m_fit = param
    m_fit_error = perr
    
    if plot == True:   
        fitted_y = quadratic_fit(dc_currents, m_fit)
        
        plt.plot(dc_currents, delta_thetas, 'o', label='Raw Data')
        plt.plot(dc_currents, fitted_y, label='Linear Fit')
        plt.xlabel(r'$I_{dc}$' + " (mA)")
        plt.ylabel(r'$-2\frac{\theta\left(I\right)-\theta_{0}}{\theta_{r}}$')
        plt.legend()
        plt.show()
        
    return [m_to_istar(m_fit), m_to_istar(m_fit_error)] 

def istar_plotting(start_frequency, stop_frequency, frequency_points, folder_name, file_zero, file_dispersion):
    '''
    This function takes the frequencies across which you want to see fits for i_star as well as 
    the resolution (spacing) of the frequencies and plots it all together using the other i_star
    fitting functions available at its disposal. 
    
    --- Inputs --- 
    start_frequency         : The starting frequency for seeing the istars
    stop_frequency          : The stopping frequency for seeing the istars
    frequency_points        : The number of frequencies at which you want to see the istars
    folder_name             : The folder in which all of the current-dependent phase files are (UPH)
    file_zero               : The file in which the zero-bias current phase file is (UPH)
    file_dispersion         : The dispersion plot of the TWPA file obtained from Milad
    
    '''
    istar_frequencies = np.linspace(start_frequency, stop_frequency, frequency_points)
    istars = []
    istars_err = []
    
    for i in istar_frequencies:
        istars.append(calculating_istar(folder_name, i, file_zero, file_dispersion)[0][0])
        istars_err.append(calculating_istar(folder_name, i, file_zero, file_dispersion)[1][0])
    
    istars = np.array(istars)
    istars_err = np.array(istars_err)
    
    full_band_avg = np.mean(istars)
    full_band_std = np.std(istars)

    mask = np.where((istar_frequencies >= 4e9) & (istar_frequencies <= 8e9))
    banded_avg = np.mean(istars[mask])
    banded_std = np.std(istars[mask])

    plt.plot(istar_frequencies/1e9, istars, '+', label='Data')
    plt.axhline(y=banded_avg, linestyle='--', label='4-8 GHz ' + r'$I_{*}=$' + ' ' + str(round(banded_avg, 2)) + ' mA')
    plt.axhspan(banded_avg - banded_std, banded_avg + banded_std, alpha=0.25)
    plt.xlabel('Frequncy (GHz)')
    plt.ylabel(r'$I_{*}$' + ' (mA)')
    plt.title('Frequency vs ' + r'$I_{*}$' + ' Measurements')
    plt.legend()
    plt.savefig('IstarVsFreq.png', bbox_inches='tight')
    
