import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

import mne

mne.set_log_level('WARNING')

from mne_connectivity import phase_slope_index
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt

from mne.time_frequency import tfr_array_morlet
from scipy.signal import hilbert

from scripts.eeg.preprocess import compute_csd


def create_epochs(raw, epoch_duration):
    """
    Create epochs from raw data with fixed-length events.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        epoch_duration (float): Duration of each epoch in seconds.
        
    Returns:
        mne.Epochs: Epochs created from the raw data.
    """
    # Check if the raw data is already preloaded
    if not raw.preload:
        raw.load_data()
    
    # Create fixed-length events
    events = mne.make_fixed_length_events(raw, id=1, duration=epoch_duration)
    
    # Create epochs
    epochs = mne.Epochs(raw, events, event_id=1, tmin=0, tmax=epoch_duration,
                        baseline=None, preload=True)
    
    return epochs

def estimate_m_range(phase1_band, phase2_band):
    """
    Estimate the range of m such that m * phase1_freq approximates phase2_freq.
    Inputs are tuples representing frequency bands, e.g., (0.5, 4) for delta.
    Returns integer m_min and m_max.
    """
    f1_min, f1_max = phase1_band
    f2_min, f2_max = phase2_band

    # Avoid division by zero
    if f1_max == 0 or f1_min == 0:
        raise ValueError("Frequency band includes 0 Hz, which is invalid.")

    m_min = round(np.floor(f2_min / f1_max))  # e.g. phase1 maxが4Hz, phase2 minが4Hzなら m_min = 1
    m_max = round(np.ceil(f2_max / f1_min))   # e.g. phase1 minが0.5Hz, phase2 maxが8Hzなら m_max = 16

    # Ensure m_min is at least 1
    m_min = max(1, m_min)

    return m_min, m_max

def find_best_m_for_n1(phase1, phase2, m_range=(1, 10), plot=False):
    """
    Find the best m value for fixed n=1 that maximizes phase-locking value (PLV)
    
    Parameters:
        phase1 (ndarray): Phase time series of the low-frequency signal
        phase2 (ndarray): Phase time series of the high-frequency signal
        m_range (tuple): Range of m values to test (min, max), inclusive
    
    Returns:
        best_m (int): Value of m with highest PLV for n=1
        psi_values (ndarray): PSI values for each m in the given range
    """
    n = 1
    #m_vals = np.arange(m_range[0], m_range[1] + 0.1, 0.1)
    m_vals = np.arange(m_range[0], m_range[1] + 1, 1)
    psi_values = []

    for m in tqdm(m_vals, desc="Searching best m", leave=False):
        # Compute phase difference for n=1:m
        phase_diff = n * phase2 - m * phase1
        # Compute PSI for the current m
        psi = np.abs(np.mean(np.exp(1j * phase_diff)))
        psi_values.append(psi)

    psi_values = np.array(psi_values)
    best_psi = np.argmax(psi_values)
    best_m = m_vals[best_psi]

    # Plot the results
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(m_vals, psi_values, marker='o')
        plt.title('Phase Synchronization Index (PSI) for n=1 and varying m')
        plt.xlabel('m (harmonic of wave)')
        plt.ylabel('Phase Synchronization Index (PSI)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return best_m, best_psi

def get_phase(args, raw1, raw2, f1, f2, method='wavelet'):
    """    Extract phase information from two raw EEG datasets using either Hilbert transform or wavelet transform. 
    Parameters:
        raw1 (mne.io.Raw): First raw EEG dataset.
        raw2 (mne.io.Raw): Second raw EEG dataset.
        f1 (tuple): Frequency band for the first dataset (e.g., theta).
        f2 (tuple): Frequency band for the second dataset (e.g., alpha).
        method (str): Method to use for phase extraction ('hilbert' or 'wavelet').
    Returns:
        phase1 (ndarray): Phase of the first dataset, shape: (n_channels, n_times).
        phase2 (ndarray): Phase of the second dataset, shape: (n_channels, n_times).
    """
    if method == 'hilbert':
        phase1 = np.angle(hilbert(raw1.get_data(), axis=1))
        phase2 = np.angle(hilbert(raw2.get_data(), axis=1))
    elif method == 'wavelet':
    
        epochs1 = create_epochs(raw1, args.epoch_duration)
        epochs2 = create_epochs(raw2, args.epoch_duration)

        freqs1 = np.linspace(f1[0], f1[1], 5)
        freqs2 = np.linspace(f2[0], f2[1], 5)

        # Phase1 extraction
        phase1 = tfr_array_morlet(epochs1, sfreq=raw1.info['sfreq'],
                                  freqs=freqs1, n_cycles=freqs1 / 2, output='phase')[0]
        phase1 = np.mean(phase1, axis=1)  # shape: (n_channels, n_times)

        # Phase2 extraction
        phase2 = tfr_array_morlet(epochs2, sfreq=raw2.info['sfreq'],
                                  freqs=freqs2, n_cycles=freqs2 / 2, output='phase')[0]
        phase2 = np.mean(phase2, axis=1)
    else:
        raise ValueError("method must be 'hilbert' or 'wavelet'")
    
    return phase1, phase2

def plot_save_psi_matrix(psi_matrix, id, condition, best_m, band1, band2, output_path):
    """
    Plot the n:m Phase Synchronization Index (PSI) matrix and save it as an image.
    Parameters:
        psi_matrix (ndarray): n:m Phase Synchronization Index matrix, shape: (n_channels, n_channels).
        id (str): Subject ID.
        condition (str): Condition label (e.g., 'EO', 'EC').
        best_m (int): Best m value found for the n:m coupling.
        f1 (str): Name of the first frequency band (e.g., 'delta').
        f2 (str): Name of the second frequency band (e.g., 'delta').
        output_path (str): Path to save the output image.
    """

    plt.figure(figsize=(6, 5))
    plt.imshow(psi_matrix, cmap='plasma', interpolation='nearest')
    plt.colorbar(label='n:m Phase Synchronization Index (PSI)')
    plt.title(f'1:{best_m} Phase Synchronization Index ({band1} Hz & {band2} Hz)')
    plt.xlabel('Channel j (Phase2)')
    plt.ylabel('Channel i (Phase1)')
    plt.tight_layout()
    
    subject_dir = os.path.join(output_path, id)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(subject_dir), exist_ok=True)
    
    # Save the figure
    img_filename = os.path.join(subject_dir, f"{id}_{condition}.png")

    plt.savefig(img_filename, dpi=300)
    plt.close()
    print(f"Saved PSI matrix plot to {img_filename}")

def save_psi_matrix(psi_matrix, id, condition, output_path):
    """
    Save the n:m Phase Synchronization Index (PSI) matrix to a file.
    
    Parameters:
        psi_matrix (ndarray): n:m Phase Synchronization Index matrix, shape: (n_channels, n_channels).
        id (str): Subject ID.
        condition (str): Condition label (e.g., 'EO', 'EC').
        output_path (str): Path to save the output file.
    """
    
    subject_dir = os.path.join(output_path, id)

    # Ensure the output directory exists

    # Create the directory if it doesn't exist
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)
    
    # Save the matrix as a numpy file
    np.save(os.path.join(subject_dir, f"{id}_{condition}_psi_matrix.npy"), psi_matrix)


def compute_nm_phase_phase_matrix(args, raw, id, condition, band1, band2, method='wavelef', simulate=False):
    """
    Compute n:m Phase Synchronization Index between all pairs of EEG channels.

    Parameters:
    -----------
    args : argparse.Namespace
        Contains paths and parameters.
    raw : mne.io.Raw
        Raw EEG data (preprocessed).
    id : str
        Subject ID (e.g., 'sub-010002').
    condition : str
        Condition label (e.g., 'EO', 'EC').
    band1 : str
        Name of the first frequency band (e.g., 'delta').
    band2 : str
        Name of the second frequency band (e.g., 'delta').
    method : str
        'hilbert' or 'wavelet'
    simulate : bool
        If True, simulate n:m phase coupling for a range of m values.
    Returns:
    --------
    phase_phase_matrix : np.ndarray, shape (n_channels, n_channels)
        Matrix of n:m Phase Synchronization Index (PSI).
    """

    bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'low_gamma': (30, 50),
    'mid_gamma': (50, 80),
    'high_gamma': (80, 100)
    }

    # Bandpass filtering
    f1 = bands[band1]
    f2 = bands[band2]
    raw1 = raw.copy().filter(f1[0], f1[1], method='iir', verbose=False)
    raw2 = raw.copy().filter(f2[0], f2[1], method='iir', verbose=False)

    # Compute CSD for both bands
    raw1 = compute_csd(raw1)
    raw2 = compute_csd(raw2)

    # Get phase information
    phase1, phase2 = get_phase(args, raw1, raw2, f1, f2, method=method)

    # Initialize the phase-phase matrix
    n_channels = phase1.shape[0]
    psi_matrix = np.zeros((n_channels, n_channels))

    if simulate:
        m_min, m_max = estimate_m_range(f1, f2)
    else:
        n = 1
        m = 1

    # Compute n:m phase-phase coupling for each channel pair
    for i in tqdm(range(n_channels), desc=f"Computing PSI ({id}, {condition})", leave=True):
        for j in range(n_channels):
            if simulate:
                # Simulate n:m phase coupling
                best_m, best_psi = find_best_m_for_n1(phase1[i], phase2[j], m_range=(m_min, m_max))
                psi_matrix[i, j] = best_psi

            else:
                phase_diff = n * phase2 - m * phase1
                # Compute PSI for the current m
                psi = np.abs(np.mean(np.exp(1j * phase_diff)))
                psi_matrix[i, j] = psi

    psi_matrix = np.array(psi_matrix)
    psi_matrix = np.mean(psi_matrix, axis=0)

    # Save the matrix to a file
    save_psi_matrix(psi_matrix, id, condition, args.psi_path)

    plot_save_psi_matrix(psi_matrix, id, condition, best_m, band1, band2, args.psi_path)

    
    return psi_matrix

def load_psi_matrix(args, id, condition):
    """ Load the saved n:m Phase Synchronization Index (PSI) matrix from a file.
    Returns:
        psi_matrix (ndarray): Loaded n:m Phase Synchronization Index matrix.
    """
    path = os.path.join(args.psi_path, id, f"{id}_{condition}_psi_matrix.npy")
    loaded_matrix = np.load(path, allow_pickle=True)

    return loaded_matrix

def run_nm_psi(args):

    id = 'sub-010002'
    condition = 'EO'  # or 'EC'
    band1 = 'delta'
    band2 = 'delta'

    # Define the paths to the raw EEG files
    path = os.path.join(args.preprocess_path, id, f"{id}_{condition}.fif")

    # Load the raw data
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw EEG file for condition {condition} does not exist: {path}")
    # Load the raw data   
    raw = mne.io.read_raw_fif(path, preload=True)
    if not raw.preload:
        raw.load_data()

    psi_matrix = compute_nm_phase_phase_matrix(args, raw, id, condition, band1, band2, method='hilbert')