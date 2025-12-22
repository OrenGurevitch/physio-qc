"""
Blood pressure signal processing functions
Pure functions with no classes
"""

import numpy as np
from scipy.signal import bessel, butter, filtfilt, find_peaks
import neurokit2 as nk

import config
from algorithms.bp_delineator import delineate_bp
from algorithms.quality_detection import detect_calibration_artifacts, filter_indices_outside_regions


def filter_bp(signal, sampling_rate, method='bessel_25hz', filter_order=3,
              cutoff_freq=25, filter_type='butterworth', lowcut=0.5, highcut=15.0,
              apply_lowcut=True, apply_highcut=True):
    """
    Filter blood pressure signal

    Parameters
    ----------
    signal : array
        Raw blood pressure signal
    sampling_rate : int
        Sampling rate in Hz
    method : str
        Filtering method ('bessel_25hz', 'butterworth', 'custom')
    filter_order : int
        Filter order
    cutoff_freq : float
        Cutoff frequency for lowpass methods (Hz)
    filter_type : str
        Filter type for custom filtering
    lowcut : float
        High-pass cutoff for custom filtering
    highcut : float
        Low-pass cutoff for custom filtering
    apply_lowcut : bool
        Apply high-pass filter (custom method only)
    apply_highcut : bool
        Apply low-pass filter (custom method only)

    Returns
    -------
    array
        Filtered blood pressure signal
    """
    if method == 'bessel_25hz':
        from scipy.signal import bessel
        wn = float(cutoff_freq) / (float(sampling_rate) / 2.0)
        wn = min(max(wn, 1e-6), 0.999999)
        b, a = bessel(filter_order, wn, btype="low", analog=False, output="ba", norm="phase")
        bp_filtered = filtfilt(b, a, signal)

    elif method == 'butterworth':
        wn = float(cutoff_freq) / (float(sampling_rate) / 2.0)
        wn = min(max(wn, 1e-6), 0.999999)
        b, a = butter(filter_order, wn, btype="low", analog=False)
        bp_filtered = filtfilt(b, a, signal)

    elif method == 'custom':
        bp_filtered = signal.copy()
        low = lowcut if apply_lowcut else None
        high = highcut if apply_highcut else None
        if (low is not None and low > 0) or (high is not None and high < sampling_rate / 2):
            bp_filtered = nk.signal_filter(
                bp_filtered,
                sampling_rate=sampling_rate,
                lowcut=low,
                highcut=high,
                method=filter_type,
                order=filter_order
            )
    else:
        bp_filtered = signal.copy()

    return bp_filtered


def detect_bp_peaks(signal, sampling_rate, method='delineator', prominence=10):
    """
    Detect systolic peaks and diastolic troughs in blood pressure signal

    Parameters
    ----------
    signal : array
        Filtered blood pressure signal
    sampling_rate : int
        Sampling rate in Hz
    method : str
        Detection method ('delineator', 'prominence')
    prominence : float
        Prominence parameter for prominence method

    Returns
    -------
    dict
        Dictionary containing:
        - peaks: Systolic peak indices
        - troughs: Diastolic trough indices
        - dicrotic_notches: Dicrotic notch indices (delineator only, 0 if not found)
    """
    if method == 'delineator':
        result = delineate_bp(signal, sampling_rate)
        return {
            'peaks': result['peaks'],
            'troughs': result['onsets'],
            'dicrotic_notches': result['dicrotic_notches']
        }

    elif method == 'prominence':
        peaks, _ = find_peaks(signal, prominence=prominence)

        inverted = -signal
        troughs, _ = find_peaks(inverted, prominence=prominence)

        return {
            'peaks': peaks,
            'troughs': troughs,
            'dicrotic_notches': np.array([], dtype=int)
        }

    return {
        'peaks': np.array([], dtype=int),
        'troughs': np.array([], dtype=int),
        'dicrotic_notches': np.array([], dtype=int)
    }


def calculate_bp_metrics(signal, peaks, troughs):
    """
    Calculate blood pressure metrics from peaks and troughs

    Parameters
    ----------
    signal : array
        Filtered blood pressure signal
    peaks : array
        Systolic peak indices
    troughs : array
        Diastolic trough indices

    Returns
    -------
    dict
        Dictionary containing:
        - sbp: Systolic blood pressure at each peak
        - dbp: Diastolic blood pressure at each trough
        - mbp: Mean arterial pressure (calculated as DBP + (SBP-DBP)/3)
        - sbp_signal: Systolic pressure as continuous signal (forward-filled)
        - dbp_signal: Diastolic pressure as continuous signal (forward-filled)
        - mbp_signal: Mean arterial pressure as continuous signal
        - mean_sbp: Mean systolic pressure
        - mean_dbp: Mean diastolic pressure
        - mean_mbp: Mean arterial pressure
    """
    sbp = signal[peaks] if len(peaks) > 0 else np.array([])
    dbp = signal[troughs] if len(troughs) > 0 else np.array([])

    sbp_signal = np.full_like(signal, np.nan, dtype=float)
    dbp_signal = np.full_like(signal, np.nan, dtype=float)

    if len(peaks) > 0:
        sbp_signal[peaks] = sbp
        sbp_signal = np.array([sbp_signal[i] if not np.isnan(sbp_signal[i]) else sbp_signal[i-1] if i > 0 else np.nan for i in range(len(sbp_signal))])

    if len(troughs) > 0:
        dbp_signal[troughs] = dbp
        dbp_signal = np.array([dbp_signal[i] if not np.isnan(dbp_signal[i]) else dbp_signal[i-1] if i > 0 else np.nan for i in range(len(dbp_signal))])

    mbp_signal = np.where(
        ~np.isnan(dbp_signal) & ~np.isnan(sbp_signal),
        dbp_signal + (sbp_signal - dbp_signal) / 3,
        np.nan
    )

    return {
        'sbp': sbp,
        'dbp': dbp,
        'mbp': dbp + (sbp - dbp) / 3 if len(sbp) > 0 and len(dbp) > 0 else np.array([]),
        'sbp_signal': sbp_signal,
        'dbp_signal': dbp_signal,
        'mbp_signal': mbp_signal,
        'mean_sbp': np.nanmean(sbp) if len(sbp) > 0 else 0.0,
        'mean_dbp': np.nanmean(dbp) if len(dbp) > 0 else 0.0,
        'mean_mbp': np.nanmean(dbp + (sbp - dbp) / 3) if len(sbp) > 0 and len(dbp) > 0 else 0.0
    }


def process_bp(signal, sampling_rate, params):
    """
    Complete blood pressure processing pipeline

    Parameters
    ----------
    signal : array
        Raw blood pressure signal
    sampling_rate : int
        Sampling rate in Hz
    params : dict
        Processing parameters

    Returns
    -------
    dict or None
        Dictionary containing all processing results, or None if insufficient peaks
        Includes:
        - raw: Raw signal
        - filtered: Filtered signal
        - auto_peaks: Auto-detected systolic peak indices (immutable)
        - current_peaks: Current systolic peak indices (mutable for editing)
        - auto_troughs: Auto-detected diastolic trough indices (immutable)
        - current_troughs: Current diastolic trough indices (mutable for editing)
        - dicrotic_notches: Dicrotic notch indices (if detected)
        - peaks_times: Peak times in seconds
        - troughs_times: Trough times in seconds
        - sbp, dbp, mbp: Blood pressure values
        - sbp_signal, dbp_signal, mbp_signal: Continuous BP signals
        - n_peaks, n_troughs: Counts
        - mean_sbp, mean_dbp, mean_mbp: Statistics
        - calibration_artifacts: Calibration artifact regions (if detected)
        - quality_dp: Derivative quality metrics (if detected)
    """
    bp_filtered = filter_bp(
        signal,
        sampling_rate,
        method=params.get('filter_method', 'bessel_25hz'),
        filter_order=params.get('filter_order', 3),
        cutoff_freq=params.get('cutoff_freq', 25),
        filter_type=params.get('filter_type', 'butterworth'),
        lowcut=params.get('lowcut', 0.5),
        highcut=params.get('highcut', 15.0),
        apply_lowcut=params.get('apply_lowcut', True),
        apply_highcut=params.get('apply_highcut', True)
    )

    peak_result = detect_bp_peaks(
        bp_filtered,
        sampling_rate,
        method=params.get('peak_method', 'delineator'),
        prominence=params.get('prominence', 10)
    )

    peaks = peak_result['peaks']
    troughs = peak_result['troughs']
    dicrotic_notches = peak_result['dicrotic_notches']

    result = {
        'raw': signal,
        'filtered': bp_filtered,
        'dicrotic_notches': dicrotic_notches
    }

    if params.get('detect_calibration', True):
        calib_result = detect_calibration_artifacts(
            signal,
            sampling_rate,
            thr_norm=params.get('calibration_threshold', 0.03),
            min_dur_s=params.get('calibration_min_duration', 2.0),
            pad_s=params.get('calibration_padding', 0.1)
        )
        result['calibration_artifacts'] = calib_result
        result['quality_dp'] = calib_result['dp_plot']

        peaks = filter_indices_outside_regions(
            peaks,
            calib_result['starts'],
            calib_result['ends']
        )
        troughs = filter_indices_outside_regions(
            troughs,
            calib_result['starts'],
            calib_result['ends']
        )
    else:
        result['calibration_artifacts'] = None
        result['quality_dp'] = np.array([])

    if len(peaks) < 2 or len(troughs) < 2:
        return None

    result['auto_peaks'] = peaks.copy()
    result['current_peaks'] = peaks.copy()
    result['auto_troughs'] = troughs.copy()
    result['current_troughs'] = troughs.copy()
    result['peaks_times'] = peaks / sampling_rate
    result['troughs_times'] = troughs / sampling_rate
    result['n_peaks'] = len(peaks)
    result['n_troughs'] = len(troughs)

    bp_metrics = calculate_bp_metrics(bp_filtered, peaks, troughs)
    result.update(bp_metrics)

    return result
