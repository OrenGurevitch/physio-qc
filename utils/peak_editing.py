"""
Manual peak editing functions
Pure functions with no classes
"""

import numpy as np

import config


def find_local_maximum(signal, center_idx, window_samples):
    """
    Find local maximum in signal within window around center index

    Parameters
    ----------
    signal : array
        Signal to search
    center_idx : int
        Center index to search around
    window_samples : int
        Window size in samples (half-window on each side)

    Returns
    -------
    int
        Index of local maximum
    """
    start = max(0, center_idx - window_samples)
    end = min(len(signal), center_idx + window_samples + 1)

    segment = signal[start:end]
    if len(segment) == 0:
        return center_idx

    local_max_idx = np.argmax(segment)
    return start + local_max_idx


def find_local_minimum(signal, center_idx, window_samples):
    """
    Find local minimum in signal within window around center index

    Parameters
    ----------
    signal : array
        Signal to search
    center_idx : int
        Center index to search around
    window_samples : int
        Window size in samples (half-window on each side)

    Returns
    -------
    int
        Index of local minimum
    """
    start = max(0, center_idx - window_samples)
    end = min(len(signal), center_idx + window_samples + 1)

    segment = signal[start:end]
    if len(segment) == 0:
        return center_idx

    local_min_idx = np.argmin(segment)
    return start + local_min_idx


def add_peak(signal, current_peaks, click_time, sampling_rate, window_seconds=None):
    """
    Add peak at clicked location by finding local maximum

    Parameters
    ----------
    signal : array
        Cleaned signal
    current_peaks : array
        Current peak indices
    click_time : float
        Time of click in seconds
    sampling_rate : int
        Sampling rate in Hz
    window_seconds : float or None
        Window size in seconds (uses config default if None)

    Returns
    -------
    array
        Updated peaks array
    """
    if window_seconds is None:
        window_seconds = config.PEAK_ADD_WINDOW_SECONDS

    click_idx = int(click_time * sampling_rate)
    window_samples = int(window_seconds * sampling_rate / 2)

    new_peak_idx = find_local_maximum(signal, click_idx, window_samples)

    current_peaks = np.array(current_peaks)
    if new_peak_idx not in current_peaks:
        current_peaks = np.sort(np.append(current_peaks, new_peak_idx))

    return current_peaks


def add_trough(signal, current_troughs, click_time, sampling_rate, window_seconds=None):
    """
    Add trough at clicked location by finding local minimum

    Parameters
    ----------
    signal : array
        Cleaned signal
    current_troughs : array
        Current trough indices
    click_time : float
        Time of click in seconds
    sampling_rate : int
        Sampling rate in Hz
    window_seconds : float or None
        Window size in seconds (uses config default if None)

    Returns
    -------
    array
        Updated troughs array
    """
    if window_seconds is None:
        window_seconds = config.PEAK_ADD_WINDOW_SECONDS

    click_idx = int(click_time * sampling_rate)
    window_samples = int(window_seconds * sampling_rate / 2)

    new_trough_idx = find_local_minimum(signal, click_idx, window_samples)

    current_troughs = np.array(current_troughs)
    if new_trough_idx not in current_troughs:
        current_troughs = np.sort(np.append(current_troughs, new_trough_idx))

    return current_troughs


def delete_peak(current_peaks, click_time, sampling_rate, tolerance_seconds=None):
    """
    Delete nearest peak to clicked location

    Parameters
    ----------
    current_peaks : array
        Current peak indices
    click_time : float
        Time of click in seconds
    sampling_rate : int
        Sampling rate in Hz
    tolerance_seconds : float or None
        Maximum distance to find peak (uses config default if None)

    Returns
    -------
    array
        Updated peaks array
    """
    if tolerance_seconds is None:
        tolerance_seconds = config.PEAK_DELETE_TOLERANCE_SECONDS

    current_peaks = np.array(current_peaks)
    if len(current_peaks) == 0:
        return current_peaks

    click_idx = int(click_time * sampling_rate)
    tolerance_samples = int(tolerance_seconds * sampling_rate)

    distances = np.abs(current_peaks - click_idx)
    nearest_idx = np.argmin(distances)

    if distances[nearest_idx] <= tolerance_samples:
        current_peaks = np.delete(current_peaks, nearest_idx)

    return current_peaks


def delete_trough(current_troughs, click_time, sampling_rate, tolerance_seconds=None):
    """
    Delete nearest trough to clicked location

    Parameters
    ----------
    current_troughs : array
        Current trough indices
    click_time : float
        Time of click in seconds
    sampling_rate : int
        Sampling rate in Hz
    tolerance_seconds : float or None
        Maximum distance to find trough (uses config default if None)

    Returns
    -------
    array
        Updated troughs array
    """
    if tolerance_seconds is None:
        tolerance_seconds = config.PEAK_DELETE_TOLERANCE_SECONDS

    current_troughs = np.array(current_troughs)
    if len(current_troughs) == 0:
        return current_troughs

    click_idx = int(click_time * sampling_rate)
    tolerance_samples = int(tolerance_seconds * sampling_rate)

    distances = np.abs(current_troughs - click_idx)
    nearest_idx = np.argmin(distances)

    if distances[nearest_idx] <= tolerance_samples:
        current_troughs = np.delete(current_troughs, nearest_idx)

    return current_troughs


def erase_peaks_in_range(current_peaks, time_start, time_end, sampling_rate):
    """
    Remove all peaks within specified time range

    Parameters
    ----------
    current_peaks : array
        Current peak indices
    time_start : float
        Start time in seconds
    time_end : float
        End time in seconds
    sampling_rate : int
        Sampling rate in Hz

    Returns
    -------
    array
        Updated peaks array
    """
    current_peaks = np.array(current_peaks)
    start_idx = int(time_start * sampling_rate)
    end_idx = int(time_end * sampling_rate)

    mask = (current_peaks >= start_idx) & (current_peaks <= end_idx)
    return current_peaks[~mask]


def erase_troughs_in_range(current_troughs, time_start, time_end, sampling_rate):
    """
    Remove all troughs within specified time range

    Parameters
    ----------
    current_troughs : array
        Current trough indices
    time_start : float
        Start time in seconds
    time_end : float
        End time in seconds
    sampling_rate : int
        Sampling rate in Hz

    Returns
    -------
    array
        Updated troughs array
    """
    current_troughs = np.array(current_troughs)
    start_idx = int(time_start * sampling_rate)
    end_idx = int(time_end * sampling_rate)

    mask = (current_troughs >= start_idx) & (current_troughs <= end_idx)
    return current_troughs[~mask]


def add_peaks_in_range(signal, current_peaks, time_start, time_end, sampling_rate, min_distance_seconds=0.3):
    """
    Auto-detect and add all peaks within specified time range

    Parameters
    ----------
    signal : array
        Signal to search for peaks
    current_peaks : array
        Current peak indices
    time_start : float
        Start time in seconds
    time_end : float
        End time in seconds
    sampling_rate : int
        Sampling rate in Hz
    min_distance_seconds : float
        Minimum distance between peaks in seconds

    Returns
    -------
    array
        Updated peaks array
    """
    from scipy.signal import find_peaks

    current_peaks = np.array(current_peaks)
    start_idx = int(time_start * sampling_rate)
    end_idx = int(time_end * sampling_rate)

    # Find peaks in the selected region
    segment = signal[start_idx:end_idx]
    if len(segment) < 2:
        return current_peaks

    min_distance = int(min_distance_seconds * sampling_rate)
    new_peaks_relative, _ = find_peaks(segment, distance=min_distance)
    new_peaks = start_idx + new_peaks_relative

    # Merge with existing peaks
    all_peaks = np.unique(np.concatenate([current_peaks, new_peaks]))
    return np.sort(all_peaks)


def add_troughs_in_range(signal, current_troughs, time_start, time_end, sampling_rate, min_distance_seconds=0.3):
    """
    Auto-detect and add all troughs within specified time range

    Parameters
    ----------
    signal : array
        Signal to search for troughs
    current_troughs : array
        Current trough indices
    time_start : float
        Start time in seconds
    time_end : float
        End time in seconds
    sampling_rate : int
        Sampling rate in Hz
    min_distance_seconds : float
        Minimum distance between troughs in seconds

    Returns
    -------
    array
        Updated troughs array
    """
    from scipy.signal import find_peaks

    current_troughs = np.array(current_troughs)
    start_idx = int(time_start * sampling_rate)
    end_idx = int(time_end * sampling_rate)

    # Find troughs (peaks in inverted signal) in the selected region
    segment = signal[start_idx:end_idx]
    if len(segment) < 2:
        return current_troughs

    min_distance = int(min_distance_seconds * sampling_rate)
    inverted_segment = -segment
    new_troughs_relative, _ = find_peaks(inverted_segment, distance=min_distance)
    new_troughs = start_idx + new_troughs_relative

    # Merge with existing troughs
    all_troughs = np.unique(np.concatenate([current_troughs, new_troughs]))
    return np.sort(all_troughs)


def calculate_peak_delta(auto_peaks, current_peaks, signal_length):
    """
    Calculate delta array encoding peak edits

    Encoding:
    - 1: Auto-detected peak (in auto_peaks)
    - 2: Manually added peak (in current_peaks but not auto_peaks)
    - 0: No peak
    - -1: Deleted peak (in auto_peaks but not current_peaks)

    Parameters
    ----------
    auto_peaks : array
        Original auto-detected peak indices
    current_peaks : array
        Current peak indices after editing
    signal_length : int
        Length of signal

    Returns
    -------
    array
        Delta array of shape (signal_length,) with int8 dtype
    """
    auto_peaks = np.array(auto_peaks, dtype=int)
    current_peaks = np.array(current_peaks, dtype=int)

    delta = np.zeros(signal_length, dtype=config.EXPORT_DTYPE_ONSETS)

    delta[auto_peaks] = config.PEAK_ENCODING['AUTO_DETECTED']

    deleted_peaks = np.setdiff1d(auto_peaks, current_peaks)
    delta[deleted_peaks] = config.PEAK_ENCODING['DELETED']

    added_peaks = np.setdiff1d(current_peaks, auto_peaks)
    delta[added_peaks] = config.PEAK_ENCODING['MANUALLY_ADDED']

    return delta


def get_edited_peaks_info(auto_peaks, current_peaks):
    """
    Get information about peak edits

    Parameters
    ----------
    auto_peaks : array
        Original auto-detected peak indices
    current_peaks : array
        Current peak indices after editing

    Returns
    -------
    dict
        Dictionary containing:
        - auto_count: Number of auto-detected peaks
        - added_count: Number of manually added peaks
        - deleted_count: Number of deleted peaks
        - final_count: Final number of peaks
        - added_indices: Indices of manually added peaks
        - deleted_indices: Indices of deleted peaks
    """
    auto_peaks = np.array(auto_peaks, dtype=int)
    current_peaks = np.array(current_peaks, dtype=int)

    deleted_peaks = np.setdiff1d(auto_peaks, current_peaks)
    added_peaks = np.setdiff1d(current_peaks, auto_peaks)

    return {
        'auto_count': len(auto_peaks),
        'added_count': len(added_peaks),
        'deleted_count': len(deleted_peaks),
        'final_count': len(current_peaks),
        'added_indices': added_peaks,
        'deleted_indices': deleted_peaks
    }
