"""
Configuration file for physiological signal QC
Edit these values to match your setup and data requirements
"""

# =============================================================================
# PATHS
# =============================================================================

# Path to raw physiological data files (.acq or .csv format)
# Expected structure: BASE_DATA_PATH/sub-{id}/ses-{id}/sub-{id}_ses-{id}_task-{task}_physio.acq
BASE_DATA_PATH = '/export02/projects/LCS/01_physio'

# Path where processed data will be saved (CSV + JSON)
# Output structure: OUTPUT_BASE_PATH/sub-{id}/ses-{id}/sub-{id}_ses-{id}_task-{task}_physio.{csv,json}
OUTPUT_BASE_PATH = '/export02/projects/LCS/02_physio_processed'

# =============================================================================
# DATA PARAMETERS
# =============================================================================

SAMPLING_RATE = 250

POWERLINE_FREQUENCIES = [60, 50]  # North America uses 60Hz

SIGNAL_PATTERNS = {
    'ecg': ['ecg', 'ekg', 'cardiac', 'heart'],
    'rsp': ['rsp', 'resp', 'respiratory', 'breathing', 'breath'],
    'ppg': ['ppg', 'pleth', 'pulse', 'photoplethysmography'],
    'bp': ['bp', 'blood_pressure', 'arterial_pressure', 'abp', 'art', 'ami', 'hlt', 'a10']
}

# =============================================================================
# ECG CONFIGURATION
# =============================================================================

ECG_CLEANING_METHODS = [
    'neurokit',
    'biosppy',
    'pantompkins1985',
    'hamilton2002',
    'elgendi2010',
    'engzeemod2012',
    'vg',
    'templateconvolution',
    'custom'
]

ECG_CLEANING_INFO = {
    'neurokit': '0.5 Hz high-pass butterworth filter (order = 5), followed by powerline filtering',
    'biosppy': 'FIR filter [0.67, 45] Hz (order = 1.5 × sampling rate)',
    'pantompkins1985': 'Pan & Tompkins (1985) method',
    'hamilton2002': 'Hamilton (2002) method',
    'elgendi2010': 'Elgendi et al. (2010) method',
    'engzeemod2012': 'Engelse & Zeelenberg (1979) modified method',
    'vg': 'Visibility Graph method - 4.0 Hz high-pass butterworth (order = 2)',
    'templateconvolution': 'Template convolution method',
    'custom': 'Apply user-specified digital filters (Butterworth, FIR, Chebyshev, Elliptic, etc.)'
}

ECG_PEAK_METHODS = [
    'neurokit',
    'pantompkins1985',
    'hamilton2002',
    'zong2003',
    'martinez2004',
    'christov2004',
    'gamboa2008',
    'elgendi2010',
    'engzeemod2012',
    'manikandan2012',
    'khamis2016',
    'kalidas2017',
    'nabian2018',
    'rodrigues2021',
    'emrich2023',
    'promac'
]

ECG_PEAK_INFO = {
    'neurokit': 'NeuroKit2 default - QRS detection based on gradient steepness',
    'pantompkins1985': 'Pan & Tompkins (1985) - Classic real-time QRS detection',
    'hamilton2002': 'Hamilton (2002) algorithm',
    'zong2003': 'Zong et al. (2003) method',
    'martinez2004': 'Martinez et al. (2004) algorithm',
    'christov2004': 'Christov (2004) method',
    'gamboa2008': 'Gamboa (2008) algorithm',
    'elgendi2010': 'Elgendi et al. (2010) method',
    'engzeemod2012': 'Engelse & Zeelenberg modified by Lourenço et al. (2012)',
    'manikandan2012': 'Manikandan & Soman (2012) - Shannon energy envelope',
    'khamis2016': 'UNSW Algorithm - designed for clinical and telehealth ECGs',
    'kalidas2017': 'Kalidas et al. (2017) algorithm',
    'nabian2018': 'Nabian et al. (2018) - Pan-Tompkins adaptation',
    'rodrigues2021': 'Rodrigues et al. (2021) adaptation',
    'emrich2023': 'FastNVG - visibility graph detector (sample-accurate)',
    'promac': 'Probabilistic combination of multiple detectors'
}

DEFAULT_ECG_PARAMS = {
    'powerline': 60,
    'method': 'neurokit',
    'lowcut': 0.5,
    'highcut': 45.0,
    'peak_method': 'neurokit',
    'correct_artifacts': False,
    'calculate_quality': False,
    'rate_method': 'monotone_cubic',
    'filter_type': 'butterworth',
    'filter_order': 5,
    'apply_lowcut': True,
    'apply_highcut': True
}

# =============================================================================
# RSP CONFIGURATION
# =============================================================================

RSP_CLEANING_METHODS = ['khodadad2018', 'biosppy', 'hampel', 'custom']

RSP_CLEANING_INFO = {
    'khodadad2018': 'Second order 0.05-3 Hz bandpass Butterworth filter (NeuroKit2 default)',
    'biosppy': 'Second order 0.1-0.35 Hz bandpass Butterworth + constant detrending',
    'hampel': 'Median-based Hampel filter - replaces outliers (3 MAD from median)',
    'custom': 'Apply user-specified bandpass/lowpass/highpass filters (Butterworth, FIR, etc.)'
}

RSP_AMPLITUDE_METHODS = ['robust', 'standardize', 'minmax', 'none']

RSP_AMPLITUDE_INFO = {
    'robust': 'Robust normalization (median + MAD) - Best for low amplitude signals with outliers',
    'standardize': 'Z-score normalization (mean + std) - Good for consistent amplitude signals',
    'minmax': 'Min-max normalization [0, 1] - Good for very low amplitude signals',
    'none': 'No normalization - Use original signal amplitude'
}

DEFAULT_RSP_PARAMS = {
    'method': 'khodadad2018',
    'rate_method': 'monotone_cubic',
    'amplitude_method': 'robust',
    'lowcut': 0.05,
    'highcut': 3.0,
    'filter_type': 'butterworth',
    'filter_order': 5,
    'apply_lowcut': True,
    'apply_highcut': True
}

# =============================================================================
# PPG CONFIGURATION
# =============================================================================

PPG_CLEANING_METHODS = ['elgendi', 'nabian2018', 'none', 'custom']

PPG_CLEANING_INFO = {
    'elgendi': 'Elgendi et al. (2013) method (NeuroKit2 default)',
    'nabian2018': 'Nabian et al. (2018) - checks heart rate for appropriate filtering',
    'none': 'No cleaning applied - returns raw signal',
    'custom': 'Apply user-specified filters (e.g., bandpass 0.5-8 Hz) instead of NeuroKit cleaning'
}

PPG_PEAK_METHODS = ['elgendi', 'bishop', 'charlton']

PPG_PEAK_INFO = {
    'elgendi': 'Elgendi et al. (2013) systolic peak detection (default)',
    'bishop': 'Bishop & Ercole (2018) - multi-scale peak detection',
    'charlton': 'Charlton et al. (2025) MSPTDfast algorithm'
}

DEFAULT_PPG_PARAMS = {
    'method': 'elgendi',
    'peak_method': 'elgendi',
    'correct_artifacts': False,
    'rate_method': 'monotone_cubic',
    'lowcut': 0.5,
    'highcut': 8.0,
    'filter_type': 'butterworth',
    'filter_order': 5,
    'apply_lowcut': True,
    'apply_highcut': True
}

# =============================================================================
# BLOOD PRESSURE CONFIGURATION
# =============================================================================

BP_FILTER_METHODS = ['bessel_25hz', 'butterworth', 'custom']

BP_FILTER_INFO = {
    'bessel_25hz': 'Third-order Bessel lowpass at 25 Hz (used by delineator algorithm)',
    'butterworth': 'Butterworth lowpass filter (configurable cutoff frequency)',
    'custom': 'User-specified digital filters (Butterworth, FIR, Chebyshev, Elliptic, etc.)'
}

BP_PEAK_METHODS = ['delineator', 'prominence']

BP_PEAK_INFO = {
    'delineator': 'MATLAB-style delineator - derivative-based detection of systolic peaks, diastolic troughs, and dicrotic notches',
    'prominence': 'Simple prominence-based peak detection using scipy.signal.find_peaks (tunable prominence parameter)'
}

DEFAULT_BP_PARAMS = {
    'filter_method': 'bessel_25hz',
    'filter_order': 3,
    'cutoff_freq': 25,
    'peak_method': 'delineator',
    'prominence': 10,
    'detect_calibration': True,
    'calibration_threshold': 0.03,
    'calibration_min_duration': 2.0,
    'calibration_padding': 0.1,
    'noise_threshold': 0.95,
    'filter_type': 'butterworth',
    'lowcut': 0.5,
    'highcut': 15.0,
    'apply_lowcut': True,
    'apply_highcut': True
}

# =============================================================================
# GENERAL FILTER CONFIGURATION
# =============================================================================

FILTER_TYPES = ['butterworth', 'fir', 'cheby1', 'cheby2', 'elliptic', 'bessel']

DEFAULT_FILTER_PARAMS = {
    'filter_type': 'butterworth',
    'filter_order': 5,
    'apply_lowcut': True,
    'apply_highcut': True
}

# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================

QUALITY_THRESHOLD_ECG = 0.5
QUALITY_THRESHOLD_BP = 0.5
QUALITY_THRESHOLD_PPG = 0.5
QUALITY_THRESHOLD_RSP = 0.5

# =============================================================================
# PEAK EDITING PARAMETERS
# =============================================================================

PEAK_ADD_WINDOW_SECONDS = 3.0
PEAK_DELETE_TOLERANCE_SECONDS = 0.5

# =============================================================================
# RATE INTERPOLATION
# =============================================================================

RATE_INTERPOLATION_METHODS = ['monotone_cubic', 'nearest', 'linear', 'quadratic', 'cubic']

RATE_INTERPOLATION_INFO = {
    'monotone_cubic': 'Monotone cubic interpolation - prevents overshoots (default, recommended)',
    'nearest': 'Nearest neighbor - step function between peaks',
    'linear': 'Linear interpolation between peaks',
    'quadratic': 'Quadratic spline interpolation',
    'cubic': 'Cubic spline interpolation'
}

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

EXPORT_DTYPE_ONSETS = 'int8'
EXPORT_DTYPE_SIGNALS = 'float32'

PEAK_ENCODING = {
    'AUTO_DETECTED': 1,
    'MANUALLY_ADDED': 2,
    'NO_PEAK': 0,
    'DELETED': -1
}

# =============================================================================
# UI THEME
# =============================================================================

THEME_COLORS = {
    'dark': {
        'background': '#0E1117',
        'secondary_bg': '#262730',
        'text': '#FAFAFA',
        'primary': '#FF4B4B'
    }
}
