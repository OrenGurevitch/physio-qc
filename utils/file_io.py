"""
File I/O functions for loading and scanning physiological data files
Pure functions with no classes
"""

import re
from pathlib import Path
import pandas as pd
import bioread

import config
from utils.conversions import convert_gas_channels


def scan_data_directory(base_path):
    """
    Scan the directory structure for available participants, sessions, and tasks

    Parameters
    ----------
    base_path : str or Path
        Base directory containing sub-*/ses-* structure

    Returns
    -------
    dict
        Nested dict: {participant: {session: [tasks]}}
    """
    base_path = Path(base_path)

    if not base_path.exists():
        return {}

    participants_data = {}

    for sub_dir in sorted(base_path.glob('sub-*')):
        if not sub_dir.is_dir():
            continue

        sub_id = sub_dir.name
        participants_data[sub_id] = {}

        for ses_dir in sorted(sub_dir.glob('ses-*')):
            if not ses_dir.is_dir():
                continue

            ses_id = ses_dir.name
            acq_files = list(ses_dir.glob('*.acq'))

            tasks = []
            for acq_file in acq_files:
                match = re.search(r'task-([^_]+)', acq_file.name)
                if match:
                    tasks.append(match.group(1))

            if tasks:
                participants_data[sub_id][ses_id] = sorted(set(tasks))

    return participants_data


def find_file_path(base_path, participant, session, task):
    """
    Find the ACQ file path for given participant, session, and task

    Parameters
    ----------
    base_path : str or Path
        Base directory containing data
    participant : str
        Participant ID (e.g., 'sub-2034')
    session : str
        Session ID (e.g., 'ses-01')
    task : str
        Task name (e.g., 'rest')

    Returns
    -------
    str or None
        Path to ACQ file or None if not found
    """
    base_path = Path(base_path)

    standard_pattern = base_path / participant / session / f"{participant}_{session}_task-{task}_physio.acq"
    if standard_pattern.exists():
        return str(standard_pattern)

    search_dir = base_path / participant / session
    if not search_dir.exists():
        return None

    matches = list(search_dir.glob(f"*task-{task}*.acq"))
    if matches:
        return str(matches[0])

    return None


def detect_signal_type(column_name):
    """
    Detect what type of signal a column contains based on its name

    Parameters
    ----------
    column_name : str
        Name of the column to check

    Returns
    -------
    str or None
        Signal type ('ecg', 'rsp', 'ppg', 'bp') or None if not detected
    """
    col_lower = column_name.lower()

    for signal_type, patterns in config.SIGNAL_PATTERNS.items():
        for pattern in patterns:
            if pattern in col_lower:
                return signal_type

    return None


def load_acq_file(file_path):
    """
    Load an ACQ file and return data with metadata

    Parameters
    ----------
    file_path : str or Path
        Path to ACQ file

    Returns
    -------
    dict or None
        Dictionary containing:
        - df: DataFrame with all channels
        - sampling_rate: Sampling rate in Hz
        - channels: List of channel names
        - signal_mappings: Dict mapping signal types to column names
        - n_samples: Number of samples
        - duration: Duration in seconds
        Returns None if file doesn't exist
    """
    if not Path(file_path).exists():
        return None

    data = bioread.read(file_path)
    sampling_rate = int(data.samples_per_second)

    channels = {}
    for ch in data.channels:
        channels[ch.name] = ch.data

    df_raw = pd.DataFrame(channels)

    # Apply gas channel conversions (voltage to mmHg for CO2/O2)
    df_raw, gas_conversions = convert_gas_channels(df_raw)

    signal_mappings = {}
    for col in df_raw.columns:
        signal_type = detect_signal_type(col)
        if signal_type:
            if signal_type not in signal_mappings:
                signal_mappings[signal_type] = col

    # Fallback: map spirometer from a fixed Biopac channel index when naming is generic
    if 'spirometer' not in signal_mappings:
        channel_idx = getattr(config, 'SPIROMETER_CHANNEL_INDEX', None)
        if isinstance(channel_idx, int) and 1 <= channel_idx <= len(df_raw.columns):
            signal_mappings['spirometer'] = df_raw.columns[channel_idx - 1]

    # Prefer converted mmHg columns over raw voltage columns
    if 'co2' in gas_conversions:
        signal_mappings['etco2'] = gas_conversions['co2']
    if 'o2' in gas_conversions:
        signal_mappings['eto2'] = gas_conversions['o2']

    return {
        'df': df_raw,
        'sampling_rate': sampling_rate,
        'channels': list(df_raw.columns),
        'signal_mappings': signal_mappings,
        'n_samples': len(df_raw),
        'duration': len(df_raw) / sampling_rate,
        'gas_conversions': gas_conversions
    }
