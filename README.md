# Physio QC

Quality control tool for physiological signals from Long COVID research project. Interactive Streamlit application for processing and manually editing ECG, RSP, PPG, and blood pressure signals.

## Features

- **Multi-signal Processing**: ECG, respiration (RSP), PPG, and blood pressure (BP)
- **Multiple Processing Methods**: 16+ algorithms for ECG peak detection, 3+ methods for each signal type
- **Manual Peak Editing**: Add, delete, and validate individual peaks and troughs
- **Quality Visualization**: Automatic detection and visualization of poor quality regions
- **Calibration Detection**: Automatic detection of calibration artifacts in blood pressure
- **BIDS-Inspired Export**: CSV data + JSON metadata with complete processing provenance
- **Custom Encoding**: Tracks auto-detected, manually added, and deleted peaks

## Installation

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### Using Makefile

```bash
make install
make run
```

### Manual

```bash
source .venv/bin/activate
streamlit run app.py
```

Then:
1. Select participant, session, and task from the sidebar
2. Click "Load Data"
3. Process each signal type in its tab
4. Manually edit peaks if needed
5. Export combined CSV + JSON

## Screenshots

### ECG Processing Interface
![ECG Menu](media/ecg_menu.png)

### Signal Visualization with Peak Editing
![ECG Visualization](media/ecg_visualization.png)

### Peak Editing Controls
![Edit Peaks Menu](media/edit_peaks_menu.png)

## Development Commands

This project includes a Makefile for common development tasks:

```bash
make help           # Show all available commands
make install        # Install dependencies
make run            # Run the Streamlit app
make lint           # Check code quality
make format         # Format code
make test           # Run tests
make clean          # Clean build artifacts
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development instructions.

## Configuration

Edit `config.py` to customize:

```python
# Data paths
BASE_DATA_PATH = '/export02/projects/LCS/01_physio'
OUTPUT_BASE_PATH = '/export02/projects/LCS/02_physio_processed'

# Processing parameters
SAMPLING_RATE = 250
POWERLINE_FREQUENCIES = [50, 60]

# Peak editing
PEAK_ADD_WINDOW_SECONDS = 3.0
PEAK_DELETE_TOLERANCE_SECONDS = 0.5
```

## Data Structure

### Input Format
Expected directory structure:
```
BASE_DATA_PATH/
├── sub-2034/
│   ├── ses-01/
│   │   ├── sub-2034_ses-01_task-rest_physio.acq
│   │   └── sub-2034_ses-01_task-stress_physio.acq
│   └── ses-02/
└── sub-2035/
```

### Output Format

#### CSV Export
Single CSV file per recording with all signals:

```csv
time,ecg_raw,ecg_clean,ecg_r_peaks,ecg_hr_interpolated,rsp_raw,rsp_clean,rsp_inhalation_onsets,rsp_exhalation_onsets,...
0.000,120.5,119.8,0,75.2,...
0.004,121.3,120.1,0,75.3,...
0.008,135.7,134.2,1,75.4,...  # Auto-detected peak
```

Peak encoding in binary columns:
- `1`: Auto-detected peak
- `2`: Manually added peak
- `0`: No peak
- `-1`: Deleted peak

#### JSON Metadata
BIDS-inspired metadata tracking all processing decisions:

```json
{
  "SamplingFrequency": 250,
  "ProcessingDate": "2025-12-18T10:30:00",
  "ECG": {
    "CleaningMethod": "neurokit",
    "PeakDetectionMethod": "neurokit",
    "AutoDetectedPeaks": 1250,
    "ManuallyAddedPeaks": 3,
    "DeletedPeaks": 7,
    "FinalPeakCount": 1246,
    "DeletedPeakIndices": [1250, 2501, ...],
    "AddedPeakIndices": [1300, 2550, ...]
  }
}
```

Output path: `{OUTPUT_BASE_PATH}/sub-{id}/ses-{id}/sub-{id}_ses-{id}_task-{task}_physio.{csv,json}`

## Signal Processing Options

### ECG
- **Cleaning**: neurokit, biosppy, Pan-Tompkins, Hamilton, Elgendi, custom filters
- **Peak Detection**: 16 algorithms including neurokit, Pan-Tompkins, Christov, Kalidas, EmrichFastNVG
- **Artifact Correction**: Lipponen & Tarvainen 2019 method
- **Quality Metrics**: Template match quality, Zhao2018 quality

### Respiration (RSP)
- **Cleaning**: Khodadad2018, BioSPPy, Hampel filter, custom
- **Detection**: Automatic peak/trough detection for inhalation/exhalation onsets
- **Metrics**: Breathing rate, breath-to-breath intervals

### PPG
- **Cleaning**: Elgendi, Nabian2018, none, custom
- **Peak Detection**: Elgendi, Bishop, Charlton MSPTDfast
- **Quality**: Continuous quality assessment

### Blood Pressure
- **Filtering**: Bessel 25Hz (default), Butterworth, custom
- **Peak Detection**:
  - **Delineator**: MATLAB-style derivative-based detection (systolic peaks, diastolic troughs, dicrotic notches)
  - **Prominence**: Simple scipy-based peak finding
- **Quality**: Calibration artifact detection, high-derivative noise detection
- **Metrics**: SBP, DBP, MAP (mean arterial pressure)

## Peak Editing

### Quick Instructions
1. **Zoom** the plot to identify the region you want to edit
2. **Enter time range** (start and end seconds)
3. **Click action**:
   - ➕ Add peaks/troughs (auto-detects in region)
   - ➖ Remove peaks/troughs (clears region)
   - 🔄 Reset (restore auto-detected)

### Visual Markers
- 🔴 Valid peaks | 🔵 Valid troughs
- ❌ Deleted | ✅ Manually added

### Available for All Signals
- **ECG**: R-peaks
- **RSP**: Inhalation peaks + exhalation troughs
- **PPG**: Systolic peaks
- **BP**: Systolic peaks + diastolic troughs

## Project Structure

```
physio-qc/
├── app.py                     # Main Streamlit application
├── config.py                  # Configuration (paths, defaults, parameters)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── metrics/                   # Signal processing per metric
│   ├── ecg.py                 # ECG processing functions
│   ├── rsp.py                 # RSP processing functions
│   ├── ppg.py                 # PPG processing functions
│   └── blood_pressure.py      # BP processing functions
│
├── algorithms/                # Specialized algorithms
│   ├── bp_delineator.py       # BP fiducial point detection
│   └── quality_detection.py   # Quality metrics and artifact detection
│
└── utils/                     # Utilities
    ├── file_io.py             # ACQ file loading, directory scanning
    ├── peak_editing.py        # Manual peak editing logic
    └── export.py              # CSV and JSON export functions
```

## Troubleshooting

### Data not loading
- Check that `BASE_DATA_PATH` in config.py points to your data directory
- Verify directory structure matches expected format (sub-*/ses-*/*.acq)
- Ensure .acq files contain recognized signal types (check SIGNAL_PATTERNS in config)

### Processing fails
- "Insufficient peaks detected": Try different cleaning or peak detection methods
- Signal too noisy: Use stronger filtering methods or custom filters
- Check sampling rate matches your data (default 250 Hz)

### Export issues
- Ensure `OUTPUT_BASE_PATH` directory exists and is writable
- Check participant/session names don't contain special characters

