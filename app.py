"""
Physiological Signal QC Application
Streamlit-based interface for quality control of physiological signals
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from utils.file_io import scan_data_directory, find_file_path, load_acq_file
from metrics import ecg, rsp, ppg, blood_pressure
from utils import peak_editing, export


st.set_page_config(
    page_title="Physio QC",
    page_icon="📈",
    layout="wide"
)


CSS = """
<style>
    .stApp {
        background-color: #0E1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px;
        padding: 10px 20px;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
    }
    .metric-box {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None

    if 'ecg_result' not in st.session_state:
        st.session_state.ecg_result = None

    if 'rsp_result' not in st.session_state:
        st.session_state.rsp_result = None

    if 'ppg_result' not in st.session_state:
        st.session_state.ppg_result = None

    if 'bp_result' not in st.session_state:
        st.session_state.bp_result = None

    if 'ecg_params' not in st.session_state:
        st.session_state.ecg_params = config.DEFAULT_ECG_PARAMS.copy()

    if 'rsp_params' not in st.session_state:
        st.session_state.rsp_params = config.DEFAULT_RSP_PARAMS.copy()

    if 'ppg_params' not in st.session_state:
        st.session_state.ppg_params = config.DEFAULT_PPG_PARAMS.copy()

    if 'bp_params' not in st.session_state:
        st.session_state.bp_params = config.DEFAULT_BP_PARAMS.copy()

    # Zoom ranges for each signal type
    if 'ecg_zoom_range' not in st.session_state:
        st.session_state.ecg_zoom_range = None

    if 'rsp_zoom_range' not in st.session_state:
        st.session_state.rsp_zoom_range = None

    if 'ppg_zoom_range' not in st.session_state:
        st.session_state.ppg_zoom_range = None

    if 'bp_zoom_range' not in st.session_state:
        st.session_state.bp_zoom_range = None


def create_signal_plot(time, raw, clean, current_peaks, auto_peaks, signal_name, sampling_rate,
                       hr_interpolated=None, hr_bpm=None, quality_continuous=None,
                       selected_quality_metrics=None, quality_data=None, ui_revision='constant',
                       zoom_range=None):
    """Create 3-panel plot for signal visualization with synchronized zooming

    Args:
        zoom_range: Optional tuple (start, end) to set the x-axis zoom range in seconds
    """
    deleted_peaks = np.setdiff1d(auto_peaks, current_peaks)
    added_peaks = np.setdiff1d(current_peaks, auto_peaks)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Raw vs Clean', 'Clean with Peaks', f'{signal_name} Rate'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Row 1: Raw vs Clean
    fig.add_trace(go.Scatter(x=time, y=raw, name='Raw', line=dict(color='#808080', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=clean, name='Clean', line=dict(color='#00D4FF', width=1)), row=1, col=1)

    # Row 2: Clean with Peaks (and quality on secondary y-axis if available)
    fig.add_trace(go.Scatter(x=time, y=clean, name='Signal', line=dict(color='#00D4FF', width=1)), row=2, col=1, secondary_y=False)

    if len(current_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[current_peaks], y=clean[current_peaks],
            mode='markers', name='Valid Peaks',
            marker=dict(color='#FF4444', size=8, symbol='circle')
        ), row=2, col=1, secondary_y=False)

    if len(deleted_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[deleted_peaks], y=clean[deleted_peaks],
            mode='markers', name='Deleted Peaks',
            marker=dict(color='#FF4444', size=10, symbol='x')
        ), row=2, col=1, secondary_y=False)

    if len(added_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[added_peaks], y=clean[added_peaks],
            mode='markers', name='Added Peaks',
            marker=dict(color='#00FF00', size=10, symbol='cross')
        ), row=2, col=1, secondary_y=False)

    # Add selected quality metrics on secondary y-axis
    if selected_quality_metrics and quality_data:
        quality_colors = {
            'quality_templatematch': '#FFA500',
            'quality_averageqrs': '#FF69B4'
        }
        quality_names = {
            'quality_templatematch': 'Template Match',
            'quality_averageqrs': 'Average QRS'
        }
        for i, metric in enumerate(selected_quality_metrics):
            if metric in quality_data and quality_data[metric] is not None:
                metric_data = quality_data[metric]
                if isinstance(metric_data, np.ndarray) and len(metric_data) > 0:
                    color = quality_colors.get(metric, f'hsl({(i*60)%360}, 70%, 60%)')
                    name = quality_names.get(metric, metric)
                    fig.add_trace(go.Scatter(
                        x=time, y=metric_data,
                        name=name,
                        line=dict(color=color, width=1.5, dash='dot'),
                        opacity=0.7
                    ), row=2, col=1, secondary_y=True)

    # Row 3: Heart Rate
    if hr_interpolated is not None:
        fig.add_trace(go.Scatter(
            x=time, y=hr_interpolated,
            name='HR Interpolated',
            line=dict(color='#FF6B6B', width=2)
        ), row=3, col=1)

    if hr_bpm is not None and len(current_peaks) > 1:
        # Plot beat-to-beat HR at peak locations (excluding first peak)
        peak_times = time[current_peaks[1:]]  # Start from second peak
        fig.add_trace(go.Scatter(
            x=peak_times, y=hr_bpm,
            mode='markers',
            name='Beat-to-Beat HR',
            marker=dict(color='#FFA500', size=6)
        ), row=3, col=1)

    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Quality (0-1)", row=2, col=1, secondary_y=True, range=[0, 1])
    fig.update_yaxes(title_text="HR (bpm)", row=3, col=1)

    # Synchronize x-axes across all subplots
    fig.update_xaxes(matches='x')

    # Apply zoom range if specified
    if zoom_range is not None:
        fig.update_xaxes(range=[zoom_range[0], zoom_range[1]])

    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified',
        uirevision=ui_revision  # Preserve zoom state across updates
    )

    return fig


def create_rsp_bp_plot(time, raw, clean, current_peaks, current_troughs, auto_peaks, auto_troughs, signal_name, rate_interpolated=None, rate_bpm=None, map_values=None, ui_revision='constant', zoom_range=None):
    """Create 3-panel plot for RSP/BP with both peaks and troughs and synchronized zooming

    Args:
        zoom_range: Optional tuple (start, end) to set the x-axis zoom range in seconds
    """
    deleted_peaks = np.setdiff1d(auto_peaks, current_peaks)
    added_peaks = np.setdiff1d(current_peaks, auto_peaks)
    deleted_troughs = np.setdiff1d(auto_troughs, current_troughs)
    added_troughs = np.setdiff1d(current_troughs, auto_troughs)

    # Determine the third panel label based on signal type
    if signal_name == 'BP':
        third_panel_label = 'Mean Arterial Pressure (MAP)'
        y_axis_label = 'MAP (mmHg)'
    else:
        third_panel_label = f'{signal_name} Rate'
        y_axis_label = 'Rate (bpm)'

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Raw vs Filtered', 'Signal with Peaks/Troughs', third_panel_label),
        vertical_spacing=0.1
    )

    # Row 1: Raw vs Filtered
    fig.add_trace(go.Scatter(x=time, y=raw, name='Raw', line=dict(color='#808080', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=clean, name='Filtered', line=dict(color='#00D4FF', width=1)), row=1, col=1)

    # Row 2: Signal with Peaks/Troughs
    fig.add_trace(go.Scatter(x=time, y=clean, name='Signal', line=dict(color='#00D4FF', width=1), showlegend=False), row=2, col=1)

    if len(current_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[current_peaks], y=clean[current_peaks],
            mode='markers', name='Valid Peaks',
            marker=dict(color='#FF4444', size=8, symbol='circle')
        ), row=2, col=1)

    if len(current_troughs) > 0:
        fig.add_trace(go.Scatter(
            x=time[current_troughs], y=clean[current_troughs],
            mode='markers', name='Valid Troughs',
            marker=dict(color='#4444FF', size=8, symbol='circle')
        ), row=2, col=1)

    if len(deleted_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[deleted_peaks], y=clean[deleted_peaks],
            mode='markers', name='Deleted Peaks',
            marker=dict(color='#FF4444', size=10, symbol='x')
        ), row=2, col=1)

    if len(added_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[added_peaks], y=clean[added_peaks],
            mode='markers', name='Added Peaks',
            marker=dict(color='#00FF00', size=10, symbol='cross')
        ), row=2, col=1)

    if len(deleted_troughs) > 0:
        fig.add_trace(go.Scatter(
            x=time[deleted_troughs], y=clean[deleted_troughs],
            mode='markers', name='Deleted Troughs',
            marker=dict(color='#4444FF', size=10, symbol='x')
        ), row=2, col=1)

    if len(added_troughs) > 0:
        fig.add_trace(go.Scatter(
            x=time[added_troughs], y=clean[added_troughs],
            mode='markers', name='Added Troughs',
            marker=dict(color='#00FFFF', size=10, symbol='cross')
        ), row=2, col=1)

    # Row 3: Rate (BR for RSP) or MAP (for BP)
    if signal_name == 'BP' and map_values is not None:
        # Plot MAP for blood pressure
        fig.add_trace(go.Scatter(
            x=time, y=map_values,
            name='MAP',
            line=dict(color='#9B59B6', width=2)
        ), row=3, col=1)
    elif rate_interpolated is not None:
        # Plot breathing rate for RSP
        fig.add_trace(go.Scatter(
            x=time, y=rate_interpolated,
            name='Rate Interpolated',
            line=dict(color='#FF6B6B', width=2)
        ), row=3, col=1)

    if rate_bpm is not None and len(current_troughs) > 1:
        # Plot rate at trough locations (excluding first trough)
        trough_times = time[current_troughs[1:]]
        fig.add_trace(go.Scatter(
            x=trough_times, y=rate_bpm,
            mode='markers',
            name='Breath-to-Breath Rate',
            marker=dict(color='#FFA500', size=6)
        ), row=3, col=1)

    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text=y_axis_label, row=3, col=1)

    # Synchronize x-axes across all subplots
    fig.update_xaxes(matches='x')

    # Apply zoom range if specified
    if zoom_range is not None:
        fig.update_xaxes(range=[zoom_range[0], zoom_range[1]])

    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified',
        uirevision=ui_revision  # Preserve zoom state across updates
    )

    return fig


def main():
    """Main application function"""
    init_session_state()

    st.title("📈 Physiological Signal QC")

    with st.sidebar:
        st.header("Data Selection")

        # Check if data path exists and provide override option
        import os
        data_path_exists = os.path.isdir(config.BASE_DATA_PATH)

        if not data_path_exists:
            st.warning(f"⚠️ Default data path doesn't exist:\n`{config.BASE_DATA_PATH}`")

            with st.expander("🔧 Configure Data Paths", expanded=True):
                st.info("Update the paths below to point to your data directories, then click Apply.")

                # Allow user to override paths
                custom_data_path = st.text_input(
                    "Data Path",
                    value=config.BASE_DATA_PATH,
                    help="Path to raw physiological data files"
                )

                custom_output_path = st.text_input(
                    "Output Path",
                    value=config.OUTPUT_BASE_PATH,
                    help="Path where processed data will be saved"
                )

                if st.button("Apply Paths", type="primary"):
                    if os.path.isdir(custom_data_path):
                        config.BASE_DATA_PATH = custom_data_path
                        config.OUTPUT_BASE_PATH = custom_output_path
                        st.success("✅ Paths updated! Scanning for data...")
                        st.rerun()
                    else:
                        st.error(f"❌ Path doesn't exist: {custom_data_path}")

                st.stop()

        participants_data = scan_data_directory(config.BASE_DATA_PATH)

        if not participants_data:
            st.error(f"No data found in {config.BASE_DATA_PATH}")
            st.info("Make sure your data follows the expected structure:\n"
                    "`sub-{id}/ses-{id}/sub-{id}_ses-{id}_task-{task}_physio.{acq,csv}`")
            return

        participants = list(participants_data.keys())
        participant = st.selectbox("Participant", participants)

        sessions = list(participants_data[participant].keys())
        session = st.selectbox("Session", sessions)

        tasks = participants_data[participant][session]
        task = st.selectbox("Task", tasks)

        if st.button("Load Data", type="primary"):
            file_path = find_file_path(config.BASE_DATA_PATH, participant, session, task)

            if file_path is None:
                st.error("File not found")
                return

            data = load_acq_file(file_path)

            if data is None:
                st.error("Failed to load file")
                return

            st.session_state.loaded_data = data
            st.session_state.data_loaded = True
            st.session_state.participant = participant
            st.session_state.session = session
            st.session_state.task = task

            st.session_state.ecg_result = None
            st.session_state.rsp_result = None
            st.session_state.ppg_result = None
            st.session_state.bp_result = None

            st.success(f"Loaded {file_path}")

        if st.session_state.data_loaded:
            data = st.session_state.loaded_data
            st.info(f"""
            **Samples**: {data['n_samples']:,}
            **Duration**: {data['duration']:.1f}s
            **Sampling Rate**: {data['sampling_rate']} Hz
            **Signals**: {', '.join(data['signal_mappings'].keys())}
            """)

    if not st.session_state.data_loaded:
        st.info("👈 Select data from the sidebar to begin")
        return

    data = st.session_state.loaded_data
    sampling_rate = data['sampling_rate']
    detected_signals = list(data['signal_mappings'].keys())

    tabs = []
    if 'ecg' in detected_signals:
        tabs.append("ECG")
    if 'rsp' in detected_signals:
        tabs.append("RSP")
    if 'ppg' in detected_signals:
        tabs.append("PPG")
    if 'bp' in detected_signals:
        tabs.append("Blood Pressure")
    tabs.append("Export")

    tab_objects = st.tabs(tabs)
    tab_idx = 0

    if 'ecg' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("ECG Processing")

            col1, col2 = st.columns([2, 1])

            with col1:
                method = st.selectbox("Cleaning Method", config.ECG_CLEANING_METHODS, key='ecg_method')
                with st.expander("ℹ️ Method Info"):
                    st.info(config.ECG_CLEANING_INFO.get(method, "No info available"))

                if method == 'custom':
                    st.subheader("Custom Filter Options")
                    filter_type = st.selectbox("Filter Type", config.FILTER_TYPES, key='ecg_filter_type')
                    filter_mode = st.radio("Filter Mode", ["Bandpass", "Lowpass", "Highpass"], horizontal=True, key='ecg_filter_mode')

                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        if filter_mode in ["Highpass", "Bandpass"]:
                            lowcut = st.number_input("High-pass (Hz)", min_value=0.01, max_value=100.0, value=0.5, step=0.1, key='ecg_lowcut')
                        else:
                            lowcut = None
                    with col_f2:
                        if filter_mode in ["Lowpass", "Bandpass"]:
                            highcut = st.number_input("Low-pass (Hz)", min_value=0.1, max_value=200.0, value=45.0, step=0.5, key='ecg_highcut')
                        else:
                            highcut = None

                    filter_order = st.slider("Filter Order", min_value=1, max_value=10, value=5, key='ecg_filter_order')
                else:
                    filter_type = 'butterworth'
                    filter_mode = "Bandpass"
                    lowcut = 0.5
                    highcut = 45.0
                    filter_order = 5

                peak_method = st.selectbox("Peak Detection", config.ECG_PEAK_METHODS, key='ecg_peak')
                with st.expander("ℹ️ Peak Method Info"):
                    st.info(config.ECG_PEAK_INFO.get(peak_method, "No info available"))

            with col2:
                powerline = st.selectbox("Powerline Frequency", config.POWERLINE_FREQUENCIES, key='ecg_powerline')
                correct_artifacts = st.checkbox("Artifact Correction", key='ecg_correct')
                calculate_quality = st.checkbox("Calculate Quality", value=True, key='ecg_quality')

            if st.button("Process ECG", type="primary"):
                signal = data['df'][data['signal_mappings']['ecg']].values

                params = {
                    'method': method,
                    'powerline': powerline,
                    'peak_method': peak_method,
                    'correct_artifacts': correct_artifacts,
                    'calculate_quality': calculate_quality,
                    'filter_type': filter_type,
                    'filter_order': filter_order,
                    'lowcut': lowcut,
                    'highcut': highcut,
                    'apply_lowcut': lowcut is not None,
                    'apply_highcut': highcut is not None
                }
                st.session_state.ecg_params.update(params)

                result = ecg.process_ecg(signal, sampling_rate, st.session_state.ecg_params)

                if result is None:
                    st.error("Processing failed: insufficient peaks detected")
                else:
                    st.session_state.ecg_result = result
                    st.success("ECG processed successfully")

            if st.session_state.ecg_result is not None:
                result = st.session_state.ecg_result

                st.subheader("Manual Peak Editing")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Auto-detected Peaks", len(result['auto_r_peaks']))
                with col2:
                    n_added = len(np.setdiff1d(result['current_r_peaks'], result['auto_r_peaks']))
                    st.metric("Manually Added", n_added)
                with col3:
                    n_deleted = len(np.setdiff1d(result['auto_r_peaks'], result['current_r_peaks']))
                    st.metric("Deleted", n_deleted)

                # Quality metrics selection
                quality_available = []
                if st.session_state.ecg_params.get('calculate_quality', False):
                    st.subheader("Quality Metrics Display")

                    # Continuous metrics
                    continuous_metrics = []
                    if result.get('quality_templatematch') is not None:
                        continuous_metrics.append('quality_templatematch')
                    if result.get('quality_averageqrs') is not None:
                        continuous_metrics.append('quality_averageqrs')

                    # Overall metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if result.get('quality_templatematch_mean') is not None:
                            st.metric("Template Match Quality", f"{result['quality_templatematch_mean']:.3f}")
                    with col2:
                        if result.get('quality_averageqrs_mean') is not None:
                            st.metric("Average QRS Quality", f"{result['quality_averageqrs_mean']:.3f}")
                    with col3:
                        if result.get('quality_mean') is not None:
                            st.metric("Overall Quality", f"{result['quality_mean']:.3f}")

                    # Multiselect for continuous metrics
                    selected_quality_metrics = st.multiselect(
                        "Select continuous quality metrics to plot:",
                        options=continuous_metrics,
                        default=continuous_metrics,
                        key='ecg_quality_select',
                        format_func=lambda x: 'Template Match' if x == 'quality_templatematch' else 'Average QRS'
                    )
                else:
                    selected_quality_metrics = []

                time = np.arange(len(result['clean'])) / sampling_rate

                # Recalculate HR based on current peaks
                from metrics.ecg import calculate_hr
                if len(result['current_r_peaks']) > 1:
                    hr_data = calculate_hr(
                        result['current_r_peaks'],
                        sampling_rate,
                        len(result['clean']),
                        rate_method=st.session_state.ecg_params.get('rate_method', 'monotone_cubic')
                    )
                    result.update(hr_data)
                else:
                    result['hr_bpm'] = np.array([])
                    result['hr_interpolated'] = np.zeros(len(result['clean']))
                    result['mean_hr'] = 0.0
                    result['std_hr'] = 0.0

                # Create quality data dict for plotting
                quality_data = {
                    'quality_templatematch': result.get('quality_templatematch'),
                    'quality_averageqrs': result.get('quality_averageqrs')
                }

                # Initialize region range in session state if not exists (needed before plotting)
                if 'ecg_region_start' not in st.session_state:
                    st.session_state.ecg_region_start = 0.0
                if 'ecg_region_end' not in st.session_state:
                    st.session_state.ecg_region_end = min(10.0, float(time[-1]))

                # Get zoom range from session state
                ecg_zoom = (st.session_state.ecg_region_start, st.session_state.ecg_region_end)

                fig = create_signal_plot(
                    time, result['raw'], result['clean'],
                    result['current_r_peaks'], result['auto_r_peaks'],
                    'ECG', sampling_rate,
                    hr_interpolated=result.get('hr_interpolated'),
                    hr_bpm=result.get('hr_bpm'),
                    selected_quality_metrics=selected_quality_metrics,
                    quality_data=quality_data,
                    ui_revision='ecg_plot',  # Preserve zoom state
                    zoom_range=ecg_zoom  # Apply zoom from region inputs
                )

                st.plotly_chart(fig, use_container_width=True)

                # Drag-based editing interface
                st.subheader("Drag-Based Peak Editing")

                # Quick navigation buttons
                st.write("**Quick Range Selection:**")
                col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
                with col_btn1:
                    if st.button("⏮️ First 10s", key='ecg_first_10s'):
                        st.session_state.ecg_region_start = 0.0
                        st.session_state.ecg_region_end = min(10.0, float(time[-1]))
                        st.rerun()
                with col_btn2:
                    if st.button("◀️ Previous 10s", key='ecg_prev_10s'):
                        window = 10.0
                        new_start = max(0.0, st.session_state.ecg_region_start - window)
                        new_end = max(window, st.session_state.ecg_region_end - window)
                        st.session_state.ecg_region_start = new_start
                        st.session_state.ecg_region_end = min(new_end, float(time[-1]))
                        st.rerun()
                with col_btn3:
                    if st.button("▶️ Next 10s", key='ecg_next_10s'):
                        window = 10.0
                        new_start = min(float(time[-1]) - window, st.session_state.ecg_region_start + window)
                        new_end = min(float(time[-1]), st.session_state.ecg_region_end + window)
                        st.session_state.ecg_region_start = new_start
                        st.session_state.ecg_region_end = new_end
                        st.rerun()
                with col_btn4:
                    if st.button("⏭️ Last 10s", key='ecg_last_10s'):
                        st.session_state.ecg_region_start = max(0.0, float(time[-1]) - 10.0)
                        st.session_state.ecg_region_end = float(time[-1])
                        st.rerun()
                with col_btn5:
                    if st.button("🔄 Reset Range", key='ecg_reset_range'):
                        st.session_state.ecg_region_start = 0.0
                        st.session_state.ecg_region_end = min(10.0, float(time[-1]))
                        st.rerun()

                st.write("**Manual Range Entry:** (Or look at zoomed plot X-axis and enter values)")
                col1, col2 = st.columns(2)
                with col1:
                    region_start = st.number_input(
                        "Region Start (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.ecg_region_start),
                        step=1.0,
                        format="%.2f",
                        key='ecg_region_start_input',
                        help="Enter the start time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.ecg_region_start = region_start
                with col2:
                    region_end = st.number_input(
                        "Region End (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.ecg_region_end),
                        step=1.0,
                        format="%.2f",
                        key='ecg_region_end_input',
                        help="Enter the end time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.ecg_region_end = region_end

                # Show current range info
                st.caption(f"Current range: {region_start:.2f}s to {region_end:.2f}s ({region_end - region_start:.2f}s window) | Full signal: {float(time[-1]):.2f}s")

                st.write("**Choose Action:**")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("➕ Add R-Peaks in Region", type="primary", key='ecg_add_region_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.ecg_result['current_r_peaks'] = peak_editing.add_peaks_in_range(
                            result['clean'],
                            result['current_r_peaks'],
                            region_start,
                            region_end,
                            sampling_rate
                        )
                        st.rerun()

                with col2:
                    if st.button("➖ Remove R-Peaks in Region", type="secondary", key='ecg_remove_region_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.ecg_result['current_r_peaks'] = peak_editing.erase_peaks_in_range(
                            result['current_r_peaks'],
                            region_start,
                            region_end,
                            sampling_rate
                        )
                        st.rerun()

                with col3:
                    if st.button("🔄 Reset to Auto-Detected", key='ecg_reset_btn', use_container_width=True):
                        st.session_state.ecg_result['current_r_peaks'] = result['auto_r_peaks'].copy()
                        st.rerun()

                # Single peak editing
                with st.expander("✏️ Single Peak Editing (Advanced)"):
                    st.write("Add or remove individual peaks at specific times.")
                    col1, col2 = st.columns(2)
                    with col1:
                        single_peak_time = st.number_input(
                            "Time (seconds)",
                            min_value=0.0,
                            max_value=float(time[-1]),
                            value=0.0,
                            step=0.1,
                            key='ecg_single_peak_time'
                        )
                    with col2:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Add at Time", key='ecg_single_add_btn'):
                                from utils import peak_editing
                                st.session_state.ecg_result['current_r_peaks'] = peak_editing.add_peak(
                                    result['clean'],
                                    result['current_r_peaks'],
                                    single_peak_time,
                                    sampling_rate
                                )
                                st.rerun()
                        with col_b:
                            if st.button("Delete at Time", key='ecg_single_del_btn'):
                                from utils import peak_editing
                                st.session_state.ecg_result['current_r_peaks'] = peak_editing.delete_peak(
                                    result['current_r_peaks'],
                                    single_peak_time,
                                    sampling_rate
                                )
                                st.rerun()

                st.subheader("Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Peak Count", len(result['current_r_peaks']))
                with col2:
                    st.metric("Mean HR", f"{result['mean_hr']:.1f} bpm")
                with col3:
                    st.metric("HR Std Dev", f"{result['std_hr']:.1f} bpm")

        tab_idx += 1

    if 'rsp' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("RSP Processing")

            col1, col2 = st.columns([2, 1])

            with col1:
                method = st.selectbox("Cleaning Method", config.RSP_CLEANING_METHODS, key='rsp_method')
                with st.expander("ℹ️ Method Info"):
                    st.info(config.RSP_CLEANING_INFO.get(method, "No info available"))

            with col2:
                amplitude_method = st.selectbox("Amplitude Normalization", config.RSP_AMPLITUDE_METHODS, key='rsp_amplitude')
                with st.expander("ℹ️ Amplitude Info"):
                    st.info(config.RSP_AMPLITUDE_INFO.get(amplitude_method, "No info available"))

            if st.button("Process RSP", type="primary"):
                signal = data['df'][data['signal_mappings']['rsp']].values

                params = {
                    'method': method,
                    'amplitude_method': amplitude_method if amplitude_method != 'none' else None
                }
                st.session_state.rsp_params.update(params)

                result = rsp.process_rsp(signal, sampling_rate, st.session_state.rsp_params)

                if result is None:
                    st.error("Processing failed: insufficient breaths detected")
                else:
                    st.session_state.rsp_result = result
                    st.success("RSP processed successfully")

            if st.session_state.rsp_result is not None:
                result = st.session_state.rsp_result

                st.subheader("Manual Breath Editing")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Auto Inhalations", len(result['auto_peaks']))
                with col2:
                    st.metric("Auto Exhalations", len(result['auto_troughs']))
                with col3:
                    n_added_peaks = len(np.setdiff1d(result['current_peaks'], result['auto_peaks']))
                    st.metric("Added Inhalations", n_added_peaks)
                with col4:
                    n_added_troughs = len(np.setdiff1d(result['current_troughs'], result['auto_troughs']))
                    st.metric("Added Exhalations", n_added_troughs)

                time = np.arange(len(result['clean'])) / sampling_rate

                # Recalculate BR based on current troughs
                from metrics.rsp import calculate_breathing_rate
                if len(result['current_troughs']) > 1:
                    br_data = calculate_breathing_rate(
                        result['current_troughs'],
                        sampling_rate,
                        len(result['clean']),
                        rate_method=st.session_state.rsp_params.get('rate_method', 'monotone_cubic')
                    )
                    result.update(br_data)
                else:
                    result['br_bpm'] = np.array([])
                    result['br_interpolated'] = np.zeros(len(result['clean']))
                    result['mean_br'] = 0.0
                    result['std_br'] = 0.0

                # Initialize region range in session state if not exists (needed before plotting)
                if 'rsp_region_start' not in st.session_state:
                    st.session_state.rsp_region_start = 0.0
                if 'rsp_region_end' not in st.session_state:
                    st.session_state.rsp_region_end = min(10.0, float(time[-1]))

                # Get zoom range from session state
                rsp_zoom = (st.session_state.rsp_region_start, st.session_state.rsp_region_end)

                fig = create_rsp_bp_plot(
                    time, result['raw'], result['clean'],
                    result['current_peaks'], result['current_troughs'],
                    result['auto_peaks'], result['auto_troughs'],
                    'RSP',
                    rate_interpolated=result.get('br_interpolated'),
                    rate_bpm=result.get('br_bpm'),
                    ui_revision='rsp_plot',  # Preserve zoom state
                    zoom_range=rsp_zoom  # Apply zoom from region inputs
                )

                st.plotly_chart(fig, use_container_width=True)

                # Drag-based editing interface
                st.subheader("Drag-Based Breath Editing")

                # Quick navigation buttons
                st.write("**Quick Range Selection:**")
                col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
                with col_btn1:
                    if st.button("⏮️ First 10s", key='rsp_first_10s'):
                        st.session_state.rsp_region_start = 0.0
                        st.session_state.rsp_region_end = min(10.0, float(time[-1]))
                        st.rerun()
                with col_btn2:
                    if st.button("◀️ Previous 10s", key='rsp_prev_10s'):
                        window = 10.0
                        new_start = max(0.0, st.session_state.rsp_region_start - window)
                        new_end = max(window, st.session_state.rsp_region_end - window)
                        st.session_state.rsp_region_start = new_start
                        st.session_state.rsp_region_end = min(new_end, float(time[-1]))
                        st.rerun()
                with col_btn3:
                    if st.button("▶️ Next 10s", key='rsp_next_10s'):
                        window = 10.0
                        new_start = min(float(time[-1]) - window, st.session_state.rsp_region_start + window)
                        new_end = min(float(time[-1]), st.session_state.rsp_region_end + window)
                        st.session_state.rsp_region_start = new_start
                        st.session_state.rsp_region_end = new_end
                        st.rerun()
                with col_btn4:
                    if st.button("⏭️ Last 10s", key='rsp_last_10s'):
                        st.session_state.rsp_region_start = max(0.0, float(time[-1]) - 10.0)
                        st.session_state.rsp_region_end = float(time[-1])
                        st.rerun()
                with col_btn5:
                    if st.button("🔄 Reset Range", key='rsp_reset_range'):
                        st.session_state.rsp_region_start = 0.0
                        st.session_state.rsp_region_end = min(10.0, float(time[-1]))
                        st.rerun()

                st.write("**Manual Range Entry:** (Or look at zoomed plot X-axis and enter values)")
                col1, col2 = st.columns(2)
                with col1:
                    region_start = st.number_input(
                        "Region Start (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.rsp_region_start),
                        step=1.0,
                        format="%.2f",
                        key='rsp_region_start_input',
                        help="Enter the start time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.rsp_region_start = region_start
                with col2:
                    region_end = st.number_input(
                        "Region End (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.rsp_region_end),
                        step=1.0,
                        format="%.2f",
                        key='rsp_region_end_input',
                        help="Enter the end time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.rsp_region_end = region_end

                # Show current range info
                st.caption(f"Current range: {region_start:.2f}s to {region_end:.2f}s ({region_end - region_start:.2f}s window) | Full signal: {float(time[-1]):.2f}s")

                st.write("**Inhalation Peaks:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("➕ Add Inhalation Peaks", type="primary", key='rsp_add_peaks_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.rsp_result['current_peaks'] = peak_editing.add_peaks_in_range(
                            result['clean'], result['current_peaks'], region_start, region_end, sampling_rate, min_distance_seconds=1.0
                        )
                        st.rerun()
                with col2:
                    if st.button("➖ Remove Inhalation Peaks", type="secondary", key='rsp_remove_peaks_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.rsp_result['current_peaks'] = peak_editing.erase_peaks_in_range(
                            result['current_peaks'], region_start, region_end, sampling_rate
                        )
                        st.rerun()
                with col3:
                    if st.button("🔄 Reset Inhalations", key='rsp_reset_peaks_btn', use_container_width=True):
                        st.session_state.rsp_result['current_peaks'] = result['auto_peaks'].copy()
                        st.rerun()

                st.write("**Exhalation Troughs:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("➕ Add Exhalation Troughs", type="primary", key='rsp_add_troughs_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.rsp_result['current_troughs'] = peak_editing.add_troughs_in_range(
                            result['clean'], result['current_troughs'], region_start, region_end, sampling_rate, min_distance_seconds=1.0
                        )
                        st.rerun()
                with col2:
                    if st.button("➖ Remove Exhalation Troughs", type="secondary", key='rsp_remove_troughs_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.rsp_result['current_troughs'] = peak_editing.erase_troughs_in_range(
                            result['current_troughs'], region_start, region_end, sampling_rate
                        )
                        st.rerun()
                with col3:
                    if st.button("🔄 Reset Exhalations", key='rsp_reset_troughs_btn', use_container_width=True):
                        st.session_state.rsp_result['current_troughs'] = result['auto_troughs'].copy()
                        st.rerun()

                # Single peak/trough editing
                with st.expander("✏️ Single Peak/Trough Editing (Advanced)"):
                    st.write("Add or remove individual peaks/troughs at specific times.")
                    col1, col2 = st.columns(2)
                    with col1:
                        single_time = st.number_input("Time (seconds)", min_value=0.0, max_value=float(time[-1]), value=0.0, step=0.1, key='rsp_single_time')

                    st.write("**Inhalation Peaks:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Add Inhalation at Time", key='rsp_single_add_peak'):
                            from utils import peak_editing
                            st.session_state.rsp_result['current_peaks'] = peak_editing.add_peak(
                                result['clean'], result['current_peaks'], single_time, sampling_rate
                            )
                            st.rerun()
                    with col_b:
                        if st.button("Delete Inhalation at Time", key='rsp_single_del_peak'):
                            from utils import peak_editing
                            st.session_state.rsp_result['current_peaks'] = peak_editing.delete_peak(
                                result['current_peaks'], single_time, sampling_rate
                            )
                            st.rerun()

                    st.write("**Exhalation Troughs:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Add Exhalation at Time", key='rsp_single_add_trough'):
                            from utils import peak_editing
                            st.session_state.rsp_result['current_troughs'] = peak_editing.add_trough(
                                result['clean'], result['current_troughs'], single_time, sampling_rate
                            )
                            st.rerun()
                    with col_b:
                        if st.button("Delete Exhalation at Time", key='rsp_single_del_trough'):
                            from utils import peak_editing
                            st.session_state.rsp_result['current_troughs'] = peak_editing.delete_trough(
                                result['current_troughs'], single_time, sampling_rate
                            )
                            st.rerun()

                st.subheader("Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Breath Count", len(result['current_troughs']))
                with col2:
                    st.metric("Mean BR", f"{result['mean_br']:.1f} bpm")
                with col3:
                    st.metric("BR Std Dev", f"{result['std_br']:.1f} bpm")

        tab_idx += 1

    if 'ppg' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("PPG Processing")

            col1, col2 = st.columns([2, 1])

            with col1:
                method = st.selectbox("Cleaning Method", config.PPG_CLEANING_METHODS, key='ppg_method')
                with st.expander("ℹ️ Method Info"):
                    st.info(config.PPG_CLEANING_INFO.get(method, "No info available"))

                peak_method = st.selectbox("Peak Detection", config.PPG_PEAK_METHODS, key='ppg_peak')
                with st.expander("ℹ️ Peak Method Info"):
                    st.info(config.PPG_PEAK_INFO.get(peak_method, "No info available"))

            with col2:
                correct_artifacts = st.checkbox("Artifact Correction", key='ppg_correct')

            if st.button("Process PPG", type="primary"):
                signal = data['df'][data['signal_mappings']['ppg']].values

                params = {
                    'method': method,
                    'peak_method': peak_method,
                    'correct_artifacts': correct_artifacts
                }
                st.session_state.ppg_params.update(params)

                result = ppg.process_ppg(signal, sampling_rate, st.session_state.ppg_params)

                if result is None:
                    st.error("Processing failed: insufficient peaks detected")
                else:
                    st.session_state.ppg_result = result
                    st.success("PPG processed successfully")

            if st.session_state.ppg_result is not None:
                result = st.session_state.ppg_result

                st.subheader("Manual Peak Editing")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Auto-detected Peaks", len(result['auto_peaks']))
                with col2:
                    n_added = len(np.setdiff1d(result['current_peaks'], result['auto_peaks']))
                    st.metric("Manually Added", n_added)
                with col3:
                    n_deleted = len(np.setdiff1d(result['auto_peaks'], result['current_peaks']))
                    st.metric("Deleted", n_deleted)

                time = np.arange(len(result['clean'])) / sampling_rate

                # Recalculate HR based on current peaks
                from metrics.ppg import calculate_hr_from_ppg
                if len(result['current_peaks']) > 1:
                    hr_data = calculate_hr_from_ppg(
                        result['current_peaks'],
                        sampling_rate,
                        len(result['clean']),
                        rate_method=st.session_state.ppg_params.get('rate_method', 'monotone_cubic')
                    )
                    result.update(hr_data)
                else:
                    result['hr_bpm'] = np.array([])
                    result['hr_interpolated'] = np.zeros(len(result['clean']))
                    result['mean_hr'] = 0.0
                    result['std_hr'] = 0.0

                # Initialize region range in session state if not exists (needed before plotting)
                if 'ppg_region_start' not in st.session_state:
                    st.session_state.ppg_region_start = 0.0
                if 'ppg_region_end' not in st.session_state:
                    st.session_state.ppg_region_end = min(10.0, float(time[-1]))

                # Get zoom range from session state
                ppg_zoom = (st.session_state.ppg_region_start, st.session_state.ppg_region_end)

                fig = create_signal_plot(
                    time, result['raw'], result['clean'],
                    result['current_peaks'], result['auto_peaks'],
                    'PPG', sampling_rate,
                    hr_interpolated=result.get('hr_interpolated'),
                    hr_bpm=result.get('hr_bpm'),
                    ui_revision='ppg_plot',  # Preserve zoom state
                    zoom_range=ppg_zoom  # Apply zoom from region inputs
                )

                st.plotly_chart(fig, use_container_width=True)

                # Drag-based editing interface
                st.subheader("Drag-Based Peak Editing")

                # Quick navigation buttons
                st.write("**Quick Range Selection:**")
                col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
                with col_btn1:
                    if st.button("⏮️ First 10s", key='ppg_first_10s'):
                        st.session_state.ppg_region_start = 0.0
                        st.session_state.ppg_region_end = min(10.0, float(time[-1]))
                        st.rerun()
                with col_btn2:
                    if st.button("◀️ Previous 10s", key='ppg_prev_10s'):
                        window = 10.0
                        new_start = max(0.0, st.session_state.ppg_region_start - window)
                        new_end = max(window, st.session_state.ppg_region_end - window)
                        st.session_state.ppg_region_start = new_start
                        st.session_state.ppg_region_end = min(new_end, float(time[-1]))
                        st.rerun()
                with col_btn3:
                    if st.button("▶️ Next 10s", key='ppg_next_10s'):
                        window = 10.0
                        new_start = min(float(time[-1]) - window, st.session_state.ppg_region_start + window)
                        new_end = min(float(time[-1]), st.session_state.ppg_region_end + window)
                        st.session_state.ppg_region_start = new_start
                        st.session_state.ppg_region_end = new_end
                        st.rerun()
                with col_btn4:
                    if st.button("⏭️ Last 10s", key='ppg_last_10s'):
                        st.session_state.ppg_region_start = max(0.0, float(time[-1]) - 10.0)
                        st.session_state.ppg_region_end = float(time[-1])
                        st.rerun()
                with col_btn5:
                    if st.button("🔄 Reset Range", key='ppg_reset_range'):
                        st.session_state.ppg_region_start = 0.0
                        st.session_state.ppg_region_end = min(10.0, float(time[-1]))
                        st.rerun()

                st.write("**Manual Range Entry:** (Or look at zoomed plot X-axis and enter values)")
                col1, col2 = st.columns(2)
                with col1:
                    region_start = st.number_input(
                        "Region Start (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.ppg_region_start),
                        step=1.0,
                        format="%.2f",
                        key='ppg_region_start_input',
                        help="Enter the start time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.ppg_region_start = region_start
                with col2:
                    region_end = st.number_input(
                        "Region End (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.ppg_region_end),
                        step=1.0,
                        format="%.2f",
                        key='ppg_region_end_input',
                        help="Enter the end time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.ppg_region_end = region_end

                # Show current range info
                st.caption(f"Current range: {region_start:.2f}s to {region_end:.2f}s ({region_end - region_start:.2f}s window) | Full signal: {float(time[-1]):.2f}s")

                st.write("**Choose Action:**")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("➕ Add Systolic Peaks in Region", type="primary", key='ppg_add_region_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.ppg_result['current_peaks'] = peak_editing.add_peaks_in_range(
                            result['clean'],
                            result['current_peaks'],
                            region_start,
                            region_end,
                            sampling_rate
                        )
                        st.rerun()

                with col2:
                    if st.button("➖ Remove Systolic Peaks in Region", type="secondary", key='ppg_remove_region_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.ppg_result['current_peaks'] = peak_editing.erase_peaks_in_range(
                            result['current_peaks'],
                            region_start,
                            region_end,
                            sampling_rate
                        )
                        st.rerun()

                with col3:
                    if st.button("🔄 Reset to Auto-Detected", key='ppg_reset_btn', use_container_width=True):
                        st.session_state.ppg_result['current_peaks'] = result['auto_peaks'].copy()
                        st.rerun()

                # Single peak editing
                with st.expander("✏️ Single Peak Editing (Advanced)"):
                    st.write("Add or remove individual peaks at specific times.")
                    col1, col2 = st.columns(2)
                    with col1:
                        single_peak_time = st.number_input(
                            "Time (seconds)",
                            min_value=0.0,
                            max_value=float(time[-1]),
                            value=0.0,
                            step=0.1,
                            key='ppg_single_peak_time'
                        )
                    with col2:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Add at Time", key='ppg_single_add_btn'):
                                from utils import peak_editing
                                st.session_state.ppg_result['current_peaks'] = peak_editing.add_peak(
                                    result['clean'],
                                    result['current_peaks'],
                                    single_peak_time,
                                    sampling_rate
                                )
                                st.rerun()
                        with col_b:
                            if st.button("Delete at Time", key='ppg_single_del_btn'):
                                from utils import peak_editing
                                st.session_state.ppg_result['current_peaks'] = peak_editing.delete_peak(
                                    result['current_peaks'],
                                    single_peak_time,
                                    sampling_rate
                                )
                                st.rerun()

                st.subheader("Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Peak Count", len(result['current_peaks']))
                with col2:
                    st.metric("Mean HR", f"{result['mean_hr']:.1f} bpm")
                with col3:
                    st.metric("HR Std Dev", f"{result['std_hr']:.1f} bpm")
                with col4:
                    st.metric("Mean Quality", f"{result['quality_mean']:.2f}")

        tab_idx += 1

    if 'bp' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("Blood Pressure Processing")

            col1, col2 = st.columns([2, 1])

            with col1:
                filter_method = st.selectbox("Filter Method", config.BP_FILTER_METHODS, key='bp_filter')
                with st.expander("ℹ️ Filter Info"):
                    st.info(config.BP_FILTER_INFO.get(filter_method, "No info available"))

                peak_method = st.selectbox("Peak Detection", config.BP_PEAK_METHODS, key='bp_peak')
                with st.expander("ℹ️ Peak Method Info"):
                    st.info(config.BP_PEAK_INFO.get(peak_method, "No info available"))

            with col2:
                detect_calib = st.checkbox("Detect Calibration Artifacts", value=True, key='bp_calib')
                if peak_method == 'prominence':
                    prominence = st.number_input("Prominence", min_value=1, max_value=100, value=10, key='bp_prom')
                else:
                    prominence = 10

            if st.button("Process Blood Pressure", type="primary"):
                signal = data['df'][data['signal_mappings']['bp']].values

                params = {
                    'filter_method': filter_method,
                    'peak_method': peak_method,
                    'prominence': prominence,
                    'detect_calibration': detect_calib
                }
                st.session_state.bp_params.update(params)

                result = blood_pressure.process_bp(signal, sampling_rate, st.session_state.bp_params)

                if result is None:
                    st.error("Processing failed: insufficient peaks detected")
                else:
                    st.session_state.bp_result = result
                    st.success("Blood pressure processed successfully")

            if st.session_state.bp_result is not None:
                result = st.session_state.bp_result

                st.subheader("Manual Blood Pressure Editing")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Auto Systolic Peaks", len(result['auto_peaks']))
                with col2:
                    st.metric("Auto Diastolic Troughs", len(result['auto_troughs']))
                with col3:
                    n_added_peaks = len(np.setdiff1d(result['current_peaks'], result['auto_peaks']))
                    st.metric("Added Systolic", n_added_peaks)
                with col4:
                    n_added_troughs = len(np.setdiff1d(result['current_troughs'], result['auto_troughs']))
                    st.metric("Added Diastolic", n_added_troughs)

                time = np.arange(len(result['filtered'])) / sampling_rate

                # Calculate continuous MAP values from the BP signal
                # MAP ≈ DBP + 1/3(SBP - DBP) = DBP + 1/3 * pulse pressure
                # For continuous signal, we can approximate MAP as: MAP = (2*DBP + SBP) / 3
                # Using a simple approximation: MAP ≈ signal * 2/3 (since waveform oscillates between SBP and DBP)
                # Better approach: Calculate MAP from the actual waveform
                from metrics.blood_pressure import calculate_bp_metrics
                current_bp_metrics = calculate_bp_metrics(
                    result['filtered'],
                    result['current_peaks'],
                    result['current_troughs']
                )
                # Create continuous MAP by interpolating between cardiac cycles
                if len(result['current_peaks']) > 0 and len(result['current_troughs']) > 0:
                    # Calculate MAP for each cardiac cycle: MAP = DBP + 1/3(SBP - DBP)
                    # Get min cycles
                    min_cycles = min(len(result['current_peaks']), len(result['current_troughs']))
                    if min_cycles > 0:
                        # Get peak and trough values
                        sbp_values = result['filtered'][result['current_peaks'][:min_cycles]]
                        dbp_values = result['filtered'][result['current_troughs'][:min_cycles]]
                        map_cycle = dbp_values + (sbp_values - dbp_values) / 3

                        # Create time points for interpolation (use trough times)
                        map_times = time[result['current_troughs'][:min_cycles]]

                        # Interpolate MAP across the entire signal
                        map_interpolated = np.interp(time, map_times, map_cycle)
                    else:
                        map_interpolated = np.full(len(time), current_bp_metrics.get('mean_mbp', 0))
                else:
                    map_interpolated = np.zeros(len(time))

                # Initialize region range in session state if not exists (needed before plotting)
                if 'bp_region_start' not in st.session_state:
                    st.session_state.bp_region_start = 0.0
                if 'bp_region_end' not in st.session_state:
                    st.session_state.bp_region_end = min(10.0, float(time[-1]))

                # Get zoom range from session state
                bp_zoom = (st.session_state.bp_region_start, st.session_state.bp_region_end)

                fig = create_rsp_bp_plot(
                    time, result['raw'], result['filtered'],
                    result['current_peaks'], result['current_troughs'],
                    result['auto_peaks'], result['auto_troughs'],
                    'BP',
                    map_values=map_interpolated,
                    ui_revision='bp_plot',  # Preserve zoom state
                    zoom_range=bp_zoom  # Apply zoom from region inputs
                )

                st.plotly_chart(fig, use_container_width=True)

                # Drag-based editing interface
                st.subheader("Drag-Based Blood Pressure Editing")

                # Quick navigation buttons
                st.write("**Quick Range Selection:**")
                col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
                with col_btn1:
                    if st.button("⏮️ First 10s", key='bp_first_10s'):
                        st.session_state.bp_region_start = 0.0
                        st.session_state.bp_region_end = min(10.0, float(time[-1]))
                        st.rerun()
                with col_btn2:
                    if st.button("◀️ Previous 10s", key='bp_prev_10s'):
                        window = 10.0
                        new_start = max(0.0, st.session_state.bp_region_start - window)
                        new_end = max(window, st.session_state.bp_region_end - window)
                        st.session_state.bp_region_start = new_start
                        st.session_state.bp_region_end = min(new_end, float(time[-1]))
                        st.rerun()
                with col_btn3:
                    if st.button("▶️ Next 10s", key='bp_next_10s'):
                        window = 10.0
                        new_start = min(float(time[-1]) - window, st.session_state.bp_region_start + window)
                        new_end = min(float(time[-1]), st.session_state.bp_region_end + window)
                        st.session_state.bp_region_start = new_start
                        st.session_state.bp_region_end = new_end
                        st.rerun()
                with col_btn4:
                    if st.button("⏭️ Last 10s", key='bp_last_10s'):
                        st.session_state.bp_region_start = max(0.0, float(time[-1]) - 10.0)
                        st.session_state.bp_region_end = float(time[-1])
                        st.rerun()
                with col_btn5:
                    if st.button("🔄 Reset Range", key='bp_reset_range'):
                        st.session_state.bp_region_start = 0.0
                        st.session_state.bp_region_end = min(10.0, float(time[-1]))
                        st.rerun()

                st.write("**Manual Range Entry:** (Or look at zoomed plot X-axis and enter values)")
                col1, col2 = st.columns(2)
                with col1:
                    region_start = st.number_input(
                        "Region Start (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.bp_region_start),
                        step=1.0,
                        format="%.2f",
                        key='bp_region_start_input',
                        help="Enter the start time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.bp_region_start = region_start
                with col2:
                    region_end = st.number_input(
                        "Region End (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.bp_region_end),
                        step=1.0,
                        format="%.2f",
                        key='bp_region_end_input',
                        help="Enter the end time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.bp_region_end = region_end

                # Show current range info
                st.caption(f"Current range: {region_start:.2f}s to {region_end:.2f}s ({region_end - region_start:.2f}s window) | Full signal: {float(time[-1]):.2f}s")

                st.write("**Systolic Peaks:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("➕ Add Systolic Peaks", type="primary", key='bp_add_peaks_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.bp_result['current_peaks'] = peak_editing.add_peaks_in_range(
                            result['filtered'], result['current_peaks'], region_start, region_end, sampling_rate, min_distance_seconds=0.5
                        )
                        st.rerun()
                with col2:
                    if st.button("➖ Remove Systolic Peaks", type="secondary", key='bp_remove_peaks_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.bp_result['current_peaks'] = peak_editing.erase_peaks_in_range(
                            result['current_peaks'], region_start, region_end, sampling_rate
                        )
                        st.rerun()
                with col3:
                    if st.button("🔄 Reset Systolic", key='bp_reset_peaks_btn', use_container_width=True):
                        st.session_state.bp_result['current_peaks'] = result['auto_peaks'].copy()
                        st.rerun()

                st.write("**Diastolic Troughs:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("➕ Add Diastolic Troughs", type="primary", key='bp_add_troughs_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.bp_result['current_troughs'] = peak_editing.add_troughs_in_range(
                            result['filtered'], result['current_troughs'], region_start, region_end, sampling_rate, min_distance_seconds=0.5
                        )
                        st.rerun()
                with col2:
                    if st.button("➖ Remove Diastolic Troughs", type="secondary", key='bp_remove_troughs_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.bp_result['current_troughs'] = peak_editing.erase_troughs_in_range(
                            result['current_troughs'], region_start, region_end, sampling_rate
                        )
                        st.rerun()
                with col3:
                    if st.button("🔄 Reset Diastolic", key='bp_reset_troughs_btn', use_container_width=True):
                        st.session_state.bp_result['current_troughs'] = result['auto_troughs'].copy()
                        st.rerun()

                # Single peak/trough editing
                with st.expander("✏️ Single Peak/Trough Editing (Advanced)"):
                    st.write("Add or remove individual systolic/diastolic events at specific times.")
                    col1, col2 = st.columns(2)
                    with col1:
                        single_time = st.number_input("Time (seconds)", min_value=0.0, max_value=float(time[-1]), value=0.0, step=0.1, key='bp_single_time')

                    st.write("**Systolic Peaks:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Add Systolic at Time", key='bp_single_add_peak'):
                            from utils import peak_editing
                            st.session_state.bp_result['current_peaks'] = peak_editing.add_peak(
                                result['filtered'], result['current_peaks'], single_time, sampling_rate
                            )
                            st.rerun()
                    with col_b:
                        if st.button("Delete Systolic at Time", key='bp_single_del_peak'):
                            from utils import peak_editing
                            st.session_state.bp_result['current_peaks'] = peak_editing.delete_peak(
                                result['current_peaks'], single_time, sampling_rate
                            )
                            st.rerun()

                    st.write("**Diastolic Troughs:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Add Diastolic at Time", key='bp_single_add_trough'):
                            from utils import peak_editing
                            st.session_state.bp_result['current_troughs'] = peak_editing.add_trough(
                                result['filtered'], result['current_troughs'], single_time, sampling_rate
                            )
                            st.rerun()
                    with col_b:
                        if st.button("Delete Diastolic at Time", key='bp_single_del_trough'):
                            from utils import peak_editing
                            st.session_state.bp_result['current_troughs'] = peak_editing.delete_trough(
                                result['current_troughs'], single_time, sampling_rate
                            )
                            st.rerun()

                st.subheader("Statistics")

                # Recalculate BP metrics based on current peaks/troughs
                from metrics.blood_pressure import calculate_bp_metrics
                current_bp_metrics = calculate_bp_metrics(result['filtered'], result['current_peaks'], result['current_troughs'])

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Cardiac Cycles", min(len(result['current_peaks']), len(result['current_troughs'])))
                with col2:
                    st.metric("Mean SBP", f"{current_bp_metrics['mean_sbp']:.1f} mmHg")
                with col3:
                    st.metric("Mean DBP", f"{current_bp_metrics['mean_dbp']:.1f} mmHg")
                with col4:
                    st.metric("Mean MAP", f"{current_bp_metrics['mean_mbp']:.1f} mmHg")

        tab_idx += 1

    with tab_objects[tab_idx]:
        st.header("Export Data")

        output_path = st.text_input(
            "Output Path",
            value=config.OUTPUT_BASE_PATH,
            help="Base output directory. Files will be saved to {output_path}/{participant}/{session}/"
        )

        processed_signals = []
        if st.session_state.ecg_result is not None:
            processed_signals.append("ECG")
        if st.session_state.rsp_result is not None:
            processed_signals.append("RSP")
        if st.session_state.ppg_result is not None:
            processed_signals.append("PPG")
        if st.session_state.bp_result is not None:
            processed_signals.append("Blood Pressure")

        if len(processed_signals) == 0:
            st.warning("No signals have been processed yet. Process at least one signal before exporting.")
        else:
            st.success(f"Processed signals ready for export: {', '.join(processed_signals)}")

            if st.button("Export All Signals", type="primary"):
                results_dict = {}
                params_dict = {}

                if st.session_state.ecg_result is not None:
                    # Recalculate HR based on current peaks before export
                    from metrics.ecg import calculate_hr
                    ecg_result = st.session_state.ecg_result.copy()
                    updated_hr = calculate_hr(
                        ecg_result['current_r_peaks'],
                        sampling_rate,
                        len(ecg_result['clean']),
                        rate_method=st.session_state.ecg_params.get('rate_method', 'monotone_cubic')
                    )
                    ecg_result.update(updated_hr)
                    results_dict['ecg'] = ecg_result
                    params_dict['ecg'] = st.session_state.ecg_params

                if st.session_state.rsp_result is not None:
                    # Recalculate BR based on current troughs before export
                    from metrics.rsp import calculate_breathing_rate
                    rsp_result = st.session_state.rsp_result.copy()
                    updated_br = calculate_breathing_rate(
                        rsp_result['current_troughs'],
                        sampling_rate,
                        len(rsp_result['clean']),
                        rate_method=st.session_state.rsp_params.get('rate_method', 'monotone_cubic')
                    )
                    rsp_result.update(updated_br)
                    results_dict['rsp'] = rsp_result
                    params_dict['rsp'] = st.session_state.rsp_params

                if st.session_state.ppg_result is not None:
                    # Recalculate HR based on current peaks before export
                    from metrics.ppg import calculate_hr_from_ppg
                    ppg_result = st.session_state.ppg_result.copy()
                    updated_hr = calculate_hr_from_ppg(
                        ppg_result['current_peaks'],
                        sampling_rate,
                        len(ppg_result['clean']),
                        rate_method=st.session_state.ppg_params.get('rate_method', 'monotone_cubic')
                    )
                    ppg_result.update(updated_hr)
                    results_dict['ppg'] = ppg_result
                    params_dict['ppg'] = st.session_state.ppg_params

                if st.session_state.bp_result is not None:
                    # Recalculate BP metrics based on current peaks/troughs before export
                    from metrics.blood_pressure import calculate_bp_metrics
                    bp_result = st.session_state.bp_result.copy()
                    updated_metrics = calculate_bp_metrics(
                        bp_result['filtered'],
                        bp_result['current_peaks'],
                        bp_result['current_troughs']
                    )
                    bp_result.update(updated_metrics)
                    results_dict['bp'] = bp_result
                    params_dict['bp'] = st.session_state.bp_params

                df = export.create_combined_dataframe(results_dict, sampling_rate)
                metadata = export.create_metadata_json(results_dict, params_dict, sampling_rate)

                paths = export.export_physio_data(
                    output_path,
                    st.session_state.participant,
                    st.session_state.session,
                    st.session_state.task,
                    df,
                    metadata
                )

                st.success("Export complete!")
                st.info(f"""
                **CSV**: `{paths['csv_path']}`
                **JSON**: `{paths['json_path']}`
                """)

                with st.expander("Preview CSV (first 10 rows)"):
                    st.dataframe(df.head(10))

                with st.expander("Preview JSON Metadata"):
                    st.json(metadata)


if __name__ == "__main__":
    main()
