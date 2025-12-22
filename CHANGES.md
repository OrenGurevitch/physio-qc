# Recent Changes Summary

## Completed Implementations

### 1. ECG Continuous Quality Metrics ✅
**File**: `metrics/ecg.py`

Added three quality metrics as requested:
- `quality_zhao`: Categorical classification (Excellent/Barely acceptable/Unacceptable)
- `quality_templatematch`: Template correlation (0-1, higher = better shape match)
- `quality_averageqrs`: Distance from average beat (0-1, higher = closer to average)

The quality metrics are calculated when "Calculate Quality" checkbox is enabled.

### 2. Custom Filter UI for ECG ✅
**File**: `app.py` (lines 365-388)

When "custom" cleaning method is selected, shows:
- Filter Type (butterworth, fir, cheby1, cheby2, elliptic, bessel)
- Filter Mode (Bandpass, Lowpass, Highpass)
- High-pass frequency (if Highpass or Bandpass)
- Low-pass frequency (if Lowpass or Bandpass)
- Filter Order (1-10)

### 3. Zoom Sync Helper for Region Editing ✅
**File**: `app.py` (line 460)

Added clear instructions and improved UX:
- Info box: "🔍 How to use: Zoom the plot above → note the X-axis time range → enter that range below → click an action button"
- Tooltips on time inputs explaining to look at X-axis
- "Show Full Range" button to display total signal duration
- Increased precision (step=0.1, format="%.2f")

### 4. BP Calibration Detection Already Integrated ✅
**File**: `algorithms/quality_detection.py`

Functions already implemented:
- `detect_calibration_artifacts()`: Uses derivative analysis to find flat calibration periods
- `detect_high_derivative_regions()`: Finds noise/movement artifacts
- `filter_indices_outside_regions()`: Removes peaks in bad regions

BP processing already uses these functions in `metrics/blood_pressure.py`

## To-Do: Apply Same Pattern to Other Signals

### RSP Custom Filters
Add to `app.py` around line 504 (RSP section):

```python
if method == 'custom':
    st.subheader("Custom Filter Options")
    filter_type = st.selectbox("Filter Type", config.FILTER_TYPES, key='rsp_filter_type')
    filter_mode = st.radio("Filter Mode", ["Bandpass", "Lowpass", "Highpass"], horizontal=True, key='rsp_filter_mode')

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if filter_mode in ["Highpass", "Bandpass"]:
            lowcut = st.number_input("High-pass (Hz)", min_value=0.01, max_value=10.0, value=0.05, step=0.01, key='rsp_lowcut')
        else:
            lowcut = None
    with col_f2:
        if filter_mode in ["Lowpass", "Bandpass"]:
            highcut = st.number_input("Low-pass (Hz)", min_value=0.1, max_value=20.0, value=3.0, step=0.1, key='rsp_highcut')
        else:
            highcut = None

    filter_order = st.slider("Filter Order", min_value=1, max_value=10, value=5, key='rsp_filter_order')
else:
    filter_type = 'butterworth'
    lowcut = 0.05
    highcut = 3.0
    filter_order = 5
```

Then update params dict to include filter options.

### PPG Custom Filters
Add to `app.py` around line 604 (PPG section) - similar pattern as RSP.

### BP Custom Filters
Add to `app.py` around line 700 (BP section):

```python
if filter_method == 'custom':
    st.subheader("Custom Filter Options")
    filter_type = st.selectbox("Filter Type", config.FILTER_TYPES, key='bp_filter_type')
    filter_mode = st.radio("Filter Mode", ["Bandpass", "Lowpass", "Highpass"], horizontal=True, key='bp_filter_mode')

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if filter_mode in ["Highpass", "Bandpass"]:
            lowcut = st.number_input("High-pass (Hz)", min_value=0.01, max_value=50.0, value=0.5, step=0.1, key='bp_lowcut')
        else:
            lowcut = None
    with col_f2:
        if filter_mode in ["Lowpass", "Bandpass"]:
            highcut = st.number_input("Low-pass (Hz)", min_value=1.0, max_value=100.0, value=15.0, step=0.5, key='bp_highcut')
        else:
            highcut = None

    filter_order = st.slider("Filter Order", min_value=1, max_value=10, value=3, key='bp_filter_order')
else:
    filter_type = 'butterworth'
    lowcut = 0.5
    highcut = 15.0
    filter_order = 3
```

### Zoom Helpers for All Signals

**RSP** (around line 584):
```python
st.info("🔍 **How to use**: Zoom the plot above → note the X-axis time range → enter that range below → click an action button")
# Update number_input widgets with step=0.1, format="%.2f", help text
```

**PPG** (around line 764):
Same pattern as RSP.

**BP** (around line 938):
Same pattern as RSP.

## Testing Checklist

### ECG
- [x] Custom filter UI appears when "custom" selected
- [x] Filter parameters are passed to processing
- [x] Quality metrics calculated correctly
- [x] Zoom helper text visible
- [x] Region editing works with new precision

### RSP
- [ ] Custom filter UI
- [ ] Zoom helper text
- [ ] Test region editing

### PPG
- [ ] Custom filter UI
- [ ] Zoom helper text
- [ ] Test region editing

### BP
- [ ] Custom filter UI
- [ ] Zoom helper text
- [ ] Test region editing
- [ ] Calibration detection working

## Notes

1. All filter type constants are already defined in `config.py` (`FILTER_TYPES`)
2. The backend processing functions already support custom filtering
3. The changes are backward compatible - default values match previous behavior
4. ECG quality is now calculated by default (checkbox value=True)
