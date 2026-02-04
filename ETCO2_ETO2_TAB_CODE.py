"""
ETCO2 and ETO2 Tab Implementation Code
Insert this code into app.py at line 1568 (right before the Export tab)
"""

# ============================================================================
# ETCO2 TAB
# ============================================================================

if 'etco2' in detected_signals:
    with tab_objects[tab_idx]:
        st.header("End-Tidal CO2 (ETCO2) Processing")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Peak detection method selection
            peak_method = st.selectbox(
                "Peak Detection Method",
                config.ETCO2_PEAK_METHODS,
                key='etco2_peak_method'
            )

            with st.expander("ℹ️ Method Info"):
                st.info(config.ETCO2_PEAK_METHOD_INFO.get(peak_method, "No info available"))

            # Main parameters
            st.subheader("Detection Parameters")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                min_peak_distance_s = st.slider(
                    "Min Peak Distance (s)",
                    min_value=0.5,
                    max_value=5.0,
                    value=st.session_state.etco2_params.get('min_peak_distance_s', 2.0),
                    step=0.1,
                    key='etco2_min_peak_distance',
                    help="Minimum time between consecutive breath peaks (prevents double-detection)"
                )

            with col_p2:
                min_prominence = st.slider(
                    "Min Prominence (mmHg)",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state.etco2_params.get('min_prominence', 1.0),
                    step=0.1,
                    key='etco2_min_prominence',
                    help="Minimum peak prominence for valid detection"
                )

            smooth_peaks = st.slider(
                "Smoothing Kernel Size",
                min_value=3,
                max_value=15,
                value=st.session_state.etco2_params.get('smooth_peaks', 5),
                step=2,
                key='etco2_smooth_peaks',
                help="Median filter kernel size (number of peaks). Must be odd."
            )

            # Advanced parameters (Savitzky-Golay filter)
            with st.expander("⚙️ Advanced: Savitzky-Golay Filter"):
                st.markdown("**For derivative-based peak detection**")

                col_sg1, col_sg2 = st.columns(2)
                with col_sg1:
                    sg_window_s = st.slider(
                        "Window Duration (s)",
                        min_value=0.1,
                        max_value=1.0,
                        value=st.session_state.etco2_params.get('sg_window_s', 0.3),
                        step=0.05,
                        key='etco2_sg_window',
                        help="Smoothing window for computing derivatives"
                    )

                with col_sg2:
                    sg_poly = st.slider(
                        "Polynomial Order",
                        min_value=1,
                        max_value=5,
                        value=st.session_state.etco2_params.get('sg_poly', 2),
                        key='etco2_sg_poly',
                        help="Polynomial order for S-G filter (2=quadratic)"
                    )

                prom_adapt = st.checkbox(
                    "Adaptive Prominence Threshold",
                    value=st.session_state.etco2_params.get('prom_adapt', False),
                    key='etco2_prom_adapt',
                    help="Use 25th percentile of detected prominences as adaptive minimum"
                )

        with col2:
            if st.button("🔬 Process ETCO2", type="primary", key='process_etco2'):
                # Update parameters
                params = {
                    'peak_method': peak_method,
                    'min_peak_distance_s': min_peak_distance_s,
                    'min_prominence': min_prominence,
                    'smooth_peaks': smooth_peaks,
                    'sg_window_s': sg_window_s,
                    'sg_poly': sg_poly,
                    'prom_adapt': prom_adapt
                }
                st.session_state.etco2_params.update(params)

                # Get CO2 signal
                co2_signal = data['df'][data['signal_mappings']['etco2']].values

                # Process
                with st.spinner("Detecting CO2 peaks and extracting envelope..."):
                    result = etco2.extract_etco2_envelope(
                        co2_signal,
                        sampling_rate,
                        st.session_state.etco2_params
                    )

                if result is not None:
                    st.session_state.etco2_result = result
                    st.success(f"✅ ETCO2 processed: {len(result['auto_peaks'])} peaks detected")
                    st.rerun()

        # Display results if available
        result = st.session_state.etco2_result
        if result is not None:
            st.divider()

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Auto-detected Peaks", len(result['auto_peaks']))
            with col2:
                n_added = len(np.setdiff1d(result['current_peaks'], result['auto_peaks']))
                st.metric("Manually Added", n_added, delta=f"+{n_added}" if n_added > 0 else None)
            with col3:
                n_deleted = len(np.setdiff1d(result['auto_peaks'], result['current_peaks']))
                st.metric("Deleted", n_deleted, delta=f"-{n_deleted}" if n_deleted > 0 else None)
            with col4:
                if len(result['etco2_envelope']) > 0:
                    mean_etco2 = np.mean(result['etco2_envelope'][np.isfinite(result['etco2_envelope'])])
                    st.metric("Mean ETCO2", f"{mean_etco2:.1f} mmHg")

            # Visualization
            st.subheader("📊 ETCO2 Trace Visualization")

            # Create plotly figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Raw CO2 Signal with Detected Peaks', 'ETCO2 Upper Envelope'),
                vertical_spacing=0.12,
                row_heights=[0.55, 0.45]
            )

            time = result['time_vector']

            # Row 1: Raw signal with peaks
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=result['raw_signal'],
                    name='Raw CO2',
                    line=dict(color='#636EFA', width=1),
                    mode='lines'
                ),
                row=1, col=1
            )

            # Auto-detected peaks
            if len(result['auto_peaks']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time[result['auto_peaks']],
                        y=result['raw_signal'][result['auto_peaks']],
                        name='Auto Peaks',
                        mode='markers',
                        marker=dict(color='#00CC96', size=8, symbol='circle')
                    ),
                    row=1, col=1
                )

            # Manually added peaks
            added_peaks = np.setdiff1d(result['current_peaks'], result['auto_peaks'])
            if len(added_peaks) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time[added_peaks],
                        y=result['raw_signal'][added_peaks],
                        name='Added Peaks',
                        mode='markers',
                        marker=dict(color='#AB63FA', size=10, symbol='x', line=dict(width=2))
                    ),
                    row=1, col=1
                )

            # Deleted peaks
            deleted_peaks = np.setdiff1d(result['auto_peaks'], result['current_peaks'])
            if len(deleted_peaks) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time[deleted_peaks],
                        y=result['raw_signal'][deleted_peaks],
                        name='Deleted Peaks',
                        mode='markers',
                        marker=dict(color='#FF4444', size=8, symbol='x')
                    ),
                    row=1, col=1
                )

            # Row 2: ETCO2 envelope
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=result['etco2_envelope'],
                    name='ETCO2 Envelope',
                    line=dict(color='#EF553B', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(239, 85, 59, 0.2)',
                    mode='lines'
                ),
                row=2, col=1
            )

            # Layout
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="CO2 (mmHg)", row=1, col=1)
            fig.update_yaxes(title_text="ETCO2 (mmHg)", row=2, col=1)

            fig.update_layout(
                height=800,
                template='plotly_dark',
                showlegend=True,
                hovermode='x unified'
            )

            # Apply zoom if set
            if st.session_state.etco2_zoom_range is not None:
                fig.update_xaxes(range=st.session_state.etco2_zoom_range)

            st.plotly_chart(fig, use_container_width=True, key='etco2_plot')

            # Manual editing interface
            with st.expander("✏️ Manual Peak Editing"):
                st.info("Add or remove peaks by specifying time ranges. Click 'Reset' to restore auto-detected peaks.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    region_start = st.number_input(
                        "Region Start (s)",
                        min_value=0.0,
                        max_value=float(result['time_vector'][-1]),
                        value=0.0,
                        step=1.0,
                        key='etco2_region_start'
                    )
                with col2:
                    region_end = st.number_input(
                        "Region End (s)",
                        min_value=0.0,
                        max_value=float(result['time_vector'][-1]),
                        value=min(10.0, float(result['time_vector'][-1])),
                        step=1.0,
                        key='etco2_region_end'
                    )
                with col3:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    if st.button("🔍 Zoom to Region", key='etco2_zoom'):
                        st.session_state.etco2_zoom_range = [region_start, region_end]
                        st.rerun()

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("➕ Add Peaks in Region", key='etco2_add'):
                        result['current_peaks'] = peak_editing.add_peaks_in_range(
                            result['current_peaks'],
                            result['raw_signal'],
                            sampling_rate,
                            region_start,
                            region_end
                        )
                        st.rerun()

                with col2:
                    if st.button("➖ Remove Peaks in Region", key='etco2_remove'):
                        result['current_peaks'] = peak_editing.erase_peaks_in_range(
                            result['current_peaks'],
                            sampling_rate,
                            region_start,
                            region_end
                        )
                        st.rerun()

                with col3:
                    if st.button("🔄 Reset All Peaks", key='etco2_reset'):
                        result['current_peaks'] = result['auto_peaks'].copy()
                        st.success("Peaks reset to auto-detected")
                        st.rerun()

                # Reset zoom button
                if st.session_state.etco2_zoom_range is not None:
                    if st.button("↔️ Reset Zoom", key='etco2_reset_zoom'):
                        st.session_state.etco2_zoom_range = None
                        st.rerun()

        else:
            st.info("👆 Configure parameters above and click 'Process ETCO2' to begin")

    tab_idx += 1


# ============================================================================
# ETO2 TAB
# ============================================================================

if 'eto2' in detected_signals:
    with tab_objects[tab_idx]:
        st.header("End-Tidal O2 (ETO2) Processing")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Trough detection method selection
            trough_method = st.selectbox(
                "Trough Detection Method",
                config.ETO2_TROUGH_METHODS,
                key='eto2_trough_method'
            )

            with st.expander("ℹ️ Method Info"):
                st.info(config.ETO2_TROUGH_METHOD_INFO.get(trough_method, "No info available"))

            # Main parameters
            st.subheader("Detection Parameters")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                min_trough_distance_s = st.slider(
                    "Min Trough Distance (s)",
                    min_value=0.5,
                    max_value=6.0,
                    value=st.session_state.eto2_params.get('min_trough_distance_s', 3.0),
                    step=0.1,
                    key='eto2_min_trough_distance',
                    help="Minimum time between consecutive breath troughs (prevents double-detection)"
                )

            with col_p2:
                min_prominence = st.slider(
                    "Min Prominence (mmHg)",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state.eto2_params.get('min_prominence', 1.0),
                    step=0.1,
                    key='eto2_min_prominence',
                    help="Minimum trough prominence for valid detection (on inverted signal)"
                )

            smooth_troughs = st.slider(
                "Smoothing Kernel Size",
                min_value=3,
                max_value=15,
                value=st.session_state.eto2_params.get('smooth_troughs', 5),
                step=2,
                key='eto2_smooth_troughs',
                help="Median filter kernel size (number of troughs). Must be odd."
            )

            # Advanced parameters (Savitzky-Golay filter)
            with st.expander("⚙️ Advanced: Savitzky-Golay Filter"):
                st.markdown("**For derivative-based trough detection**")

                col_sg1, col_sg2 = st.columns(2)
                with col_sg1:
                    sg_window_s = st.slider(
                        "Window Duration (s)",
                        min_value=0.1,
                        max_value=1.0,
                        value=st.session_state.eto2_params.get('sg_window_s', 0.2),
                        step=0.05,
                        key='eto2_sg_window',
                        help="Smoothing window for computing derivatives"
                    )

                with col_sg2:
                    sg_poly = st.slider(
                        "Polynomial Order",
                        min_value=1,
                        max_value=5,
                        value=st.session_state.eto2_params.get('sg_poly', 2),
                        key='eto2_sg_poly',
                        help="Polynomial order for S-G filter (2=quadratic)"
                    )

                prom_adapt = st.checkbox(
                    "Adaptive Prominence Threshold",
                    value=st.session_state.eto2_params.get('prom_adapt', False),
                    key='eto2_prom_adapt',
                    help="Use 25th percentile of detected prominences as adaptive minimum"
                )

        with col2:
            if st.button("🔬 Process ETO2", type="primary", key='process_eto2'):
                # Update parameters
                params = {
                    'trough_method': trough_method,
                    'min_trough_distance_s': min_trough_distance_s,
                    'min_prominence': min_prominence,
                    'smooth_troughs': smooth_troughs,
                    'sg_window_s': sg_window_s,
                    'sg_poly': sg_poly,
                    'prom_adapt': prom_adapt
                }
                st.session_state.eto2_params.update(params)

                # Get O2 signal
                o2_signal = data['df'][data['signal_mappings']['eto2']].values

                # Process
                with st.spinner("Detecting O2 troughs and extracting envelope..."):
                    result = eto2.extract_eto2_envelope(
                        o2_signal,
                        sampling_rate,
                        st.session_state.eto2_params
                    )

                if result is not None:
                    st.session_state.eto2_result = result
                    st.success(f"✅ ETO2 processed: {len(result['auto_troughs'])} troughs detected")
                    st.rerun()

        # Display results if available
        result = st.session_state.eto2_result
        if result is not None:
            st.divider()

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Auto-detected Troughs", len(result['auto_troughs']))
            with col2:
                n_added = len(np.setdiff1d(result['current_troughs'], result['auto_troughs']))
                st.metric("Manually Added", n_added, delta=f"+{n_added}" if n_added > 0 else None)
            with col3:
                n_deleted = len(np.setdiff1d(result['auto_troughs'], result['current_troughs']))
                st.metric("Deleted", n_deleted, delta=f"-{n_deleted}" if n_deleted > 0 else None)
            with col4:
                if len(result['eto2_envelope']) > 0:
                    mean_eto2 = np.mean(result['eto2_envelope'][np.isfinite(result['eto2_envelope'])])
                    st.metric("Mean ETO2", f"{mean_eto2:.1f} mmHg")

            # Visualization
            st.subheader("📊 ETO2 Trace Visualization")

            # Create plotly figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Raw O2 Signal with Detected Troughs', 'ETO2 Lower Envelope'),
                vertical_spacing=0.12,
                row_heights=[0.55, 0.45]
            )

            time = result['time_vector']

            # Row 1: Raw signal with troughs
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=result['raw_signal'],
                    name='Raw O2',
                    line=dict(color='#FFA15A', width=1),
                    mode='lines'
                ),
                row=1, col=1
            )

            # Auto-detected troughs
            if len(result['auto_troughs']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time[result['auto_troughs']],
                        y=result['raw_signal'][result['auto_troughs']],
                        name='Auto Troughs',
                        mode='markers',
                        marker=dict(color='#00CC96', size=8, symbol='circle')
                    ),
                    row=1, col=1
                )

            # Manually added troughs
            added_troughs = np.setdiff1d(result['current_troughs'], result['auto_troughs'])
            if len(added_troughs) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time[added_troughs],
                        y=result['raw_signal'][added_troughs],
                        name='Added Troughs',
                        mode='markers',
                        marker=dict(color='#AB63FA', size=10, symbol='x', line=dict(width=2))
                    ),
                    row=1, col=1
                )

            # Deleted troughs
            deleted_troughs = np.setdiff1d(result['auto_troughs'], result['current_troughs'])
            if len(deleted_troughs) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time[deleted_troughs],
                        y=result['raw_signal'][deleted_troughs],
                        name='Deleted Troughs',
                        mode='markers',
                        marker=dict(color='#FF4444', size=8, symbol='x')
                    ),
                    row=1, col=1
                )

            # Row 2: ETO2 envelope
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=result['eto2_envelope'],
                    name='ETO2 Envelope',
                    line=dict(color='#19D3F3', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(25, 211, 243, 0.2)',
                    mode='lines'
                ),
                row=2, col=1
            )

            # Layout
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="O2 (mmHg)", row=1, col=1)
            fig.update_yaxes(title_text="ETO2 (mmHg)", row=2, col=1)

            fig.update_layout(
                height=800,
                template='plotly_dark',
                showlegend=True,
                hovermode='x unified'
            )

            # Apply zoom if set
            if st.session_state.eto2_zoom_range is not None:
                fig.update_xaxes(range=st.session_state.eto2_zoom_range)

            st.plotly_chart(fig, use_container_width=True, key='eto2_plot')

            # Manual editing interface
            with st.expander("✏️ Manual Trough Editing"):
                st.info("Add or remove troughs by specifying time ranges. Click 'Reset' to restore auto-detected troughs.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    region_start = st.number_input(
                        "Region Start (s)",
                        min_value=0.0,
                        max_value=float(result['time_vector'][-1]),
                        value=0.0,
                        step=1.0,
                        key='eto2_region_start'
                    )
                with col2:
                    region_end = st.number_input(
                        "Region End (s)",
                        min_value=0.0,
                        max_value=float(result['time_vector'][-1]),
                        value=min(10.0, float(result['time_vector'][-1])),
                        step=1.0,
                        key='eto2_region_end'
                    )
                with col3:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    if st.button("🔍 Zoom to Region", key='eto2_zoom'):
                        st.session_state.eto2_zoom_range = [region_start, region_end]
                        st.rerun()

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("➕ Add Troughs in Region", key='eto2_add'):
                        # For troughs, we want minima not maxima
                        result['current_troughs'] = peak_editing.add_troughs_in_range(
                            result['current_troughs'],
                            result['raw_signal'],
                            sampling_rate,
                            region_start,
                            region_end
                        )
                        st.rerun()

                with col2:
                    if st.button("➖ Remove Troughs in Region", key='eto2_remove'):
                        result['current_troughs'] = peak_editing.erase_peaks_in_range(
                            result['current_troughs'],
                            sampling_rate,
                            region_start,
                            region_end
                        )
                        st.rerun()

                with col3:
                    if st.button("🔄 Reset All Troughs", key='eto2_reset'):
                        result['current_troughs'] = result['auto_troughs'].copy()
                        st.success("Troughs reset to auto-detected")
                        st.rerun()

                # Reset zoom button
                if st.session_state.eto2_zoom_range is not None:
                    if st.button("↔️ Reset Zoom", key='eto2_reset_zoom'):
                        st.session_state.eto2_zoom_range = None
                        st.rerun()

        else:
            st.info("👆 Configure parameters above and click 'Process ETO2' to begin")

    tab_idx += 1
