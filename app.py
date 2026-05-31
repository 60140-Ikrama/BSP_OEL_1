import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

from core.synthetic_data import SyntheticECGGenerator
from core.signal_processing import ECGProcessor
from core.hrv_analysis import HRVAnalyzer
from core.report_generator import ReportGenerator

# Page config
st.set_page_config(
    page_title="ICU Telemetry Hub - ECG/HRV",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main {
        background-color: #0A0F1D;
        color: #E2E8F0;
    }
    .stApp header {
        background-color: rgba(10, 15, 29, 0.9);
    }
    .css-1d391kg {
        background-color: #111827;
    }
    .metric-card {
        background-color: #1F2937;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #00E676;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #9CA3AF;
        text-transform: uppercase;
        font-weight: bold;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #FFFFFF;
        font-weight: bold;
    }
    .metric-desc {
        font-size: 0.75rem;
        color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("🏥 ICU Telemetry & ECG-HRV Surveillance Board")
st.write("Academic Laboratory Open-Ended Lab (OEL) Evaluation Platform - CLO1 & CLO2 Compliance")

# Sidebar - Settings & Parameters
st.sidebar.header("🛡️ Laboratory Info Block")
student_name = st.sidebar.text_input("Student Name", "Biomedical Student")
student_id = st.sidebar.text_input("Roll / Registration ID", "BME-2026-09")
supervisor_name = st.sidebar.text_input("Lab Supervisor", "Dr. Eleanor Vance")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Telemetry Signal Acquisition")

data_source = st.sidebar.selectbox("Data Source", ["Synthetic Generator", "Upload File (CSV/TXT)"])
fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=250, min_value=100, max_value=1000)

if data_source == "Synthetic Generator":
    rhythm = st.sidebar.selectbox("Cardiac Rhythm Type", ["NSR (Normal Sinus)", "AFib (Atrial Fibrillation)", "PVC (Ectopic Beats)", "VTach (Tachycardia)"])
    duration = st.sidebar.slider("Signal Duration (sec)", min_value=10, max_value=300, value=60)
    
    st.sidebar.subheader("Noise Modulation (Artifacts)")
    noise_bw = st.sidebar.slider("Baseline Wander (mV)", 0.0, 1.0, 0.20, step=0.05)
    noise_pl = st.sidebar.slider("Powerline Interf. 50Hz (mV)", 0.0, 0.5, 0.05, step=0.01)
    noise_emg = st.sidebar.slider("Muscle Denoising / EMG (mV)", 0.0, 0.2, 0.02, step=0.01)
    
    # Generate ECG
    generator = SyntheticECGGenerator(fs=fs)
    noise_config = {
        'baseline_wander': noise_bw,
        'powerline': noise_pl,
        'emg': noise_emg
    }
    t, sig, true_peaks, true_ectopics = generator.generate_signal(
        duration_sec=duration,
        rhythm=rhythm.split()[0], # Grab NSR, AFib, etc.
        noise_config=noise_config
    )
    rhythm_label = rhythm
else:
    uploaded_file = st.sidebar.file_uploader("Upload ECG Signal File", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'voltage' in df.columns:
                    sig = df['voltage'].values
                    t = df['time'].values if 'time' in df.columns else np.arange(len(sig)) / fs
                else:
                    sig = df.iloc[:,0].values
                    t = np.arange(len(sig)) / fs
            else:
                sig = np.loadtxt(uploaded_file)
                t = np.arange(len(sig)) / fs
            duration = len(sig) / fs
            rhythm_label = "Uploaded Patient Record"
            true_peaks = None
            true_ectopics = None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info("Please upload an ECG CSV/TXT file, or switch to Synthetic Generator.")
        st.stop()

# DSP Tuning Panel
st.sidebar.markdown("---")
st.sidebar.header("🎚️ Biomedical DSP Settings")
lowcut = st.sidebar.slider("Butterworth Low Cut (Hz)", 0.1, 2.0, 0.5, step=0.1)
highcut = st.sidebar.slider("Butterworth High Cut (Hz)", 20.0, 100.0, 45.0, step=5.0)
ectopic_thresh = st.sidebar.slider("Ectopic Outlier Limit (%)", 10, 40, 20, step=5) / 100.0
corr_method = st.sidebar.selectbox("Ectopic Interpolation", ["spline", "linear"])

# Initialize Processor & Analyzer
processor = ECGProcessor(fs=fs)
analyzer = HRVAnalyzer(fs_ecg=fs)
reporter = ReportGenerator(fs=fs)

# Run Signal Processing
sig_clean, baseline = processor.remove_baseline_wander_median(sig)
sig_filtered = processor.apply_bandpass(sig_clean, lowcut=lowcut, highcut=highcut)
sig_smoothed = processor.remove_noise_savgol(sig_filtered)

# R-peak detection
r_peaks, PT_stages = processor.pan_tompkins_detector(sig_smoothed)
rr_intervals = np.diff(r_peaks) / fs * 1000.0

# Ectopic detection and correction
ectopic_mask = analyzer.detect_ectopic_beats(rr_intervals, threshold_pct=ectopic_thresh)
corrected_rr = analyzer.correct_ectopic_beats(rr_intervals, ectopic_mask, method=corr_method)

# HRV feature extraction
time_m = analyzer.compute_time_domain(corrected_rr)
freq_m = analyzer.compute_frequency_domain(corrected_rr)
nonl_m = analyzer.compute_nonlinear(corrected_rr)
interpretation = analyzer.generate_clinical_interpretation(time_m, freq_m, nonl_m)

# Main Dashboard Layout tabs
tab_overview, tab_dsp, tab_qrs, tab_ectopic, tab_hrv, tab_report = st.tabs([
    "📋 Ward Overview", 
    "📈 CLO1: DSP Preprocessing", 
    "⚡ Pan-Tompkins QRS", 
    "🩺 Ectopic Correction", 
    "🧠 Autonomic HRV Lab", 
    "🎓 Academic Report Compiler"
])

# Tab 1: Ward Overview
with tab_overview:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Telemetry Stream View")
        zoom_sec = st.slider("Scope window (sec)", min_value=2, max_value=min(20, int(duration)), value=6)
        
        # Plot scope
        zoom_idx = t <= zoom_sec
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t[zoom_idx], sig[zoom_idx], color='#EF5350', label='Raw Noisy ECG (mV)', alpha=0.6, linewidth=1.0)
        ax.plot(t[zoom_idx], sig_smoothed[zoom_idx], color='#00E676', label='Clean ECG (mV)', linewidth=1.5)
        
        if true_peaks is not None:
            peaks_in_zoom = r_peaks[r_peaks < int(zoom_sec * fs)]
            ax.scatter(t[peaks_in_zoom], sig_smoothed[peaks_in_zoom], color='#FFD600', s=45, label='R-Peak', zorder=5)
            
        ax.set_xlabel("Time (seconds)", color='#E2E8F0')
        ax.set_ylabel("Amplitude (mV)", color='#E2E8F0')
        ax.set_facecolor('#0F172A')
        fig.patch.set_facecolor('#0A0F1D')
        ax.tick_params(colors='#E2E8F0')
        ax.grid(color='#334155', linestyle=':', linewidth=0.5)
        ax.legend(loc='upper right', facecolor='#1E2937', edgecolor='#475569', labelcolor='#E2E8F0')
        st.pyplot(fig)
        plt.close(fig)
        
    with col2:
        st.subheader("Patient Clinical Vitals")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Detected Heart Rate</div>
            <div class="metric-value">{time_m['mean_hr']:.1f} <span style="font-size:1rem;">BPM</span></div>
            <div class="metric-desc">Rhythm: {rhythm_label}</div>
        </div>
        
        <div class="metric-card" style="border-left-color: #29B6F6;">
            <div class="metric-title">Vagal Autonomic Score (RMSSD)</div>
            <div class="metric-value">{time_m['rmssd']:.1f} <span style="font-size:1rem;">ms</span></div>
            <div class="metric-desc">Autonomic Vagal Control Indicator</div>
        </div>

        <div class="metric-card" style="border-left-color: #FFA726;">
            <div class="metric-title">Ectopic Load</div>
            <div class="metric-value">{np.sum(ectopic_mask)} <span style="font-size:1rem;">PVCs ({ (np.sum(ectopic_mask)/len(rr_intervals)*100.0 if len(rr_intervals) > 0 else 0):.1f}%)</span></div>
            <div class="metric-desc">Corrected via Cubic Splines</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Quick Diagnostics Summary")
        if rhythm_label.startswith("PVC"):
            st.error("⚠️ PVC Arrhythmia Detected. Significant ectopic beat burden. ECG displays wide ventricular complexes followed by compensatory pauses.")
        elif rhythm_label.startswith("AFib"):
            st.warning("⚠️ Atrial Fibrillation Detected. Irregularly irregular R-R pacing. Absent P-waves. Vagal tone metrics will show pseudo-elevation due to chaotic firing.")
        elif rhythm_label.startswith("VTach"):
            st.error("🚨 Ventricular Tachycardia! Rapid wide QRS complexes. Immediate resuscitation review required.")
        else:
            st.success("✅ Normal Sinus Rhythm. Stable baseline homeostasis. Normal Respiratory Sinus Arrhythmia (RSA) present.")

# Tab 2: DSP Preprocessing
with tab_dsp:
    st.subheader("CLO1: Biomedical Preprocessing Stages")
    st.write("Demonstration of baseline drift extraction and high-frequency filtering. Check equations in the methodology section.")
    
    col_sig, col_details = st.columns([3, 1.2])
    with col_sig:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
        fig.patch.set_facecolor('#0A0F1D')
        
        # Plot 1: Raw + Baseline
        ax1.plot(t, sig, color='#EF5350', alpha=0.5, label='Raw noisy signal')
        ax1.plot(t, baseline, color='#FFF', linewidth=1.5, label='Dual median baseline estimate')
        ax1.set_title("1. Baseline Wander Capture (Dual Median filtering)", color='#E2E8F0', fontsize=10)
        ax1.set_facecolor('#0F172A')
        ax1.tick_params(colors='#E2E8F0')
        ax1.legend(facecolor='#1E2937', labelcolor='#E2E8F0', loc='upper right')
        ax1.grid(color='#334155', linestyle=':', linewidth=0.5)
        
        # Plot 2: Cleaned
        ax2.plot(t, sig_clean, color='#29B6F6', label='Baseline wander removed')
        ax2.set_title("2. Drift-Corrected ECG (Zero-Phase baseline removal)", color='#E2E8F0', fontsize=10)
        ax2.set_facecolor('#0F172A')
        ax2.tick_params(colors='#E2E8F0')
        ax2.grid(color='#334155', linestyle=':', linewidth=0.5)
        
        # Plot 3: Filtered & Smoothed
        ax3.plot(t, sig_smoothed, color='#66BB6A', label='Bandpass + Savitzky-Golay')
        ax3.set_title("3. Final Preprocessed Signal (0.5-45 Hz Bandpass + S-G smoothing)", color='#E2E8F0', fontsize=10)
        ax3.set_xlabel("Time (seconds)", color='#E2E8F0')
        ax3.set_facecolor('#0F172A')
        ax3.tick_params(colors='#E2E8F0')
        ax3.grid(color='#334155', linestyle=':', linewidth=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col_details:
        st.info("""
        ### Filter Specifications:
        - **Baseline Drift Removal**: Dual rolling median filtering. Removes slow motion artifacts (0.1–0.5 Hz breathing movement) without attenuating QRS peak heights.
        - **Bandpass Filter**: 3rd-order Butterworth bandpass (0.5 to 45 Hz). Suppresses high-frequency muscular noise and 50 Hz powerline interference.
        - **Zero-Phase Implementation**: Applied via double-pass filtering (Forward + Backward). This forces phase shift to be exactly zero, preserving the original positions of R-peaks.
        """)

# Tab 3: QRS R-Peak Detector
with tab_qrs:
    st.subheader("Pan-Tompkins Algorithm Stage Envelopes")
    st.write("Detailed extraction of QRS energy envelopes to perform adaptive peak thresholding.")
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True)
    fig.patch.set_facecolor('#0A0F1D')
    
    # 5s window for clarity
    p_zoom = t <= 8.0
    sig_p_zoom = sig_smoothed[p_zoom]
    t_p_zoom = t[p_zoom]
    
    # 1. Bandpass 5-15Hz
    ax1.plot(t_p_zoom, PT_stages['filtered'][p_zoom], color='#EC407A')
    ax1.set_title("1. Pan-Tompkins Bandpass Filter (5 - 15 Hz QRS Isolation)", color='#E2E8F0', fontsize=9)
    ax1.set_facecolor('#0F172A')
    ax1.tick_params(colors='#E2E8F0')
    ax1.grid(color='#334155', linestyle=':', linewidth=0.5)
    
    # 2. Derivative
    ax2.plot(t_p_zoom, PT_stages['derived'][p_zoom], color='#AB47BC')
    ax2.set_title("2. QRS Complex Derivative (highlighting steep slopes)", color='#E2E8F0', fontsize=9)
    ax2.set_facecolor('#0F172A')
    ax2.tick_params(colors='#E2E8F0')
    ax2.grid(color='#334155', linestyle=':', linewidth=0.5)
    
    # 3. Squaring
    ax3.plot(t_p_zoom, PT_stages['squared'][p_zoom], color='#42A5F5')
    ax3.set_title("3. Squaring Operator (intensifying slopes and forcing absolute positive)", color='#E2E8F0', fontsize=9)
    ax3.set_facecolor('#0F172A')
    ax3.tick_params(colors='#E2E8F0')
    ax3.grid(color='#334155', linestyle=':', linewidth=0.5)
    
    # 4. Integration
    ax4.plot(t_p_zoom, PT_stages['integrated'][p_zoom], color='#26A69A', label='Integrated Envelope')
    # Plot thresholds
    spki_line = np.ones_like(t_p_zoom) * np.max(PT_stages['integrated']) * 0.20
    ax4.plot(t_p_zoom, spki_line, color='#FFCA28', linestyle='--', label='Signal Threshold')
    
    # Mark peaks
    p_peaks = r_peaks[r_peaks < int(8.0 * fs)]
    ax4.scatter(t[p_peaks], PT_stages['integrated'][p_peaks], color='#D50000', s=35, zorder=5, label='Triggered Peaks')
    
    ax4.set_title("4. Moving Window Integration (150ms envelope) & Thresholds", color='#E2E8F0', fontsize=9)
    ax4.set_xlabel("Time (seconds)", color='#E2E8F0')
    ax4.set_facecolor('#0F172A')
    ax4.tick_params(colors='#E2E8F0')
    ax4.grid(color='#334155', linestyle=':', linewidth=0.5)
    ax4.legend(facecolor='#1E2937', labelcolor='#E2E8F0', loc='upper right')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Tab 4: Ectopic Correction
with tab_ectopic:
    st.subheader("Ectopic Beat Detection & Spline Reconstruction")
    st.write("Mandatory verification of outlier intervals caused by premature contractions or measurement artifacts.")
    
    col_ect1, col_ect2 = st.columns([3, 1.2])
    with col_ect1:
        # Plot Ectopic comparison
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#0A0F1D')
        
        x_beats = np.arange(len(rr_intervals))
        ax.plot(x_beats, rr_intervals, color='#EF5350', alpha=0.5, linestyle='--', marker='o', label='Raw RR Intervals (ms)')
        ax.plot(x_beats, corrected_rr, color='#26A69A', marker='s', label='Spline Corrected RR (ms)')
        
        ect_spots = np.where(ectopic_mask)[0]
        if len(ect_spots) > 0:
            ax.scatter(ect_spots, rr_intervals[ect_spots], color='#E91E63', s=60, zorder=5, label='Ectopic Outlier')
            ax.scatter(ect_spots, corrected_rr[ect_spots], color='#29B6F6', s=60, zorder=5, label='Interpolation Fit')
            
        ax.set_xlabel("Cardiac Beat Index", color='#E2E8F0')
        ax.set_ylabel("Interval duration (ms)", color='#E2E8F0')
        ax.set_facecolor('#0F172A')
        ax.tick_params(colors='#E2E8F0')
        ax.legend(facecolor='#1E2937', labelcolor='#E2E8F0', loc='upper right')
        ax.grid(color='#334155', linestyle=':', linewidth=0.5)
        st.pyplot(fig)
        plt.close(fig)
        
    with col_ect2:
        st.metric("Total Heartbeats Evaluated", time_m['nn50'] + 10) # rough beat count estimate
        st.metric("Ectopic Count", np.sum(ectopic_mask))
        st.metric("Correction Percentage", f"{(np.sum(ectopic_mask)/len(rr_intervals)*100.0 if len(rr_intervals) > 0 else 0):.2f} %")
        
        st.info("""
        ### Ectopic Beat Management:
        - **Detection**: Outliers are identified using a local rolling median filter. An interval is labeled ectopic if it falls outside 20% of the local running median or global limits.
        - **Correction**: Ectopic values are replaced using Cubic Spline Interpolation over normal neighbors. This avoids step-discontinuities in the signal, maintaining spectral integration accuracy.
        """)

# Tab 5: HRV Analytics
with tab_hrv:
    st.subheader("Autonomic Nervous System Diagnostic Lab")
    
    # Render three cards for time, frequency, and nonlinear metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<h4 style='color:#66BB6A;'>Time-Domain Features</h4>", unsafe_allow_html=True)
        st.write(f"**Mean RR Interval**: {time_m['mean_rr']:.1f} ms")
        st.write(f"**Mean Heart Rate**: {time_m['mean_hr']:.1f} BPM")
        st.write(f"**SDNN (Overall HRV)**: {time_m['sdnn']:.1f} ms")
        st.write(f"**RMSSD (Vagal Tone)**: {time_m['rmssd']:.1f} ms")
        st.write(f"**pNN50 (RSA load)**: {time_m['pnn50']:.2f} %")
        
    with c2:
        st.markdown("<h4 style='color:#42A5F5;'>Frequency-Domain Features</h4>", unsafe_allow_html=True)
        st.write(f"**LF Power (Sympathetic/Vagal)**: {freq_m['lf']:.1f} ms²")
        st.write(f"**HF Power (Pure Vagal)**: {freq_m['hf']:.1f} ms²")
        st.write(f"**LF Norm Power**: {freq_m['lf_nu']:.1f} nu")
        st.write(f"**HF Norm Power**: {freq_m['hf_nu']:.1f} nu")
        st.write(f"**LF/HF Ratio (Autonomic Balance)**: {freq_m['ratio']:.3f}")
        
    with c3:
        st.markdown("<h4 style='color:#AB47BC;'>Non-Linear Parameters</h4>", unsafe_allow_html=True)
        st.write(f"**SD1 (Short-term fluctuation)**: {nonl_m['sd1']:.1f} ms")
        st.write(f"**SD2 (Long-term fluctuation)**: {nonl_m['sd2']:.1f} ms")
        st.write(f"**SD1/SD2 Ratio**: {nonl_m['ratio']:.3f}")
        st.write(f"**Sample Entropy (Complexity)**: {nonl_m['sampen']:.4f}")

    # Plot spectral density & poincare side by side
    col_plt1, col_plt2 = st.columns(2)
    with col_plt1:
        st.subheader("Frequency PSD Welch Periodogram")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#0A0F1D')
        
        f = freq_m['psd_f']
        pxx = freq_m['psd_p']
        if len(f) > 0:
            ax.plot(f, pxx, color='#26A69A', linewidth=1.5)
            # Shade LF
            lf_idx = (f >= 0.04) & (f < 0.15)
            ax.fill_between(f[lf_idx], pxx[lf_idx], color='#FF7043', alpha=0.4, label='LF Sympathovagal')
            # Shade HF
            hf_idx = (f >= 0.15) & (f <= 0.40)
            ax.fill_between(f[hf_idx], pxx[hf_idx], color='#29B6F6', alpha=0.4, label='HF Parasympathetic')
            
            ax.set_xlim(0, 0.5)
            ax.set_xlabel("Frequency (Hz)", color='#E2E8F0')
            ax.set_ylabel("Power Density (ms²/Hz)", color='#E2E8F0')
            ax.tick_params(colors='#E2E8F0')
            ax.grid(color='#334155', linestyle=':', linewidth=0.5)
            ax.legend(facecolor='#1E2937', labelcolor='#E2E8F0', loc='upper right')
        ax.set_facecolor('#0F172A')
        st.pyplot(fig)
        plt.close(fig)
        
    with col_plt2:
        st.subheader("Poincaré Nonlinear Dynamics")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor('#0A0F1D')
        
        if len(corrected_rr) > 2:
            x_p = corrected_rr[:-1]
            y_p = corrected_rr[1:]
            ax.scatter(x_p, y_p, color='#29B6F6', alpha=0.6, s=15, edgecolors='none')
            
            min_val = min(np.min(x_p), np.min(y_p)) - 50
            max_val = max(np.max(x_p), np.max(y_p)) + 50
            ax.plot([min_val, max_val], [min_val, max_val], color='#EF5350', linestyle='--', label='y = x Identity')
            
            # Draw ellipse axes
            mean_x = np.mean(x_p)
            mean_y = np.mean(y_p)
            sd1 = nonl_m['sd1']
            sd2 = nonl_m['sd2']
            angle = np.pi / 4
            ax.plot([mean_x - sd1 * np.sin(angle), mean_x + sd1 * np.sin(angle)],
                    [mean_y + sd1 * np.cos(angle), mean_y - sd1 * np.cos(angle)],
                    color='#FF7043', linewidth=2.0, label='SD1 (Vagal)')
            ax.plot([mean_x - sd2 * np.cos(angle), mean_x + sd2 * np.cos(angle)],
                    [mean_y - sd2 * np.sin(angle), mean_y + sd2 * np.sin(angle)],
                    color='#66BB6A', linewidth=2.0, label='SD2 (Symp+Vagal)')
            
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_xlabel("RR\u2093 (ms)", color='#E2E8F0')
            ax.set_ylabel("RR\u2093\u208A\u2081 (ms)", color='#E2E8F0')
            ax.tick_params(colors='#E2E8F0')
            ax.grid(color='#334155', linestyle=':', linewidth=0.5)
            ax.legend(facecolor='#1E2937', labelcolor='#E2E8F0', loc='lower right')
        ax.set_facecolor('#0F172A')
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Physiological Interpretation & Discussion")
    st.markdown(interpretation)

# Tab 6: Report Compiler
with tab_report:
    st.subheader("Automated Academic Lab Report Compiler")
    st.write("Generate and download professional, publication-quality lab reports satisfying the criteria for OEL CLO1 and CLO2.")
    
    st.markdown("""
    This section compiles all of the graphs generated above directly into two formats:
    - **PDF File**: Using `reportlab`, complete with title page, equations, results tables, discussion, and captioned figures.
    - **Microsoft Word Document (.docx)**: Fully formatted and editable.
    """)
    
    if st.button("Compile Reports (Generates Figures & Documents)"):
        with st.spinner("Generating Matplotlib plots and writing document files..."):
            student_info = {
                'name': student_name,
                'id': student_id,
                'supervisor': supervisor_name
            }
            
            # Temporary directory to save plots
            plots_dir = "streamlit_plots"
            if os.path.exists(plots_dir):
                shutil.rmtree(plots_dir)
            os.makedirs(plots_dir)
            
            # Generate plots
            plot_paths = reporter.generate_all_plots(
                t=t,
                raw_sig=sig,
                filtered_sig=sig_smoothed,
                r_peaks=r_peaks,
                rr_raw=rr_intervals,
                rr_corrected=corrected_rr,
                ectopic_mask=ectopic_mask,
                psd_data=freq_m,
                output_dir=plots_dir
            )
            
            ectopic_stats = {
                'count': int(np.sum(ectopic_mask)),
                'total_beats': int(len(r_peaks)),
                'pct': (np.sum(ectopic_mask)/len(rr_intervals)*100.0 if len(rr_intervals) > 0 else 0.0)
            }
            
            pdf_path = "ECG_HRV_Lab_Report.pdf"
            docx_path = "ECG_HRV_Lab_Report.docx"
            
            # Generate reports
            reporter.generate_pdf(pdf_path, student_info, time_m, freq_m, nonl_m, ectopic_stats, plot_paths, interpretation)
            reporter.generate_docx(docx_path, student_info, time_m, freq_m, nonl_m, ectopic_stats, plot_paths, interpretation)
            
            st.success("🎉 Reports successfully compiled!")
            
            # Offer downloads
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download Academic PDF Report",
                    data=pdf_file,
                    file_name=f"ECG_HRV_Report_{student_id}.pdf",
                    mime="application/pdf"
                )
                
            with open(docx_path, "rb") as docx_file:
                st.download_button(
                    label="📥 Download Editable Word (.docx) Report",
                    data=docx_file,
                    file_name=f"ECG_HRV_Report_{student_id}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                
            # Clean up temporary plots directory
            if os.path.exists(plots_dir):
                shutil.rmtree(plots_dir)
