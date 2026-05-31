import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import shutil
import tempfile
import zipfile

from core.synthetic_data import SyntheticECGGenerator
from core.signal_processing import ECGProcessor
from core.hrv_analysis import HRVAnalyzer
from core.report_generator import ReportGenerator
from core.file_loader import load_ecg_file

# Page configuration
st.set_page_config(
    page_title="Clinical ICU Telemetry & ECG-HRV Analytics Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-mode premium medical styling
st.markdown("""
<style>
    /* Main Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #080C16 !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #080C16 !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(8, 12, 22, 0.8) !important;
        backdrop-filter: blur(12px);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F1423 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #38BDF8 !important;
        font-weight: 700;
    }

    /* Custom Premium Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.85));
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 5px solid #38BDF8;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(6px);
        margin-bottom: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.15);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .metric-title {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.2rem;
        color: #FFFFFF;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-desc {
        font-size: 0.75rem;
        color: #38BDF8;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Custom Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        color: #94A3B8 !important;
        font-weight: 600;
        border: none !important;
        background-color: transparent !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        background-color: rgba(56, 189, 248, 0.12) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.05);
    }
    
    /* Code Blocks */
    code {
        color: #F1F5F9 !important;
        background-color: #0F172A !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284C7, #0369A1) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369A1, #075985) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header Card Style
st.markdown("""
<div style="background: linear-gradient(90deg, #0B0F19 0%, #172033 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #38BDF8; margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);">
    <h1 style="color: #FFFFFF; margin: 0; font-size: 2.2rem; font-weight: 800; display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 2.5rem;">🏥</span> Clinical ICU Telemetry & ECG-HRV Analytics Hub
    </h1>
    <p style="color: #94A3B8; margin: 8px 0 0 0; font-size: 1.1rem; font-weight: 500;">
        Academic Laboratory Open-Ended Lab (OEL) Evaluation Platform — <span style="color: #38BDF8; font-weight: 700;">CLO1 & CLO2 Compliance</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Settings Panel
st.sidebar.header("🎛️ Analysis Settings")

# Common settings
fs = st.sidebar.number_input("Sampling Frequency (fs in Hz)", value=250, min_value=100, max_value=1000)

st.sidebar.subheader("ECG Filter Cutoffs")
lowcut = st.sidebar.slider("Butterworth Low Cut (Hz)", 0.1, 2.0, 0.5, step=0.1)
highcut = st.sidebar.slider("Butterworth High Cut (Hz)", 20.0, 100.0, 40.0, step=5.0)

st.sidebar.subheader("QRS R-Peak Method")
rpeak_method = st.sidebar.selectbox(
    "Algorithm Selection",
    [
        "Pan-Tompkins (Custom)",
        "NeuroKit2 (Default)",
        "NeuroKit2 (Hamilton)",
        "NeuroKit2 (Elgendi)",
        "NeuroKit2 (Kalidas)",
        "NeuroKit2 (Engzee)"
    ]
)

st.sidebar.subheader("Ectopic Beat Handling")
ectopic_corrected = st.sidebar.checkbox("Enable Ectopic Correction", value=True)
ectopic_thresh = st.sidebar.slider("Outlier Percent Threshold (%)", 10, 40, 20, step=5) / 100.0
corr_method = st.sidebar.selectbox("Interpolation Type", ["spline", "linear"])

st.sidebar.subheader("Welch PSD Parameters")
welch_win_sec = st.sidebar.slider("Welch Window Size (sec)", 16, 128, 64, step=8)
welch_overlap_pct = st.sidebar.slider("Welch Overlap (%)", 0, 90, 50, step=10)

# Build settings dict
settings = {
    'fs': fs,
    'lowcut': lowcut,
    'highcut': highcut,
    'rpeak_method': rpeak_method,
    'ectopic_corrected': ectopic_corrected,
    'ectopic_thresh': ectopic_thresh,
    'corr_method': corr_method,
    'welch_win_sec': welch_win_sec,
    'welch_overlap_pct': welch_overlap_pct
}

# --- Load files ---
sig_files = []
active_sig_name = None
true_peaks = None

uploaded_files = st.sidebar.file_uploader(
    "Upload ECG Records",
    type=["csv", "txt", "mat", "dat", "edf"],
    accept_multiple_files=True
)

if len(uploaded_files) > 0:
    for f in uploaded_files:
        # Save uploaded file in temp location to read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as temp_f:
            temp_f.write(f.getbuffer())
            temp_path = temp_f.name
            
        try:
            t, sig, file_fs = load_ecg_file(temp_path, fs=fs)
            sig_files.append({
                "name": f.name,
                "t": t,
                "sig": sig,
                "fs": file_fs
            })
        except Exception as e:
            st.error(f"Error loading {f.name}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    if len(sig_files) > 0:
        active_sig_name = st.sidebar.selectbox("Select Active File to Inspect", [s["name"] for s in sig_files])

# Helper function to run processing on a signal record
def process_record(record, settings):
    # Pull parameters
    record_fs = record.get("fs", settings['fs'])
    processor = ECGProcessor(fs=record_fs)
    analyzer = HRVAnalyzer(fs_ecg=record_fs)
    
    sig = record["sig"]
    t = record["t"]
    
    # 1. DSP filtering
    sig_clean, baseline = processor.remove_baseline_wander_median(sig)
    sig_filtered = processor.apply_bandpass(sig_clean, lowcut=settings['lowcut'], highcut=settings['highcut'])
    sig_smoothed = processor.remove_noise_savgol(sig_filtered)
    
    # 2. R-peak detection
    method = settings['rpeak_method']
    if method == "Pan-Tompkins (Custom)":
        r_peaks, stages = processor.pan_tompkins_detector(sig_smoothed)
    else:
        # Convert NeuroKit2 method string
        nk_methods = {
            "NeuroKit2 (Default)": "neurokit",
            "NeuroKit2 (Hamilton)": "hamilton2002",
            "NeuroKit2 (Elgendi)": "elgendi2010",
            "NeuroKit2 (Kalidas)": "kalidas2016",
            "NeuroKit2 (Engzee)": "engzee2012"
        }
        nk_method = nk_methods.get(method, "neurokit")
        r_peaks = processor.detect_peaks_nk(sig_smoothed, method=nk_method)
        stages = None
        
    # 3. RR interval extraction
    rr_intervals = np.diff(r_peaks) / record_fs * 1000.0
    
    # 4. Ectopic correction
    ectopic_mask = analyzer.detect_ectopic_beats(rr_intervals, threshold_pct=settings['ectopic_thresh'])
    if settings['ectopic_corrected']:
        corrected_rr = analyzer.correct_ectopic_beats(rr_intervals, ectopic_mask, method=settings['corr_method'])
    else:
        corrected_rr = rr_intervals
        
    ectopic_count = np.sum(ectopic_mask)
    ectopic_pct = (ectopic_count / len(rr_intervals)) * 100.0 if len(rr_intervals) > 0 else 0.0
    ectopic_stats = {
        'count': int(ectopic_count),
        'total_beats': int(len(r_peaks)),
        'pct': ectopic_pct
    }
    
    # 5. HRV features
    time_m = analyzer.compute_time_domain(corrected_rr)
    freq_m = analyzer.compute_frequency_domain(
        corrected_rr,
        welch_win_sec=settings['welch_win_sec'],
        welch_overlap_pct=settings['welch_overlap_pct']
    )
    nonl_m = analyzer.compute_nonlinear(corrected_rr)
    interpretation = analyzer.generate_clinical_interpretation(time_m, freq_m, nonl_m)
    
    return {
        't': t,
        'sig': sig,
        'sig_smoothed': sig_smoothed,
        'baseline': baseline,
        'r_peaks': r_peaks,
        'rr_intervals': rr_intervals,
        'corrected_rr': corrected_rr,
        'ectopic_mask': ectopic_mask,
        'ectopic_stats': ectopic_stats,
        'time_m': time_m,
        'freq_m': freq_m,
        'nonl_m': nonl_m,
        'interpretation': interpretation,
        'stages': stages,
        'fs': record_fs
    }

# Process the active file
res = None
if len(sig_files) > 0 and active_sig_name is not None:
    active_record = next(s for s in sig_files if s["name"] == active_sig_name)
    res = process_record(active_record, settings)

# Dashboard tabs
tab_intro, tab_overview, tab_dsp, tab_qrs, tab_ectopic, tab_hrv_time, tab_hrv_freq, tab_hrv_nonl, tab_report = st.tabs([
    "🏠 Academic Intro Desk",
    "📋 Ward Overview", 
    "📈 ECG Signal", 
    "⚡ QRS Detector", 
    "🩺 RR & Ectopic Analysis", 
    "📊 Time-Domain HRV", 
    "📈 Frequency-Domain HRV",
    "🔬 Non-Linear Analysis",
    "🎓 Report Desk"
])

# Tab 0: Welcome & Academic Desk (Introductory Page)
with tab_intro:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.4), rgba(30, 41, 59, 0.6)); padding: 25px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 20px;">
        <h2 style="color: #38BDF8; margin-top: 0; font-weight: 700; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.8rem;">🔬</span> ECG-HRV System Architecture & CLO Alignment
        </h2>
        <p style="font-size: 1.05rem; line-height: 1.6; color: #E2E8F0; margin-bottom: 0;">
            Welcome to the <b>Clinical ICU Telemetry & ECG-HRV Analytics Hub</b>. This workspace is a 
            high-fidelity clinical signal processing engine designed to meet the rigorous standards for 
            academic evaluations. It translates raw electrocardiogram (ECG) data into actionable 
            Heart Rate Variability (HRV) parameters, representing autonomic nervous system (ANS) tone.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Specific OEL core objective requested by the user
    st.markdown("""
    <div style="background-color: rgba(56, 189, 248, 0.05); border: 1px dashed rgba(56, 189, 248, 0.25); padding: 18px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="color: #38BDF8; margin: 0 0 10px 0; font-weight: 700; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">
            <span>🎯</span> Open-Ended Lab (OEL) Core Objective
        </h4>
        <p style="margin: 0; font-size: 1.02rem; color: #F1F5F9; line-height: 1.6; font-style: italic;">
            "Develop time-domain, statistical, and non-linear HRV analysis modules from ECG signals, including RR-interval 
            extraction, SDNN, RMSSD, Poincaré plots, and entropy measures for comprehensive variability assessment. 
            Implement frequency-domain HRV analysis with power spectral density (LF/HF bands) to quantify parasympathetic (HF) 
            and sympathetic (LF, LF/HF ratio) autonomic responses in an interactive ECG-HRV dashboard."
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
        ### ⚙️ The Biomedical Signal Processing (BSP) Pipeline
        The system implements a robust, modular signal processing pipeline:
        
        1. **ECG Acquisition (Stationary)**: Loads high-fidelity records from the **Custom File Uploads** (.csv, .txt, .mat, .dat, .edf). 
           *(Note: Live streaming sensor inputs and pre-loaded databases are removed to ensure direct student file analysis and reproducibility).*
        2. **Baseline Drift Suppression**: Eliminates low-frequency respiration swings using a non-linear dual median filter (200ms and 600ms cascaded windows) preserving absolute QRS amplitudes.
        3. **High-Frequency Noise Denoising**: Filters out muscle tremors (EMG) and 50Hz powerline hum using a zero-phase 3rd-order Butterworth bandpass filter.
        4. **QRS Complex & R-Peak Detection**: Extracts heartbeat indices using the adaptive Pan-Tompkins derivative-thresholding algorithm or NeuroKit2.
        5. **Ectopic Outlier Correction**: Identifies premature ventricular contractions (PVCs) and repairs them using cubic spline interpolation.
        """)
        
    with col_i2:
        st.markdown("""
        ### 🎓 Academic Evaluation & Syllabus Mapping
        This platform is designed to align directly with grading metrics for Open-Ended Labs (OEL):
        
        * **CLO1: Digital Filter Design & Peak Alignment**: Demonstrates zero-phase delay filtering, baseline wander subtraction, and QRS window integration.
        * **CLO2: Feature Synthesis & Interpretation**: Computes and explains statistical time-domain features (SDNN, RMSSD, pNN50), spectral frequency bands (Welch PSD integration for LF and HF), and non-linear complexity metrics (Poincaré scatter plots, Sample Entropy).
        * **Scientific Reporting**: Compiles all results, discussion logs, and 7 mandatory evaluation figures into downloadable, print-ready PDF and editable Word DOCX templates.
        """)
        
    st.markdown("""
    <div style="background-color: rgba(56, 189, 248, 0.06); border-left: 5px solid #38BDF8; padding: 15px; border-radius: 6px; margin-top: 15px; border-top: 1px solid rgba(56, 189, 248, 0.1); border-right: 1px solid rgba(56, 189, 248, 0.1); border-bottom: 1px solid rgba(56, 189, 248, 0.1);">
        <h4 style="color: #38BDF8; margin: 0 0 6px 0; font-weight: 700;">📡 Operational Notice: Stationary File Processing Only</h4>
        <p style="margin: 0; font-size: 0.95rem; color: #E2E8F0; line-height: 1.5;">
            To ensure zero package drop, avoid active hardware sensor calibration errors, and maintain strict academic reproducibility, 
            <b>the option for live streaming input is removed from this system</b>. All digital processing, filtering, and HRV assessments 
            are performed on pre-acquired stationary patient datasets or custom telemetry files uploaded by the user.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if res is None:
        st.markdown("---")
        st.info("👈 **Please upload one or more ECG records in the sidebar to activate the clinical analytics dashboard tabs.**")
        st.stop()

# Tab 1: Ward Overview
with tab_overview:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Interactive Scope: {active_sig_name}")
        zoom_sec = st.slider("Scope window (sec)", min_value=2, max_value=min(30, int(len(res['sig'])/res['fs'])), value=8)
        
        # Plotly ECG Preprocessing Visualizer
        zoom_idx = res['t'] <= zoom_sec
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['t'][zoom_idx], y=res['sig'][zoom_idx], name="Raw Noisy ECG", line=dict(color="#EF5350", width=1), opacity=0.6))
        fig.add_trace(go.Scatter(x=res['t'][zoom_idx], y=res['sig_smoothed'][zoom_idx], name="Cleaned ECG", line=dict(color="#00E676", width=1.5)))
        
        # Peak markers
        peaks_in_zoom = res['r_peaks'][res['r_peaks'] < int(zoom_sec * res['fs'])]
        fig.add_trace(go.Scatter(x=res['t'][peaks_in_zoom], y=res['sig_smoothed'][peaks_in_zoom], mode="markers", name="Detected R-Peak", marker=dict(color="#FFD600", size=10, line=dict(color="black", width=1))))
        
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (mV)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Telemetry Vitals")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Heart Rate</div>
            <div class="metric-value">{res['time_m']['mean_hr']:.1f} <span style="font-size:1rem;">BPM</span></div>
            <div class="metric-desc">Method: {settings['rpeak_method']}</div>
        </div>
        
        <div class="metric-card" style="border-left-color: #29B6F6;">
            <div class="metric-title">Vagal Autonomic Score (RMSSD)</div>
            <div class="metric-value">{res['time_m']['rmssd']:.1f} <span style="font-size:1rem;">ms</span></div>
            <div class="metric-desc">Parasympathetic Vagal Control</div>
        </div>

        <div class="metric-card" style="border-left-color: #FFA726;">
            <div class="metric-title">Ectopic Load</div>
            <div class="metric-value">{res['ectopic_stats']['count']} <span style="font-size:1rem;">PVCs ({res['ectopic_stats']['pct']:.2f}%)</span></div>
            <div class="metric-desc">Correction: {"ON (" + settings['corr_method'] + ")" if settings['ectopic_corrected'] else "OFF"}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Batch summaries if multiple files
    if len(sig_files) > 1:
        st.markdown("---")
        st.subheader("📊 Batch Processing Summary Overview")
        
        batch_data = []
        for s in sig_files:
            try:
                s_res = process_record(s, settings)
                batch_data.append({
                    "File Name": s["name"],
                    "Heart Rate (BPM)": f"{s_res['time_m']['mean_hr']:.1f}",
                    "SDNN (ms)": f"{s_res['time_m']['sdnn']:.1f}",
                    "RMSSD (ms)": f"{s_res['time_m']['rmssd']:.1f}",
                    "LF/HF Ratio": f"{s_res['freq_m']['ratio']:.2f}",
                    "Ectopic Count": f"{s_res['ectopic_stats']['count']} ({s_res['ectopic_stats']['pct']:.1f}%)",
                    "Complexity (SampEn)": f"{s_res['nonl_m']['sampen']:.3f}"
                })
            except Exception as e:
                st.write(f"Could not process {s['name']}: {e}")
                
        df_batch = pd.DataFrame(batch_data)
        st.dataframe(df_batch, use_container_width=True)

# Tab 2: ECG Preprocessing
with tab_dsp:
    st.subheader("ECG Preprocessing & Baseline Wander removal (CLO1)")
    st.write("Demonstration of baseline drift extraction and bandpass noise suppression.")
    
    col_sig, col_details = st.columns([3, 1.2])
    with col_sig:
        # Plotly chart of raw, baseline drift, and cleaned signal
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['t'], y=res['sig'], name="Raw Signal", line=dict(color="#EF5350", width=1), opacity=0.5))
        fig.add_trace(go.Scatter(x=res['t'], y=res['baseline'], name="Baseline Wander (Dual Median)", line=dict(color="#FFFFFF", width=1.5)))
        fig.add_trace(go.Scatter(x=res['t'], y=res['sig_smoothed'], name="Processed (Bandpass)", line=dict(color="#00E676", width=1.2)))
        
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (mV)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_details:
        st.info(f"""
        ### Filter Designs:
        - **Baseline Wander Removal**: Dual median filter (200ms and 600ms windows) successfully isolate respiratory baseline drifts.
        - **Denoising (Bandpass)**: Butterworth 3rd-order digital bandpass filter ($ {settings['lowcut']:.1f} - {settings['highcut']:.1f} $ Hz) applied with zero phase shift.
        - **Smoothing**: Savitzky-Golay filtering removes muscle tremor artifacts.
        """)

# Tab 3: QRS Detector
with tab_qrs:
    st.subheader("Pan-Tompkins Algorithm Stage Visualizer")
    st.write("Examine QRS energy envelopes to test derivative-threshold parameters.")
    
    if res['stages'] is not None:
        p_zoom = res['t'] <= 8.0
        t_zoom = res['t'][p_zoom]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_zoom, y=res['stages']['filtered'][p_zoom], name="Bandpass 5-15Hz", line=dict(color="#EC407A")))
        fig.add_trace(go.Scatter(x=t_zoom, y=res['stages']['derived'][p_zoom], name="Derivative Stage", line=dict(color="#AB47BC")))
        fig.add_trace(go.Scatter(x=t_zoom, y=res['stages']['squared'][p_zoom], name="Squaring Stage", line=dict(color="#42A5F5")))
        fig.add_trace(go.Scatter(x=t_zoom, y=res['stages']['integrated'][p_zoom], name="Integrated Envelope", line=dict(color="#26A69A", width=2)))
        
        # Overlay peaks
        z_peaks = res['r_peaks'][res['r_peaks'] < int(8.0 * res['fs'])]
        fig.add_trace(go.Scatter(x=res['t'][z_peaks], y=res['stages']['integrated'][z_peaks], mode="markers", name="Detected R-Peak", marker=dict(color="#D50000", size=10, symbol="x")))
        
        fig.update_layout(
            title="Intermediate Pan-Tompkins Signal Stages (First 8 seconds)",
            xaxis_title="Time (seconds)",
            yaxis_title="Normalized Amplitude",
            template="plotly_dark",
            height=480
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"R-peaks detected using {settings['rpeak_method']}. intermediate Pan-Tompkins filter envelopes are only available when the Custom Pan-Tompkins method is selected.")

# Tab 4: RR & Ectopic Analysis
with tab_ectopic:
    st.subheader("RR Tachogram & Ectopic Beat Management")
    st.write("Inspect outliers representing ectopic cardiac contractions.")
    
    # Plotly Tachogram (Raw vs Corrected)
    fig = go.Figure()
    x_idx = np.arange(len(res['rr_intervals']))
    
    fig.add_trace(go.Scatter(x=x_idx, y=res['rr_intervals'], name="Raw RR Intervals", line=dict(color="#EF5350", width=1.2, dash="dash"), mode="lines+markers"))
    
    if settings['ectopic_corrected']:
        fig.add_trace(go.Scatter(x=x_idx, y=res['corrected_rr'], name="Corrected RR (Cubic Spline)", line=dict(color="#00E676", width=1.5), mode="lines+markers"))
        
    ect_idx = np.where(res['ectopic_mask'])[0]
    if len(ect_idx) > 0:
        fig.add_trace(go.Scatter(x=ect_idx, y=res['rr_intervals'][ect_idx], mode="markers", name="Flagged Ectopics (Outliers)", marker=dict(color="#FF1744", size=10)))
        
    fig.update_layout(
        xaxis_title="Heartbeat index",
        yaxis_title="R-R interval (ms)",
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.metric("Total Heartbeats", len(res['r_peaks']))
        st.metric("Ectopics Detected", res['ectopic_stats']['count'])
    with col_e2:
        st.metric("Ectopic Burden (%)", f"{res['ectopic_stats']['pct']:.2f} %")
        st.metric("Correction Interpolation", settings['corr_method'].upper() if settings['ectopic_corrected'] else "DISABLED")

# Tab 5: Time-Domain HRV
with tab_hrv_time:
    st.subheader("Time-Domain HRV Parameters")
    
    # Grid of Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean RR Interval", f"{res['time_m']['mean_rr']:.1f} ms")
    c2.metric("Mean Heart Rate", f"{res['time_m']['mean_hr']:.1f} BPM")
    c3.metric("SDNN (Overall HRV)", f"{res['time_m']['sdnn']:.1f} ms")
    c4.metric("RMSSD (Vagal Tone)", f"{res['time_m']['rmssd']:.1f} ms")
    c5.metric("pNN50 (RSA Load)", f"{res['time_m']['pnn50']:.2f} %")
    
    st.markdown("---")
    
    # RR Interval Distribution Histogram (Plotly)
    fig = px.histogram(
        x=res['corrected_rr'], 
        nbins=20,
        title="R-R Interval Probability Distribution Histogram",
        labels={'x': 'RR Interval (ms)'},
        color_discrete_sequence=['#00ACC1'],
        template="plotly_dark"
    )
    fig.update_layout(yaxis_title="Frequency Count", height=380)
    st.plotly_chart(fig, use_container_width=True)

# Tab 6: Frequency-Domain HRV
with tab_hrv_freq:
    st.subheader("Frequency-Domain Autonomic Assessment")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("LF Power (Sympathovagal)", f"{res['freq_m']['lf']:.1f} ms²")
    c2.metric("HF Power (Pure Vagal)", f"{res['freq_m']['hf']:.1f} ms²")
    c3.metric("LF/HF Ratio (Autonomic Balance)", f"{res['freq_m']['ratio']:.3f}")
    
    st.markdown("---")
    
    # Plotly PSD Welch with Shaded bands
    f = res['freq_m']['psd_f']
    pxx = res['freq_m']['psd_p']
    
    fig = go.Figure()
    if len(f) > 0:
        fig.add_trace(go.Scatter(x=f, y=pxx, name="PSD Curve", line=dict(color="#AA00FF", width=2)))
        
        # Shade VLF
        vlf_mask = (f >= 0.00) & (f < 0.04)
        fig.add_trace(go.Scatter(x=f[vlf_mask], y=pxx[vlf_mask], fill='tozeroy', name="VLF", fillcolor="rgba(189, 189, 189, 0.3)", line=dict(color="rgba(0,0,0,0)")))
        
        # Shade LF
        lf_mask = (f >= 0.04) & (f < 0.15)
        fig.add_trace(go.Scatter(x=f[lf_mask], y=pxx[lf_mask], fill='tozeroy', name="LF Sympathovagal", fillcolor="rgba(255, 112, 67, 0.4)", line=dict(color="rgba(0,0,0,0)")))
        
        # Shade HF
        hf_mask = (f >= 0.15) & (f <= 0.40)
        fig.add_trace(go.Scatter(x=f[hf_mask], y=pxx[hf_mask], fill='tozeroy', name="HF Parasympathetic", fillcolor="rgba(38, 166, 154, 0.4)", line=dict(color="rgba(0,0,0,0)")))
        
        fig.update_layout(
            xaxis_range=[0, 0.5],
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power Spectral Density (ms²/Hz)",
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
    st.plotly_chart(fig, use_container_width=True)

# Tab 7: Non-Linear Analysis
with tab_hrv_nonl:
    st.subheader("Non-Linear Poincaré & Complexity Dynamics")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Poincaré SD1 (Vagal)", f"{res['nonl_m']['sd1']:.1f} ms")
    c2.metric("Poincaré SD2 (Symp/Vagal)", f"{res['nonl_m']['sd2']:.1f} ms")
    c3.metric("Sample Entropy (Complexity)", f"{res['nonl_m']['sampen']:.4f}")
    
    st.markdown("---")
    
    # Plotly Poincaré Scatter with SD1/SD2 Axes
    x_p = res['corrected_rr'][:-1]
    y_p = res['corrected_rr'][1:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_p, y=y_p, mode="markers", name="RR Pairs", marker=dict(color="#29B6F6", size=6, opacity=0.7)))
    
    min_val = min(x_p.min(), y_p.min()) - 50
    max_val = max(x_p.max(), y_p.max()) + 50
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], line=dict(color="#EF5350", dash="dash"), name="Line of Identity (y = x)"))
    
    # Plot SD1/SD2 lines
    mean_x = np.mean(x_p)
    mean_y = np.mean(y_p)
    sd1 = res['nonl_m']['sd1']
    sd2 = res['nonl_m']['sd2']
    angle = np.pi / 4
    
    fig.add_trace(go.Scatter(x=[mean_x - sd1 * np.sin(angle), mean_x + sd1 * np.sin(angle)],
                             y=[mean_y + sd1 * np.cos(angle), mean_y - sd1 * np.cos(angle)],
                             line=dict(color="#FF7043", width=3), name=f"SD1: {sd1:.1f} ms"))
                             
    fig.add_trace(go.Scatter(x=[mean_x - sd2 * np.cos(angle), mean_x + sd2 * np.cos(angle)],
                             y=[mean_y - sd2 * np.sin(angle), mean_y + sd2 * np.sin(angle)],
                             line=dict(color="#66BB6A", width=3), name=f"SD2: {sd2:.1f} ms"))
                             
    fig.update_layout(
        xaxis_title="RR_n (ms)",
        yaxis_title="RR_n+1 (ms)",
        template="plotly_dark",
        height=450,
        width=550,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 8: Report Desk
with tab_report:
    st.subheader("Academic Report Compiler")
    st.write("Generate and download comprehensive PDF/DOCX templates compiling clinical analysis details and figures.")
    
    st.markdown("---")
    st.markdown("### 🛡️ Laboratory & Academic Information")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        student_name = st.text_input("Student Name", "Biomedical Student")
    with col_s2:
        student_id = st.text_input("Roll / Registration ID", "BME-2026-09")
        
    st.markdown("---")
    
    # Settings panel summary printout
    st.subheader("Selected Settings Summary")
    st.code(f"""
Student Name:      {student_name}
Student ID:        {student_id}
R-peak Algorithm:  {settings['rpeak_method']}
Ectopic Handling:  {"ENABLED (" + settings['corr_method'] + ")" if settings['ectopic_corrected'] else "DISABLED"}
Welch PSD Setup:   Window: {settings['welch_win_sec']}s, Overlap: {settings['welch_overlap_pct']}%
Filter Cutoffs:    Low cut: {settings['lowcut']}Hz, High cut: {settings['highcut']}Hz
    """)
    
    if st.button("Compile Academic Reports"):
        with st.spinner("Generating plots and compiling report templates..."):
            student_info = {
                'name': student_name,
                'id': student_id
            }
            
            # Temporary directory to write matplotlib static figures
            plots_dir = "temp_report_plots"
            if os.path.exists(plots_dir):
                shutil.rmtree(plots_dir)
            os.makedirs(plots_dir)
            
            # Single file compiler
            if len(sig_files) == 1:
                # Initialize Report Generator
                reporter = ReportGenerator(fs=res['fs'])
                
                # Generate plots
                plot_paths = reporter.generate_all_plots(
                    t=res['t'],
                    raw_sig=res['sig'],
                    filtered_sig=res['sig_smoothed'],
                    r_peaks=res['r_peaks'],
                    rr_raw=res['rr_intervals'],
                    rr_corrected=res['corrected_rr'],
                    ectopic_mask=res['ectopic_mask'],
                    psd_data=res['freq_m'],
                    output_dir=plots_dir
                )
                
                pdf_path = f"ECG_HRV_Report_{student_id}.pdf"
                docx_path = f"ECG_HRV_Report_{student_id}.docx"
                
                reporter.generate_pdf(
                    pdf_path=pdf_path,
                    student_info=student_info,
                    time_metrics=res['time_m'],
                    freq_metrics=res['freq_m'],
                    nonlinear_metrics=res['nonl_m'],
                    ectopic_stats=res['ectopic_stats'],
                    plot_paths=plot_paths,
                    interpretation_text=res['interpretation'],
                    settings=settings
                )
                
                reporter.generate_docx(
                    docx_path=docx_path,
                    student_info=student_info,
                    time_metrics=res['time_m'],
                    freq_metrics=res['freq_m'],
                    nonlinear_metrics=res['nonl_m'],
                    ectopic_stats=res['ectopic_stats'],
                    plot_paths=plot_paths,
                    interpretation_text=res['interpretation'],
                    settings=settings
                )
                
                st.success("Reports successfully compiled!")
                
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button("📥 Download Academic PDF Report", pdf_file, file_name=pdf_path, mime="application/pdf")
                    
                with open(docx_path, "rb") as docx_file:
                    st.download_button("📥 Download Editable DOCX Report", docx_file, file_name=docx_path, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            
            else:
                # Multi-file batch compilation
                zip_filename = f"ECG_HRV_Batch_Reports_{student_id}.zip"
                with zipfile.ZipFile(zip_filename, 'w') as zip_f:
                    for s in sig_files:
                        st.write(f"Compiling reports for: {s['name']}...")
                        s_res = process_record(s, settings)
                        
                        s_reporter = ReportGenerator(fs=s_res['fs'])
                        
                        s_plots_dir = os.path.join(plots_dir, s['name'].replace('.','_'))
                        s_plot_paths = s_reporter.generate_all_plots(
                            t=s_res['t'],
                            raw_sig=s_res['sig'],
                            filtered_sig=s_res['sig_smoothed'],
                            r_peaks=s_res['r_peaks'],
                            rr_raw=s_res['rr_intervals'],
                            rr_corrected=s_res['corrected_rr'],
                            ectopic_mask=s_res['ectopic_mask'],
                            psd_data=s_res['freq_m'],
                            output_dir=s_plots_dir
                        )
                        
                        s_pdf = f"ECG_HRV_Report_{os.path.splitext(s['name'])[0]}.pdf"
                        s_docx = f"ECG_HRV_Report_{os.path.splitext(s['name'])[0]}.docx"
                        
                        s_reporter.generate_pdf(
                            pdf_path=s_pdf,
                            student_info=student_info,
                            time_metrics=s_res['time_m'],
                            freq_metrics=s_res['freq_m'],
                            nonlinear_metrics=s_res['nonl_m'],
                            ectopic_stats=s_res['ectopic_stats'],
                            plot_paths=s_plot_paths,
                            interpretation_text=s_res['interpretation'],
                            settings=settings
                        )
                        
                        s_reporter.generate_docx(
                            docx_path=s_docx,
                            student_info=student_info,
                            time_metrics=s_res['time_m'],
                            freq_metrics=s_res['freq_m'],
                            nonlinear_metrics=s_res['nonl_m'],
                            ectopic_stats=s_res['ectopic_stats'],
                            plot_paths=s_plot_paths,
                            interpretation_text=s_res['interpretation'],
                            settings=settings
                        )
                        
                        zip_f.write(s_pdf)
                        zip_f.write(s_docx)
                        
                        # Clean temp local files after zipping
                        os.remove(s_pdf)
                        os.remove(s_docx)
                        
                st.success("Batch processing complete! Zip archive compiled.")
                with open(zip_filename, "rb") as z_file:
                    st.download_button("📥 Download All Reports (ZIP Archive)", z_file, file_name=zip_filename, mime="application/zip")
                    
            # Clean up matplotlib plots
            if os.path.exists(plots_dir):
                shutil.rmtree(plots_dir)
