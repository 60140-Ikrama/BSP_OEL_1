import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import freqz, butter
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
    page_title="ECG-HRV Analytics Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark medical styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #080C16 !important;
        color: #F8FAFC !important;
    }
    [data-testid="stAppViewContainer"] { background-color: #080C16 !important; }
    [data-testid="stHeader"] { background-color: rgba(8,12,22,0.8) !important; backdrop-filter: blur(12px); }
    [data-testid="stSidebar"] {
        background-color: #0F1423 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #38BDF8 !important; font-weight: 700; }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.75), rgba(15,23,42,0.85));
        border-radius: 12px; padding: 18px 20px;
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 5px solid #38BDF8;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.35);
        backdrop-filter: blur(6px); margin-bottom: 14px;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px 0 rgba(56,189,248,0.15); border-color: rgba(56,189,248,0.3); }
    .metric-title { font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.8px; margin-bottom: 5px; }
    .metric-value { font-size: 2rem; color: #FFFFFF; font-weight: 800; line-height: 1.1; }
    .metric-desc { font-size: 0.72rem; color: #38BDF8; margin-top: 5px; font-weight: 500; }
    .metric-sub { font-size: 0.95rem; color: #CBD5E1; font-weight: 500; margin-top: 2px; }



    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background-color: rgba(15,23,42,0.6); padding: 6px;
        border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 9px 16px; border-radius: 8px; color: #94A3B8 !important;
        font-weight: 600; border: none !important; background-color: transparent !important;
        font-size: 0.82rem; transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #F8FAFC !important; background-color: rgba(255,255,255,0.03) !important; }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important; background-color: rgba(56,189,248,0.12) !important;
        border: 1px solid rgba(56,189,248,0.25) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284C7, #0369A1) !important;
        color: white !important; font-weight: 700 !important; border: none !important;
        border-radius: 8px !important; padding: 12px 24px !important;
        box-shadow: 0 4px 14px rgba(2,132,199,0.4) !important; transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369A1, #075985) !important;
        transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(2,132,199,0.6) !important;
    }

    /* Section Dividers */
    .section-header {
        color: #38BDF8; font-size: 1.15rem; font-weight: 700;
        border-bottom: 2px solid rgba(56,189,248,0.25); padding-bottom: 8px; margin-bottom: 16px;
    }

    /* Data Table Override */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── App Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(90deg, #0B0F19 0%, #172033 100%); padding: 22px 28px;
     border-radius: 15px; border-left: 6px solid #38BDF8; margin-bottom: 22px;
     border: 1px solid rgba(255,255,255,0.05); box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3);">
    <h1 style="color:#FFFFFF; margin:0; font-size:2.0rem; font-weight:800; display:flex; align-items:center; gap:12px;">
        🫀 ECG-HRV Analytics Dashboard
    </h1>
    <p style="color:#94A3B8; margin:7px 0 0 0; font-size:0.95rem; font-weight:500;">
        Biomedical Signal Processing & Heart Rate Variability (HRV) Analysis Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Analysis Settings")

fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=250, min_value=100, max_value=1000)

st.sidebar.subheader("ECG Filter Cutoffs")
lowcut  = st.sidebar.slider("Bandpass Low Cut (Hz)",  0.1, 2.0,  0.5,  step=0.1)
highcut = st.sidebar.slider("Bandpass High Cut (Hz)", 20.0, 100.0, 40.0, step=5.0)

# Live filter Bode plot in sidebar
try:
    nyq = 0.5 * fs
    low_n  = max(0.01, min(lowcut, nyq - 1.0)) / nyq
    high_n = max(low_n + 0.01, min(highcut, nyq - 1.0)) / nyq
    b_bode, a_bode = butter(3, [low_n, high_n], btype='band')
    w_bode, h_bode = freqz(b_bode, a_bode, worN=512)
    freq_hz = w_bode / np.pi * nyq
    mag_db  = 20 * np.log10(np.abs(h_bode) + 1e-12)
    fig_bode = go.Figure()
    fig_bode.add_trace(go.Scatter(
        x=freq_hz, y=mag_db, mode='lines',
        line=dict(color='#38BDF8', width=2), name='|H(f)|'
    ))
    fig_bode.add_vrect(x0=lowcut, x1=highcut, fillcolor="rgba(56,189,248,0.08)",
                       line_width=0, annotation_text="Passband", annotation_position="top left",
                       annotation_font_color="#38BDF8", annotation_font_size=9)
    fig_bode.update_layout(
        title=dict(text="Filter Response (Bode)", font=dict(size=12, color='#94A3B8')),
        xaxis_title="Freq (Hz)", yaxis_title="dB",
        template="plotly_dark",
        margin=dict(l=15, r=10, t=35, b=30), height=190,
        showlegend=False,
        yaxis_range=[-60, 5],
        xaxis_range=[0, min(nyq, 60)]
    )
    st.sidebar.plotly_chart(fig_bode, use_container_width=True)
except Exception:
    pass

st.sidebar.subheader("QRS R-Peak Algorithm")
rpeak_method = st.sidebar.selectbox(
    "Detection Method",
    ["Pan-Tompkins (Custom)", "NeuroKit2 (Default)", "NeuroKit2 (Hamilton)",
     "NeuroKit2 (Elgendi)", "NeuroKit2 (Kalidas)", "NeuroKit2 (Engzee)"]
)

st.sidebar.subheader("Ectopic Beat Handling")
ectopic_corrected = st.sidebar.checkbox("Enable Ectopic Correction", value=True)
ectopic_thresh    = st.sidebar.slider("Outlier Threshold (%)", 10, 40, 20, step=5) / 100.0
corr_method       = st.sidebar.selectbox("Interpolation Type", ["spline", "linear"])

st.sidebar.subheader("Welch PSD Parameters")
welch_win_sec    = st.sidebar.slider("Welch Window (sec)",  16, 128, 64, step=8)
welch_overlap_pct = st.sidebar.slider("Welch Overlap (%)",   0,  90, 50, step=10)

settings = {
    'fs': fs, 'lowcut': lowcut, 'highcut': highcut,
    'rpeak_method': rpeak_method,
    'ectopic_corrected': ectopic_corrected,
    'ectopic_thresh': ectopic_thresh,
    'corr_method': corr_method,
    'welch_win_sec': welch_win_sec,
    'welch_overlap_pct': welch_overlap_pct
}

# ─── File Upload ──────────────────────────────────────────────────────────────
# ─── Data Source Selection ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📂 ECG Data Source")
data_source = st.sidebar.radio(
    "Select Input Source",
    ["Synthetic Generator", "Upload Files"],
    index=0
)

sig_files = []
active_sig_name = None

if data_source == "Synthetic Generator":
    st.sidebar.subheader("Synthetic Generator Settings")
    synth_rhythm = st.sidebar.selectbox(
        "Rhythm Type",
        ["NSR (Normal)", "AFib (Atrial Fibrillation)", "PVC (Ectopic Beats)", "VTach (Ventricular Tachycardia)"]
    )
    synth_duration = st.sidebar.slider("Signal Duration (sec)", 10, 180, 60, step=10)
    
    st.sidebar.markdown("**Noise Levels**")
    noise_wander = st.sidebar.slider("Baseline Wander", 0.0, 0.5, 0.15, step=0.05)
    noise_powerline = st.sidebar.slider("Powerline (50Hz)", 0.0, 0.2, 0.03, step=0.01)
    noise_emg = st.sidebar.slider("EMG Muscle Noise", 0.0, 0.1, 0.02, step=0.01)
    
    # Cache key based on parameters
    params_key = f"synth_{synth_rhythm}_{synth_duration}_{noise_wander}_{noise_powerline}_{noise_emg}_{fs}"
    
    if "synth_params_key" not in st.session_state or st.session_state.synth_params_key != params_key:
        st.session_state.synth_params_key = params_key
        # Generate signal
        generator = SyntheticECGGenerator(fs=fs)
        rhythm_map = {
            "NSR (Normal)": "NSR",
            "AFib (Atrial Fibrillation)": "AFib",
            "PVC (Ectopic Beats)": "PVC",
            "VTach (Ventricular Tachycardia)": "VTach"
        }
        noise_cfg = {
            'baseline_wander': noise_wander,
            'powerline': noise_powerline,
            'emg': noise_emg
        }
        t_syn, sig_syn, _, _ = generator.generate_signal(
            duration_sec=synth_duration,
            rhythm=rhythm_map[synth_rhythm],
            noise_config=noise_cfg
        )
        st.session_state.synth_sig = {
            "name": f"synthetic_{rhythm_map[synth_rhythm].lower()}.csv",
            "t": t_syn,
            "sig": sig_syn,
            "fs": fs
        }
    
    sig_files = [st.session_state.synth_sig]
    active_sig_name = st.session_state.synth_sig["name"]

else: # Upload Files
    uploaded_files = st.sidebar.file_uploader(
        "Upload ECG Files (.csv, .txt, .mat, .dat, .edf)",
        type=["csv", "txt", "mat", "dat", "edf"],
        accept_multiple_files=True
    )
    
    if "uploaded_sig_files" not in st.session_state:
        st.session_state.uploaded_sig_files = []
        
    current_uploaded_names = [f.name for f in uploaded_files] if uploaded_files else []
    cached_names = [s["name"] for s in st.session_state.uploaded_sig_files]
    
    if set(current_uploaded_names) != set(cached_names):
        new_sig_files = []
        for f in uploaded_files:
            # Reuse already parsed file to avoid reloading/reparsing
            existing = next((s for s in st.session_state.uploaded_sig_files if s["name"] == f.name), None)
            if existing:
                new_sig_files.append(existing)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                    tmp.write(f.getbuffer())
                    tmp_path = tmp.name
                try:
                    t, sig, file_fs = load_ecg_file(tmp_path, fs=fs)
                    new_sig_files.append({"name": f.name, "t": t, "sig": sig, "fs": file_fs})
                except Exception as e:
                    st.sidebar.error(f"❌ {f.name}: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        st.session_state.uploaded_sig_files = new_sig_files
        
    sig_files = st.session_state.uploaded_sig_files
    
    if sig_files:
        if "active_sig_name" not in st.session_state or st.session_state.active_sig_name not in [s["name"] for s in sig_files]:
            st.session_state.active_sig_name = sig_files[0]["name"]
            
        active_sig_name = st.sidebar.selectbox(
            "Active Record",
            [s["name"] for s in sig_files],
            index=[s["name"] for s in sig_files].index(st.session_state.active_sig_name)
        )
        st.session_state.active_sig_name = active_sig_name

# ─── Processing Function ──────────────────────────────────────────────────────
def process_record(record, settings):
    record_fs = record.get("fs", settings['fs'])
    processor = ECGProcessor(fs=record_fs)
    analyzer  = HRVAnalyzer(fs_ecg=record_fs)

    sig = record["sig"]
    t   = record["t"]

    sig_clean, baseline = processor.remove_baseline_wander_median(sig)
    sig_filtered = processor.apply_bandpass(sig_clean, lowcut=settings['lowcut'], highcut=settings['highcut'])
    sig_smoothed = processor.remove_noise_savgol(sig_filtered)

    method = settings['rpeak_method']
    if method == "Pan-Tompkins (Custom)":
        r_peaks, stages = processor.pan_tompkins_detector(sig_smoothed)
    else:
        nk_map = {
            "NeuroKit2 (Default)": "neurokit",
            "NeuroKit2 (Hamilton)": "hamilton2002",
            "NeuroKit2 (Elgendi)": "elgendi2010",
            "NeuroKit2 (Kalidas)": "kalidas2016",
            "NeuroKit2 (Engzee)": "engzee2012"
        }
        r_peaks = processor.detect_peaks_nk(sig_smoothed, method=nk_map.get(method, "neurokit"))
        stages = None

    rr_intervals = np.diff(r_peaks) / record_fs * 1000.0

    ectopic_mask = analyzer.detect_ectopic_beats(rr_intervals, threshold_pct=settings['ectopic_thresh'])
    corrected_rr = analyzer.correct_ectopic_beats(rr_intervals, ectopic_mask, method=settings['corr_method']) \
                   if settings['ectopic_corrected'] else rr_intervals.copy()

    ectopic_count = int(np.sum(ectopic_mask))
    ectopic_pct   = (ectopic_count / len(rr_intervals)) * 100.0 if len(rr_intervals) > 0 else 0.0
    ectopic_stats = {'count': ectopic_count, 'total_beats': int(len(r_peaks)), 'pct': ectopic_pct}

    time_m  = analyzer.compute_time_domain(corrected_rr)
    freq_m  = analyzer.compute_frequency_domain(corrected_rr,
                  welch_win_sec=settings['welch_win_sec'],
                  welch_overlap_pct=settings['welch_overlap_pct'])
    nonl_m  = analyzer.compute_nonlinear(corrected_rr)
    interp  = analyzer.generate_clinical_interpretation(time_m, freq_m, nonl_m)

    return {
        't': t, 'sig': sig, 'sig_smoothed': sig_smoothed, 'baseline': baseline,
        'r_peaks': r_peaks, 'rr_intervals': rr_intervals,
        'corrected_rr': corrected_rr, 'ectopic_mask': ectopic_mask,
        'ectopic_stats': ectopic_stats, 'time_m': time_m, 'freq_m': freq_m,
        'nonl_m': nonl_m, 'interpretation': interp, 'stages': stages, 'fs': record_fs
    }

# ─── Process Active Record ────────────────────────────────────────────────────
res = None
if sig_files and active_sig_name:
    active_record = next(s for s in sig_files if s["name"] == active_sig_name)
    try:
        res = process_record(active_record, settings)
    except Exception as e:
        st.error(f"Processing error: {e}")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
(tab_intro, tab_overview, tab_dsp, tab_qrs,
 tab_ectopic, tab_hrv_time, tab_hrv_freq, tab_hrv_nonl, tab_report) = st.tabs([
    "🏠 Introduction",
    "📋 Overview",
    "📈 ECG Filtering",
    "⚡ QRS Detection",
    "🩺 RR & Ectopics",
    "📊 Time-Domain HRV",
    "📈 Frequency HRV",
    "🔬 Non-Linear HRV",
    "🎓 Report Desk"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — INTRO & ACADEMIC OBJECTIVES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_intro:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(15,23,42,0.5), rgba(30,41,59,0.7)); padding: 26px;
         border-radius: 14px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
        <h2 style="color:#38BDF8; margin-top:0; font-weight:800; font-size:1.6rem;">
            🔬 ECG-HRV Analysis System
        </h2>
        <p style="font-size:1.0rem; line-height:1.7; color:#E2E8F0; margin-bottom:0;">
            This platform implements a <b>complete biomedical signal processing pipeline</b> for
            ECG acquisition, multi-stage digital filtering, R-peak detection, RR-interval extraction,
            and comprehensive Heart Rate Variability (HRV) computation across time-domain,
            frequency-domain, and non-linear analysis domains.
        </p>
    </div>
    """, unsafe_allow_html=True)



    # Operational Notice
    st.markdown("""
    <div style="background: rgba(56,189,248,0.05); border-left: 5px solid #38BDF8; padding: 14px 18px;
         border-radius: 6px; border: 1px solid rgba(56,189,248,0.15); margin-top: 10px;">
        <h4 style="color:#38BDF8; margin:0 0 6px 0; font-weight:700;">
            📡 Stationary File Processing Only
        </h4>
        <p style="margin:0; font-size:0.92rem; color:#E2E8F0; line-height:1.5;">
            Live streaming sensor input is removed for strict academic reproducibility.
            All processing is performed on pre-acquired patient datasets or custom ECG files uploaded via the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if res is None:
        st.markdown("---")
        st.info("👈 **Select 'Synthetic Generator' or upload an ECG file in the sidebar to activate the analysis dashboard.**")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-header">📋 Signal Overview &amp; Telemetry Vitals</div>', unsafe_allow_html=True)

    col_plot, col_vitals = st.columns([2.2, 1])
    with col_plot:
        st.caption(f"**Active Record:** `{active_sig_name}`  |  fs = {res['fs']} Hz  |  Duration: {res['t'][-1]:.1f} s  |  Method: {settings['rpeak_method']}")

        zoom_sec = st.slider("Scope window (sec)", min_value=2,
                             max_value=min(30, int(len(res['sig']) / res['fs'])), value=8,
                             key="overview_zoom")
        zoom_idx = res['t'] <= zoom_sec
        fig_ov = go.Figure()
        fig_ov.add_trace(go.Scatter(x=res['t'][zoom_idx], y=res['sig'][zoom_idx],
                                    name="Raw ECG", line=dict(color="#EF5350", width=1), opacity=0.6))
        fig_ov.add_trace(go.Scatter(x=res['t'][zoom_idx], y=res['sig_smoothed'][zoom_idx],
                                    name="Processed ECG", line=dict(color="#00E676", width=1.5)))
        pk_zoom = res['r_peaks'][res['r_peaks'] < int(zoom_sec * res['fs'])]
        fig_ov.add_trace(go.Scatter(x=res['t'][pk_zoom], y=res['sig_smoothed'][pk_zoom],
                                    mode="markers", name="R-Peaks",
                                    marker=dict(color="rgba(0,0,0,0)", size=10, line=dict(color="red", width=2.0))))
        fig_ov.update_xaxes(rangeslider_visible=True)
        fig_ov.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
            template="plotly_dark", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_ov, use_container_width=True)

    with col_vitals:
        st.markdown("**Telemetry Vitals**")
        tm = res['time_m']; fm = res['freq_m']; nm = res['nonl_m']
        vitals = [
            ("#38BDF8",  "Mean Heart Rate",      f"{tm['mean_hr']:.1f}", "BPM",
             f"Mean RR = {tm['mean_rr']:.0f} ms"),
            ("#29B6F6",  "SDNN (Total HRV)",     f"{tm['sdnn']:.1f}", "ms",
             "Overall autonomic variance"),
            ("#66BB6A",  "RMSSD (Vagal Tone)",   f"{tm['rmssd']:.1f}", "ms",
             "Parasympathetic modulation"),
            ("#FFA726",  "LF/HF Ratio",          f"{fm['ratio']:.3f}", "",
             "Sympathovagal balance index"),
            ("#EC407A",  "Poincaré SD1/SD2",     f"{nm['sd1']:.1f}/{nm['sd2']:.1f}", "ms",
             "Short / Long-term HRV"),
            ("#AB47BC",  "Sample Entropy",       f"{nm['sampen']:.3f}", "",
             "Signal complexity index"),
        ]
        for color, title, val, unit, desc in vitals:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}; padding:14px 16px;">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{val}<span style="font-size:0.9rem;color:#94A3B8;"> {unit}</span></div>
                <div class="metric-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Batch summary
    if len(sig_files) > 1:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Batch Processing Summary</div>', unsafe_allow_html=True)
        batch_data = []
        for s in sig_files:
            try:
                sr = process_record(s, settings)
                batch_data.append({
                    "File": s["name"],
                    "HR (BPM)": f"{sr['time_m']['mean_hr']:.1f}",
                    "Mean RR (ms)": f"{sr['time_m']['mean_rr']:.0f}",
                    "SDNN (ms)": f"{sr['time_m']['sdnn']:.1f}",
                    "RMSSD (ms)": f"{sr['time_m']['rmssd']:.1f}",
                    "pNN50 (%)": f"{sr['time_m']['pnn50']:.2f}",
                    "LF (ms²)": f"{sr['freq_m']['lf']:.1f}",
                    "HF (ms²)": f"{sr['freq_m']['hf']:.1f}",
                    "LF/HF": f"{sr['freq_m']['ratio']:.3f}",
                    "SampEn": f"{sr['nonl_m']['sampen']:.3f}",
                    "Ectopics": f"{sr['ectopic_stats']['count']} ({sr['ectopic_stats']['pct']:.1f}%)"
                })
            except Exception as e:
                st.warning(f"Could not process {s['name']}: {e}")
        if batch_data:
            st.dataframe(pd.DataFrame(batch_data), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ECG FILTERING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dsp:
    st.markdown('<div class="section-header">📈 ECG Signal Conditioning — Baseline Removal &amp; Bandpass Filtering</div>', unsafe_allow_html=True)

    col_sig, col_info = st.columns([3, 1.2])
    with col_sig:
        fig_dsp = go.Figure()
        fig_dsp.add_trace(go.Scatter(x=res['t'], y=res['sig'],
                                     name="Raw ECG", line=dict(color="#EF5350", width=1), opacity=0.55))
        fig_dsp.add_trace(go.Scatter(x=res['t'], y=res['baseline'],
                                     name="Baseline Wander (Dual Median)", line=dict(color="#FFFFFF", width=1.2, dash='dot')))
        fig_dsp.add_trace(go.Scatter(x=res['t'], y=res['sig_smoothed'],
                                     name="Filtered ECG (Bandpass + Savitzky-Golay)", line=dict(color="#00E676", width=1.5)))
        fig_dsp.update_xaxes(rangeslider_visible=True)
        fig_dsp.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
            template="plotly_dark", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_dsp, use_container_width=True)

        # Bode plot (large version)
        st.markdown("**Butterworth Bandpass Filter — Frequency Response (Bode Magnitude)**")
        try:
            nyq = 0.5 * res['fs']
            low_n  = max(0.01, min(lowcut,  nyq - 1.0)) / nyq
            high_n = max(low_n + 0.01, min(highcut, nyq - 1.0)) / nyq
            b2, a2 = butter(3, [low_n, high_n], btype='band')
            w2, h2 = freqz(b2, a2, worN=1024)
            fhz2   = w2 / np.pi * nyq
            mdb2   = 20 * np.log10(np.abs(h2) + 1e-12)
            fig_bode2 = go.Figure()
            fig_bode2.add_trace(go.Scatter(x=fhz2, y=mdb2, line=dict(color="#38BDF8", width=2.5), name="|H(f)| dB"))
            fig_bode2.add_vrect(x0=lowcut, x1=highcut, fillcolor="rgba(56,189,248,0.08)",
                                annotation_text=f"Passband {lowcut}–{highcut} Hz",
                                annotation_position="top left",
                                annotation_font_color="#38BDF8", annotation_font_size=11, line_width=0)
            fig_bode2.add_hline(y=-3, line_dash="dash", line_color="#FF7043",
                                annotation_text="-3 dB cutoff", annotation_position="right",
                                annotation_font_color="#FF7043")
            fig_bode2.update_layout(
                xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)",
                template="plotly_dark", height=280, yaxis_range=[-60, 5],
                xaxis_range=[0, min(nyq, 80)],
                margin=dict(l=20, r=20, t=10, b=20)
            )
            st.plotly_chart(fig_bode2, use_container_width=True)
        except Exception as e:
            st.warning(f"Filter response unavailable: {e}")

    with col_info:
        st.info(f"""
**Filter Design Parameters**

- **Baseline Wander Removal**
  - Method: Dual Median Filter
  - Window 1: 200ms (removes QRS/T-waves)
  - Window 2: 600ms (removes P-waves)
  - Subtracted from raw signal

- **Bandpass Filter**
  - Type: Butterworth, Order 3
  - Low Cut: **{settings['lowcut']:.1f} Hz**
  - High Cut: **{settings['highcut']:.1f} Hz**
  - Applied with `filtfilt` (zero phase shift)

- **Smoothing**
  - Method: Savitzky-Golay
  - Window: 80ms polynomial fit
  - Preserves R-peak amplitudes
        """)
        st.markdown("**Signal Quality Metrics**")
        snr_raw = float(np.std(res['sig_smoothed']) / (np.std(res['sig'] - res['sig_smoothed']) + 1e-6))
        st.metric("Signal-to-Noise Ratio", f"{snr_raw:.2f}")
        st.metric("Signal Length",  f"{res['t'][-1]:.1f} s")
        st.metric("Sampling Rate",  f"{res['fs']} Hz")
        st.metric("Total Samples",  f"{len(res['sig'])}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QRS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_qrs:
    st.markdown('<div class="section-header">⚡ QRS Complex Detection — R-Peak Extraction</div>', unsafe_allow_html=True)

    col_qrs, col_qinfo = st.columns([3, 1.2])
    with col_qrs:
        if res['stages'] is not None:
            p_zoom = res['t'] <= 8.0
            t_zoom = res['t'][p_zoom]
            fig_qrs = go.Figure()
            fig_qrs.add_trace(go.Scatter(x=t_zoom, y=res['stages']['filtered'][p_zoom],
                                          name="1. Bandpass 5–15Hz", line=dict(color="#EC407A", width=1.5)))
            fig_qrs.add_trace(go.Scatter(x=t_zoom, y=res['stages']['derived'][p_zoom],
                                          name="2. Derivative", line=dict(color="#AB47BC", width=1.5)))
            fig_qrs.add_trace(go.Scatter(x=t_zoom, y=res['stages']['squared'][p_zoom],
                                          name="3. Squaring", line=dict(color="#42A5F5", width=1.5)))
            fig_qrs.add_trace(go.Scatter(x=t_zoom, y=res['stages']['integrated'][p_zoom],
                                          name="4. MWI Envelope", line=dict(color="#26A69A", width=2.5)))
            z_peaks = res['r_peaks'][res['r_peaks'] < int(8.0 * res['fs'])]
            fig_qrs.add_trace(go.Scatter(
                x=res['t'][z_peaks], y=res['stages']['integrated'][z_peaks],
                mode="markers", name="Detected R-Peaks",
                marker=dict(color="#FFD600", size=12, symbol="x", line=dict(color="black", width=1.5))
            ))
            fig_qrs.update_xaxes(rangeslider_visible=True)
            fig_qrs.update_layout(
                title="Pan-Tompkins Algorithm — Signal Processing Stages (First 8s)",
                xaxis_title="Time (s)", yaxis_title="Normalized Amplitude",
                template="plotly_dark", height=460,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_qrs, use_container_width=True)
        else:
            st.warning(f"Stage visualization is only available for Pan-Tompkins method. Current: **{settings['rpeak_method']}**")
            # Still show the clean ECG with R-peaks
            p_zoom = res['t'] <= 10.0
            fig_qrs2 = go.Figure()
            fig_qrs2.add_trace(go.Scatter(x=res['t'][p_zoom], y=res['sig_smoothed'][p_zoom],
                                           name="Processed ECG", line=dict(color="#00E676", width=1.5)))
            pk_10 = res['r_peaks'][res['r_peaks'] < int(10.0 * res['fs'])]
            fig_qrs2.add_trace(go.Scatter(
                x=res['t'][pk_10], y=res['sig_smoothed'][pk_10],
                mode="markers", name="R-Peaks",
                marker=dict(color="rgba(0,0,0,0)", size=10, line=dict(color="red", width=2.0))
            ))
            fig_qrs2.update_xaxes(rangeslider_visible=True)
            fig_qrs2.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
                                    template="plotly_dark", height=400,
                                    margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_qrs2, use_container_width=True)

    with col_qinfo:
        total_beats = len(res['r_peaks'])
        duration    = res['t'][-1]
        mean_hr     = res['time_m']['mean_hr']
        st.info(f"""
**Pan-Tompkins Algorithm Steps**

1. **Bandpass Filter (5–15 Hz)**
   Isolates QRS energy band

2. **Differentiation**
   `y[n] = (2x[n]+x[n-1]-x[n-3]-2x[n-4])/8`
   Highlights steep QRS slope

3. **Squaring**
   `y[n] = x[n]²`
   Amplifies QRS, all-positive

4. **Moving Window Integration (150ms)**
   Produces smooth QRS envelope

5. **Adaptive Thresholding**
   Signal/noise peak tracking
   with 200ms refractory period
        """)
        st.markdown("**Detection Statistics**")
        st.metric("Total R-Peaks",   total_beats)
        st.metric("Recording Time",  f"{duration:.1f} s")
        st.metric("Mean Heart Rate", f"{mean_hr:.1f} BPM")
        expected_beats = int(mean_hr / 60 * duration)
        detection_rate = min(100.0, total_beats / max(expected_beats, 1) * 100)
        st.metric("Detection Rate",  f"{detection_rate:.1f} %")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RR & ECTOPIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ectopic:
    st.markdown('<div class="section-header">🩺 RR Tachogram &amp; Ectopic Beat Correction</div>', unsafe_allow_html=True)

    col_tach, col_einfo = st.columns([3, 1.2])
    with col_tach:
        x_idx = np.arange(len(res['rr_intervals']))
        fig_rr = go.Figure()
        fig_rr.add_trace(go.Scatter(x=x_idx, y=res['rr_intervals'],
                                    name="Raw RR Intervals", mode="lines+markers",
                                    line=dict(color="#EF5350", width=1.2, dash="dash"),
                                    marker=dict(size=3)))
        if settings['ectopic_corrected']:
            fig_rr.add_trace(go.Scatter(x=x_idx, y=res['corrected_rr'],
                                        name="Corrected RR (Cubic Spline)", mode="lines+markers",
                                        line=dict(color="#00E676", width=1.5),
                                        marker=dict(size=3)))
        ect_idx = np.where(res['ectopic_mask'])[0]
        if len(ect_idx) > 0:
            fig_rr.add_trace(go.Scatter(x=ect_idx, y=res['rr_intervals'][ect_idx],
                                        mode="markers", name="Ectopic Beats (Flagged)",
                                        marker=dict(color="#FF1744", size=11, symbol="x")))
        fig_rr.update_xaxes(rangeslider_visible=True)
        fig_rr.update_layout(
            xaxis_title="Beat Index (n)", yaxis_title="RR Interval (ms)",
            template="plotly_dark", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_rr, use_container_width=True)

        # Beat-by-beat RR table (first 40 beats)
        st.markdown("**Beat-by-Beat RR Interval Table (First 40 Beats)**")
        n_show = min(40, len(res['rr_intervals']))
        rr_table = pd.DataFrame({
            "Beat #":         np.arange(1, n_show + 1),
            "Raw RR (ms)":    np.round(res['rr_intervals'][:n_show], 1),
            "Corrected (ms)": np.round(res['corrected_rr'][:n_show], 1),
            "ΔRR (ms)":       np.concatenate([[0], np.round(np.diff(res['corrected_rr'][:n_show]), 1)]),
            "Ectopic":        ["⚠️ YES" if res['ectopic_mask'][i] else "✓ Normal" for i in range(n_show)]
        })
        st.dataframe(rr_table, use_container_width=True, height=230)

    with col_einfo:
        st.info(f"""
**Ectopic Detection Method**

- **Physiological Limits**: Flag RR < 300ms or > 2000ms
- **Rolling Median Window**: 15-beat sliding window
- **Deviation Threshold**: >{settings['ectopic_thresh']*100:.0f}% from local median
- **MAD Check**: ±3.5× Global Median Absolute Deviation
        """)
        st.markdown("**Correction Statistics**")
        es = res['ectopic_stats']
        st.metric("Total Beats",      es['total_beats'])
        st.metric("Ectopics Flagged", es['count'])
        st.metric("Ectopic Burden",   f"{es['pct']:.2f} %")
        st.metric("Correction Mode",  settings['corr_method'].upper() if settings['ectopic_corrected'] else "DISABLED")
        if len(res['rr_intervals']) > 0:
            st.metric("RR Range",     f"{res['rr_intervals'].min():.0f}–{res['rr_intervals'].max():.0f} ms")
            st.metric("RR Std Dev",   f"{res['rr_intervals'].std():.1f} ms")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TIME-DOMAIN HRV
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hrv_time:
    st.markdown('<div class="section-header">📊 Time-Domain HRV Analysis</div>', unsafe_allow_html=True)

    tm = res['time_m']
    cvnn = (tm['sdnn'] / tm['mean_rr'] * 100) if tm['mean_rr'] > 0 else 0.0

    # Metric cards row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Mean RR",   f"{tm['mean_rr']:.1f} ms",  help="Mean of all NN intervals")
    c2.metric("Mean HR",   f"{tm['mean_hr']:.1f} BPM", help="60000/Mean_RR")
    c3.metric("SDNN",      f"{tm['sdnn']:.1f} ms",     help="Std Dev of all NN intervals")
    c4.metric("RMSSD",     f"{tm['rmssd']:.1f} ms",    help="Root Mean Square Successive Differences")
    c5.metric("NN50",      f"{tm['nn50']}",             help="Count of |ΔRR| > 50 ms")
    c6.metric("pNN50",     f"{tm['pnn50']:.2f} %",     help="NN50/total × 100")

    c7, c8 = st.columns([1, 5])
    c7.metric("CVNN (%)", f"{cvnn:.2f}", help="SDNN/Mean_RR × 100 — Coefficient of Variation")

    st.markdown("---")
    col_hist, col_math = st.columns([2.5, 1.5])

    with col_hist:
        # RR Distribution Histogram
        fig_hist = px.histogram(
            x=res['corrected_rr'], nbins=25,
            title="RR Interval Distribution",
            labels={'x': 'RR Interval (ms)', 'y': 'Count'},
            color_discrete_sequence=['#00ACC1'],
            template="plotly_dark"
        )
        fig_hist.add_vline(x=tm['mean_rr'], line_dash="dash", line_color="#FFD600",
                           annotation_text=f"Mean = {tm['mean_rr']:.0f} ms",
                           annotation_font_color="#FFD600")
        fig_hist.update_layout(yaxis_title="Frequency Count", height=340,
                               margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

        # Successive differences plot
        diff_rr = np.diff(res['corrected_rr'])
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(x=np.arange(len(diff_rr)), y=diff_rr,
                                  marker_color=["#FF1744" if abs(d) > 50 else "#00E676" for d in diff_rr],
                                  name="|ΔRR|"))
        fig_diff.add_hline(y=50,  line_dash="dash", line_color="#FFA726", annotation_text="+50ms")
        fig_diff.add_hline(y=-50, line_dash="dash", line_color="#FFA726", annotation_text="−50ms")
        fig_diff.update_layout(
            title="Successive RR Differences (ΔRR[n] = RR[n+1] − RR[n])",
            xaxis_title="Beat Index", yaxis_title="ΔRR (ms)",
            template="plotly_dark", height=260,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_diff, use_container_width=True)

    with col_math:
        st.markdown("### 📐 Statistical Formulations")

        st.markdown("**SDNN**")
        st.latex(r"\text{SDNN} = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (RR_i - \overline{RR})^2}")
        st.caption("Total autonomic variance — both sympathetic & parasympathetic.")

        st.markdown("**RMSSD**")
        st.latex(r"\text{RMSSD} = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N-1}(RR_{i+1}-RR_i)^2}")
        st.caption("High-frequency parasympathetic vagal tone index.")

        st.markdown("**pNN50**")
        st.latex(r"\text{pNN50} = \frac{\sum \mathbf{1}[|\Delta RR_i|>50\text{ ms}]}{N-1}\times100\%")
        st.caption(f"Current: {tm['pnn50']:.2f}% ({tm['nn50']} beats > 50ms)")

        st.markdown("**CVNN (Coefficient of Variation)**")
        st.latex(r"\text{CVNN} = \frac{\text{SDNN}}{\overline{RR}} \times 100\%")
        st.caption(f"Current: {cvnn:.2f}% — normalises SDNN for heart rate.")

        # Clinical reference table
        st.markdown("**📋 Normal Reference Ranges**")
        ref_df = pd.DataFrame({
            "Metric": ["SDNN", "RMSSD", "pNN50", "LF/HF"],
            "Normal Range": ["50–100 ms", "20–50 ms", "5–25%", "0.5–2.0"],
            "Interpretation": ["Total HRV", "Vagal tone", "Parasympathetic", "Balance"]
        })
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FREQUENCY-DOMAIN HRV
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hrv_freq:
    st.markdown('<div class="section-header">📈 Frequency-Domain HRV — Welch Power Spectral Density</div>', unsafe_allow_html=True)

    fm = res['freq_m']
    total_p = fm.get('total_power', fm['vlf'] + fm['lf'] + fm['hf'])

    # Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("VLF Power",    f"{fm['vlf']:.1f} ms²",  help="Very Low Freq: 0–0.04 Hz")
    c2.metric("LF Power",     f"{fm['lf']:.1f} ms²",   help="Low Freq: 0.04–0.15 Hz (Sympathovagal)")
    c3.metric("HF Power",     f"{fm['hf']:.1f} ms²",   help="High Freq: 0.15–0.40 Hz (Pure Vagal)")
    c4.metric("LF/HF Ratio",  f"{fm['ratio']:.3f}",    help="Sympathovagal balance index")
    c5.metric("Total Power",  f"{total_p:.1f} ms²",    help="VLF + LF + HF")

    c6, c7 = st.columns(2)
    c6.metric("LF (n.u.)", f"{fm['lf_nu']:.1f} %", help="LF / (Total − VLF) × 100")
    c7.metric("HF (n.u.)", f"{fm['hf_nu']:.1f} %", help="HF / (Total − VLF) × 100")

    st.markdown("---")
    col_psd, col_fmath = st.columns([2.5, 1.5])

    with col_psd:
        f_arr = fm['psd_f']
        pxx   = fm['psd_p']
        fig_psd = go.Figure()
        if len(f_arr) > 0:
            fig_psd.add_trace(go.Scatter(x=f_arr, y=pxx, name="PSD (Welch)", line=dict(color="#AA00FF", width=2)))
            vlf_m = (f_arr >= 0.00) & (f_arr < 0.04)
            lf_m  = (f_arr >= 0.04) & (f_arr < 0.15)
            hf_m  = (f_arr >= 0.15) & (f_arr <= 0.40)
            fig_psd.add_trace(go.Scatter(x=f_arr[vlf_m], y=pxx[vlf_m], fill='tozeroy',
                                         name=f"VLF ({fm['vlf']:.1f} ms²)", fillcolor="rgba(189,189,189,0.3)",
                                         line=dict(color="rgba(0,0,0,0)")))
            fig_psd.add_trace(go.Scatter(x=f_arr[lf_m], y=pxx[lf_m], fill='tozeroy',
                                         name=f"LF ({fm['lf']:.1f} ms²)", fillcolor="rgba(255,112,67,0.4)",
                                         line=dict(color="rgba(0,0,0,0)")))
            fig_psd.add_trace(go.Scatter(x=f_arr[hf_m], y=pxx[hf_m], fill='tozeroy',
                                         name=f"HF ({fm['hf']:.1f} ms²)", fillcolor="rgba(38,166,154,0.4)",
                                         line=dict(color="rgba(0,0,0,0)")))
            # Band boundary lines
            for fx, label in [(0.04, "VLF|LF"), (0.15, "LF|HF"), (0.40, "HF limit")]:
                fig_psd.add_vline(x=fx, line_dash="dot", line_color="rgba(255,255,255,0.25)",
                                  annotation_text=f"{fx} Hz", annotation_font_size=10,
                                  annotation_font_color="#94A3B8")
        fig_psd.update_xaxes(rangeslider_visible=True, range=[0, 0.5])
        fig_psd.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="PSD (ms²/Hz)",
            template="plotly_dark", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_psd, use_container_width=True)

        # Band power comparison bar chart
        fig_bar = go.Figure(go.Bar(
            x=["VLF\n0–0.04Hz", "LF\n0.04–0.15Hz", "HF\n0.15–0.40Hz"],
            y=[fm['vlf'], fm['lf'], fm['hf']],
            marker_color=["#9E9E9E", "#FF7043", "#26A69A"],
            text=[f"{fm['vlf']:.1f}", f"{fm['lf']:.1f}", f"{fm['hf']:.1f}"],
            textposition="outside"
        ))
        fig_bar.update_layout(
            title="Frequency Band Power Comparison (ms²)",
            yaxis_title="Power (ms²)", template="plotly_dark",
            height=240, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_fmath:
        st.markdown("### 📐 Spectral Formulations")

        st.markdown("**Welch Periodogram**")
        st.latex(r"P_{xx}(f)=\frac{1}{K}\sum_{k=1}^K\left|\frac{1}{N}\sum_{n=0}^{N-1}x_k(n)w(n)e^{-j2\pi fn}\right|^2")
        st.caption(f"RR series resampled at 4 Hz · {settings['welch_win_sec']}s window · {settings['welch_overlap_pct']}% overlap")

        st.markdown("**Normalized Power Units (n.u.)**")
        st.latex(r"\text{LF}_{nu}=\frac{P_{LF}}{P_{total}-P_{VLF}}\times100")
        st.caption("Removes slow vasomotor VLF component for cross-subject comparison.")

        st.markdown("**Autonomic Balance**")
        st.latex(r"\text{LF/HF}=\frac{P_{LF}}{P_{HF}}")

        # Full Power Table
        st.markdown("**📋 Frequency Band Summary**")
        pct_total = total_p if total_p > 0 else 1e-9
        band_df = pd.DataFrame({
            "Band":   ["VLF", "LF", "HF", "Total"],
            "Range (Hz)": ["0–0.04", "0.04–0.15", "0.15–0.40", "0–0.40"],
            "Power (ms²)": [f"{fm['vlf']:.2f}", f"{fm['lf']:.2f}", f"{fm['hf']:.2f}", f"{total_p:.2f}"],
            "% Total": [f"{fm['vlf']/pct_total*100:.1f}%", f"{fm['lf']/pct_total*100:.1f}%",
                        f"{fm['hf']/pct_total*100:.1f}%", "100%"],
            "n.u.": ["—", f"{fm['lf_nu']:.1f}", f"{fm['hf_nu']:.1f}", "—"]
        })
        st.dataframe(band_df, use_container_width=True, hide_index=True)

        # Clinical significance
        st.markdown("**Physiological Significance**")
        ratio_val = fm['ratio']
        if ratio_val > 2.0:
            sym_msg = "🔴 Sympathetic dominance"
        elif ratio_val < 0.5:
            sym_msg = "🟢 Parasympathetic dominance"
        else:
            sym_msg = "🟡 Balanced autonomic state"
        st.info(f"LF/HF = **{ratio_val:.3f}** → {sym_msg}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — NON-LINEAR HRV
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hrv_nonl:
    st.markdown('<div class="section-header">🔬 Non-Linear HRV — Poincaré Plot &amp; Complexity Analysis</div>', unsafe_allow_html=True)

    nm = res['nonl_m']

    # Metric row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Poincaré SD1",  f"{nm['sd1']:.2f} ms",   help="Short-term HRV (perpendicular to line of identity)")
    c2.metric("Poincaré SD2",  f"{nm['sd2']:.2f} ms",   help="Long-term HRV (along line of identity)")
    c3.metric("SD1/SD2 Ratio", f"{nm['ratio']:.4f}",    help="Parasympathetic/Total HRV ratio")
    c4.metric("Sample Entropy",f"{nm['sampen']:.4f}",   help="Signal complexity (higher = healthier)")

    st.markdown("---")
    col_sc, col_nmath = st.columns([2.2, 1.8])

    with col_sc:
        x_p = res['corrected_rr'][:-1]
        y_p = res['corrected_rr'][1:]
        mean_x = np.mean(x_p); mean_y = np.mean(y_p)
        sd1 = nm['sd1']; sd2 = nm['sd2']
        min_v = min(mean_x - sd2 - 100, x_p.min() - 100)
        max_v = max(mean_x + sd2 + 100, x_p.max() + 100)

        fig_poi = go.Figure()
        # RR points plotted as blue squares
        fig_poi.add_trace(go.Scatter(x=x_p, y=y_p, mode="markers", name="RR points",
                                     marker=dict(color="blue", symbol="square", size=5)))
        # Identity line (black/gray dashed line)
        fig_poi.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                     line=dict(color="#888888", dash="dash", width=1.0),
                                     name="Identity line"))
        
        # Rotated ellipse (red line)
        theta = np.linspace(0, 2*np.pi, 200)
        cos_ang, sin_ang = np.cos(np.pi/4), np.sin(np.pi/4)
        x_ellipse = mean_x + sd2 * np.cos(theta) * cos_ang - sd1 * np.sin(theta) * sin_ang
        y_ellipse = mean_y + sd2 * np.cos(theta) * sin_ang + sd1 * np.sin(theta) * cos_ang
        fig_poi.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode="lines",
                                     line=dict(color="red", width=2.0), name="Ellipse"))
        
        ang = np.pi / 4
        # SD1 axis (green line)
        fig_poi.add_trace(go.Scatter(
            x=[mean_x - sd1*np.sin(ang), mean_x + sd1*np.sin(ang)],
            y=[mean_y + sd1*np.cos(ang), mean_y - sd1*np.cos(ang)],
            line=dict(color="green", width=2.0), name="SD1 axis"))
        # SD2 axis (magenta line)
        fig_poi.add_trace(go.Scatter(
            x=[mean_x - sd2*np.cos(ang), mean_x + sd2*np.cos(ang)],
            y=[mean_y - sd2*np.sin(ang), mean_y + sd2*np.sin(ang)],
            line=dict(color="magenta", width=2.0), name="SD2 axis"))
        
        fig_poi.update_layout(
            title="Poincaré Plot",
            xaxis_title="RR(n) (ms)", yaxis_title="RR(n+1) (ms)",
            template="plotly_dark", height=460,
            xaxis=dict(range=[min_v, max_v]),
            yaxis=dict(range=[min_v, max_v]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_poi, use_container_width=True)

        # SampEn interpretation
        sampen_v = nm['sampen']
        if sampen_v > 1.2:
            comp_label = "🟢 High complexity — healthy adaptive variability"
        elif sampen_v > 0.7:
            comp_label = "🟡 Moderate complexity — normal variability"
        else:
            comp_label = "🔴 Low complexity — possible rigidity or pathology"
        st.info(f"SampEn = **{sampen_v:.4f}** → {comp_label}")

    with col_nmath:
        st.markdown("### 📐 Non-Linear Formulations")

        st.markdown("**Poincaré Geometry (SD1 & SD2)**")
        st.latex(r"\text{SD1}^2=\frac{1}{2}\text{Var}(RR_n-RR_{n+1})")
        st.latex(r"\text{SD2}^2=2\,\text{Var}(RR)-\text{SD1}^2")
        st.caption("SD1 ← parasympathetic (fast RSA); SD2 ← combined long-term regulation.")

        st.markdown("**SD1/SD2 Ratio**")
        st.latex(r"\text{ratio}=\frac{\text{SD1}}{\text{SD2}}")
        st.caption("High ratio → parasympathetic dominance; Low ratio → sympathetic dominance.")

        st.markdown("**Sample Entropy SampEn(m, r, N)**")
        st.latex(r"\text{SampEn}=-\ln\!\left[\frac{A}{B}\right]")
        st.caption("m=2, r=0.2σ. A = templates matching at m+1; B = at m. Excludes self-matches.")

        # Complexity summary table
        st.markdown("**📋 Non-Linear Complexity Summary**")
        comp_df = pd.DataFrame({
            "Metric":     ["SD1", "SD2", "SD1/SD2", "SampEn"],
            "Value":      [f"{nm['sd1']:.2f} ms", f"{nm['sd2']:.2f} ms", f"{nm['ratio']:.4f}", f"{nm['sampen']:.4f}"],
            "Represents": ["Short-term HRV", "Long-term HRV", "Parasympathetic/Total", "Complexity"],
            "Normal":     ["> 20ms", "> 30ms", "0.2–0.5", "0.7–2.0"]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — REPORT DESK
# ═══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown('<div class="section-header">📄 Report Compiler</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        student_name = st.text_input("Analyst Name", "ECG Analyst")
    with col_s2:
        student_id = st.text_input("Record / Reference ID", "REC-2026-09")

    # Settings summary
    st.markdown("**Analysis Configuration**")
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    col_cfg1.info(f"**Algorithm**: {settings['rpeak_method']}\n\n**Filter**: {settings['lowcut']}–{settings['highcut']} Hz Butterworth")
    col_cfg2.info(f"**Ectopic**: {'ON — ' + settings['corr_method'] if settings['ectopic_corrected'] else 'Disabled'}\n\n**Threshold**: {settings['ectopic_thresh']*100:.0f}%")
    col_cfg3.info(f"**Welch Window**: {settings['welch_win_sec']}s\n\n**Overlap**: {settings['welch_overlap_pct']}%")

    # Clinical summary
    st.markdown("**📋 Full HRV Results Summary**")
    tm = res['time_m']; fm = res['freq_m']; nm = res['nonl_m']
    summary_df = pd.DataFrame([
        ("DOMAIN", "METRIC", "VALUE", "UNIT", "CLINICAL SIGNIFICANCE"),
        ("Time",  "Mean RR",    f"{tm['mean_rr']:.1f}", "ms",  "Average cardiac cycle length"),
        ("Time",  "Mean HR",    f"{tm['mean_hr']:.1f}", "BPM", "Heart rate (beats/min)"),
        ("Time",  "SDNN",       f"{tm['sdnn']:.1f}",   "ms",  "Total HRV / autonomic variance"),
        ("Time",  "RMSSD",      f"{tm['rmssd']:.1f}",  "ms",  "Parasympathetic vagal modulation"),
        ("Time",  "NN50",       f"{tm['nn50']}",        "count","Successive differences > 50ms"),
        ("Time",  "pNN50",      f"{tm['pnn50']:.2f}",  "%",   "Percentage NN50"),
        ("Freq",  "VLF Power",  f"{fm['vlf']:.2f}",    "ms²", "Very low frequency (0–0.04 Hz)"),
        ("Freq",  "LF Power",   f"{fm['lf']:.2f}",     "ms²", "Low freq — sympathovagal (0.04–0.15 Hz)"),
        ("Freq",  "HF Power",   f"{fm['hf']:.2f}",     "ms²", "High freq — vagal (0.15–0.40 Hz)"),
        ("Freq",  "LF/HF",      f"{fm['ratio']:.3f}",  "—",   "Sympathovagal balance"),
        ("Freq",  "LF n.u.",    f"{fm['lf_nu']:.1f}",  "%",   "Normalized LF power"),
        ("Freq",  "HF n.u.",    f"{fm['hf_nu']:.1f}",  "%",   "Normalized HF power"),
        ("NL",    "SD1",        f"{nm['sd1']:.2f}",    "ms",  "Short-term HRV (Poincaré)"),
        ("NL",    "SD2",        f"{nm['sd2']:.2f}",    "ms",  "Long-term HRV (Poincaré)"),
        ("NL",    "SD1/SD2",    f"{nm['ratio']:.4f}",  "—",   "Parasympathetic-to-total ratio"),
        ("NL",    "SampEn",     f"{nm['sampen']:.4f}", "—",   "Signal complexity index"),
    ], columns=["Domain", "Metric", "Value", "Unit", "Clinical Significance"])
    summary_df = summary_df.iloc[1:]  # remove header row which is already column names
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("**Automated Clinical Interpretation**")
    st.markdown(res['interpretation'])

    st.markdown("---")
    if st.button("📄 Compile Reports (PDF + DOCX)"):
        with st.spinner("Generating figures and compiling report…"):
            student_info = {'name': student_name, 'id': student_id}
            plots_dir = "temp_report_plots"
            if os.path.exists(plots_dir):
                shutil.rmtree(plots_dir)
            os.makedirs(plots_dir)

            if len(sig_files) == 1:
                reporter = ReportGenerator(fs=res['fs'])
                plot_paths = reporter.generate_all_plots(
                    t=res['t'], raw_sig=res['sig'], filtered_sig=res['sig_smoothed'],
                    r_peaks=res['r_peaks'], rr_raw=res['rr_intervals'],
                    rr_corrected=res['corrected_rr'], ectopic_mask=res['ectopic_mask'],
                    psd_data=res['freq_m'], output_dir=plots_dir
                )
                pdf_path  = f"ECG_HRV_Report_{student_id}.pdf"
                docx_path = f"ECG_HRV_Report_{student_id}.docx"
                reporter.generate_pdf(pdf_path, student_info, res['time_m'], res['freq_m'],
                                      res['nonl_m'], res['ectopic_stats'], plot_paths,
                                      res['interpretation'], settings)
                reporter.generate_docx(docx_path, student_info, res['time_m'], res['freq_m'],
                                       res['nonl_m'], res['ectopic_stats'], plot_paths,
                                       res['interpretation'], settings)
                st.success("✅ Reports compiled successfully!")
                with open(pdf_path,  "rb") as f:
                    st.download_button("📥 Download PDF Report",  f, file_name=pdf_path,  mime="application/pdf")
                with open(docx_path, "rb") as f:
                    st.download_button("📥 Download DOCX Report", f, file_name=docx_path, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            else:
                zip_fn = f"ECG_HRV_Batch_{student_id}.zip"
                with zipfile.ZipFile(zip_fn, 'w') as zf:
                    for s in sig_files:
                        st.write(f"Compiling: {s['name']}…")
                        sr = process_record(s, settings)
                        sr_rep = ReportGenerator(fs=sr['fs'])
                        sp_dir = os.path.join(plots_dir, s['name'].replace('.', '_'))
                        sp = sr_rep.generate_all_plots(
                            t=sr['t'], raw_sig=sr['sig'], filtered_sig=sr['sig_smoothed'],
                            r_peaks=sr['r_peaks'], rr_raw=sr['rr_intervals'],
                            rr_corrected=sr['corrected_rr'], ectopic_mask=sr['ectopic_mask'],
                            psd_data=sr['freq_m'], output_dir=sp_dir
                        )
                        s_pdf  = f"ECG_HRV_{os.path.splitext(s['name'])[0]}.pdf"
                        s_docx = f"ECG_HRV_{os.path.splitext(s['name'])[0]}.docx"
                        sr_rep.generate_pdf(s_pdf,  student_info, sr['time_m'], sr['freq_m'],
                                            sr['nonl_m'], sr['ectopic_stats'], sp,
                                            sr['interpretation'], settings)
                        sr_rep.generate_docx(s_docx, student_info, sr['time_m'], sr['freq_m'],
                                             sr['nonl_m'], sr['ectopic_stats'], sp,
                                             sr['interpretation'], settings)
                        zf.write(s_pdf); zf.write(s_docx)
                        os.remove(s_pdf); os.remove(s_docx)
                st.success("✅ Batch reports compiled!")
                with open(zip_fn, "rb") as f:
                    st.download_button("📥 Download All Reports (ZIP)", f, file_name=zip_fn, mime="application/zip")

            if os.path.exists(plots_dir):
                shutil.rmtree(plots_dir)
