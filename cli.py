import os
import argparse
import json
import numpy as np
import pandas as pd
from core.synthetic_data import SyntheticECGGenerator
from core.signal_processing import ECGProcessor
from core.hrv_analysis import HRVAnalyzer
from core.report_generator import ReportGenerator

def generate_test_data(output_dir, fs=250, duration=60):
    """
    Helper function to generate multiple synthetic ECG files to simulate a multi-file analysis ward.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    generator = SyntheticECGGenerator(fs=fs)
    rhythms = ['NSR', 'AFib', 'PVC', 'VTach']
    
    # Noise configs
    noises = {
        'NSR': {'baseline_wander': 0.1, 'powerline': 0.02, 'emg': 0.01},
        'AFib': {'baseline_wander': 0.15, 'powerline': 0.05, 'emg': 0.03},
        'PVC': {'baseline_wander': 0.20, 'powerline': 0.06, 'emg': 0.04}, # PVC has higher noise
        'VTach': {'baseline_wander': 0.05, 'powerline': 0.01, 'emg': 0.01}
    }
    
    generated_files = []
    print(f"--- Generating Synthetic ECG Files (fs={fs}Hz, duration={duration}s) ---")
    for rhythm in rhythms:
        filename = f"ecg_{rhythm.lower()}_raw.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Generate
        t, sig, true_peaks, ectopics = generator.generate_signal(
            duration_sec=duration,
            rhythm=rhythm,
            noise_config=noises[rhythm]
        )
        
        # Save to CSV (single column of voltages, or two columns time/voltage)
        df = pd.DataFrame({'time': t, 'voltage': sig})
        df.to_csv(filepath, index=False)
        print(f"Generated: {filepath} ({len(sig)} samples, rhythm: {rhythm})")
        generated_files.append(filepath)
        
    return generated_files

def process_ecg_file(filepath, fs, output_dir, student_info, correction_method='spline'):
    """
    Processes a single ECG file through the entire pipeline:
    1. Load CSV/TXT data
    2. DSP: Baseline Wander and Bandpass filtering
    3. Pan-Tompkins R-peak detection
    4. HRV calculations (Time, Freq, Nonlinear)
    5. Ectopic detection & correction
    6. Plot generation
    7. PDF/DOCX compilation
    """
    print(f"\nProcessing file: {filepath}")
    
    # Load data
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            # Support time/voltage columns, or raw single column
            if 'voltage' in df.columns:
                sig = df['voltage'].values
                t = df['time'].values if 'time' in df.columns else np.arange(len(sig)) / fs
            else:
                sig = df.iloc[:, 0].values
                t = np.arange(len(sig)) / fs
        else:
            # Assume space/tab separated values
            sig = np.loadtxt(filepath)
            t = np.arange(len(sig)) / fs
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

    # Initialize processors
    processor = ECGProcessor(fs=fs)
    analyzer = HRVAnalyzer(fs_ecg=fs)
    reporter = ReportGenerator(fs=fs)
    
    # 1. DSP Pipeline
    sig_clean, baseline = processor.remove_baseline_wander_median(sig)
    sig_filtered = processor.apply_bandpass(sig_clean, lowcut=0.5, highcut=45.0)
    sig_smoothed = processor.remove_noise_savgol(sig_filtered)
    
    # 2. Pan-Tompkins peak detection
    r_peaks, stages = processor.pan_tompkins_detector(sig_smoothed)
    
    if len(r_peaks) < 5:
        print(f"Warning: Too few R-peaks detected in {filepath} ({len(r_peaks)} peaks). Skipping.")
        return None
        
    # 3. Calculate RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) / fs * 1000.0
    
    # 4. Detect and correct ectopic beats
    ectopic_mask = analyzer.detect_ectopic_beats(rr_intervals)
    corrected_rr = analyzer.correct_ectopic_beats(rr_intervals, ectopic_mask, method=correction_method)
    
    ectopic_count = np.sum(ectopic_mask)
    ectopic_pct = (ectopic_count / len(rr_intervals)) * 100.0 if len(rr_intervals) > 0 else 0.0
    ectopic_stats = {
        'count': int(ectopic_count),
        'total_beats': int(len(r_peaks)),
        'pct': ectopic_pct
    }
    
    # 5. Extract HRV Features
    time_metrics = analyzer.compute_time_domain(corrected_rr)
    freq_metrics = analyzer.compute_frequency_domain(corrected_rr)
    nonlinear_metrics = analyzer.compute_nonlinear(corrected_rr)
    
    # 6. Physiological Interpretation
    interpretation = analyzer.generate_clinical_interpretation(time_metrics, freq_metrics, nonlinear_metrics)
    
    # 7. Generate Figure Plots
    file_basename = os.path.splitext(os.path.basename(filepath))[0]
    plots_subdir = os.path.join(output_dir, f"{file_basename}_figures")
    
    plot_paths = reporter.generate_all_plots(
        t=t,
        raw_sig=sig,
        filtered_sig=sig_smoothed,
        r_peaks=r_peaks,
        rr_raw=rr_intervals,
        rr_corrected=corrected_rr,
        ectopic_mask=ectopic_mask,
        psd_data=freq_metrics,
        output_dir=plots_subdir
    )
    
    # 8. Compile Reports (PDF & DOCX)
    pdf_report_path = os.path.join(output_dir, f"{file_basename}_report.pdf")
    docx_report_path = os.path.join(output_dir, f"{file_basename}_report.docx")
    
    # Ensure interpretation can be safely rendered in ReportLab
    reporter.generate_pdf(
        pdf_path=pdf_report_path,
        student_info=student_info,
        time_metrics=time_metrics,
        freq_metrics=freq_metrics,
        nonlinear_metrics=nonlinear_metrics,
        ectopic_stats=ectopic_stats,
        plot_paths=plot_paths,
        interpretation_text=interpretation
    )
    
    reporter.generate_docx(
        docx_path=docx_report_path,
        student_info=student_info,
        time_metrics=time_metrics,
        freq_metrics=freq_metrics,
        nonlinear_metrics=nonlinear_metrics,
        ectopic_stats=ectopic_stats,
        plot_paths=plot_paths,
        interpretation_text=interpretation
    )
    
    print(f"Successfully processed {filepath}!")
    print(f" - Ectopics Corrected: {ectopic_count} ({ectopic_pct:.2f}%)")
    print(f" - Heart Rate: {time_metrics['mean_hr']:.1f} BPM")
    print(f" - SDNN: {time_metrics['sdnn']:.1f} ms, RMSSD: {time_metrics['rmssd']:.1f} ms")
    print(f" - LF/HF Ratio: {freq_metrics['ratio']:.2f}")
    print(f" - PDF Report saved: {pdf_report_path}")
    print(f" - DOCX Report saved: {docx_report_path}")
    
    # Clean up Matplotlib figures to save memory
    import matplotlib.pyplot as plt
    plt.close('all')
    
    return {
        'file': filepath,
        'time_metrics': time_metrics,
        'freq_metrics': {k: v for k, v in freq_metrics.items() if k not in ['psd_f', 'psd_p']},
        'nonlinear_metrics': nonlinear_metrics,
        'ectopic_stats': ectopic_stats
    }

def main():
    parser = argparse.ArgumentParser(description="Academic ECG-HRV Processing & Analysis System")
    parser.add_argument('--input', type=str, help='Path to an ECG file (CSV/TXT) or directory of files')
    parser.add_argument('--fs', type=int, default=250, help='ECG sampling rate in Hz (default: 250)')
    parser.add_argument('--outdir', type=str, default='output_results', help='Directory to save outputs and reports')
    parser.add_argument('--generate-test', action='store_true', help='Generate synthetic test ECG files in output directory')
    parser.add_argument('--test-duration', type=int, default=60, help='Duration in seconds of synthetic test files (default: 60)')
    parser.add_argument('--student-name', type=str, default='Biomedical Student', help='Student Name for Report Title Block')
    parser.add_argument('--student-id', type=str, default='BME-2026-09', help='Student Roll No/ID for Report Title Block')
    parser.add_argument('--supervisor', type=str, default='Dr. Eleanor Vance', help='Supervisor Name for Report Title Block')
    parser.add_argument('--correction', type=str, choices=['linear', 'spline'], default='spline', help='Ectopic correction method')
    
    args = parser.parse_args()
    
    student_info = {
        'name': args.student_name,
        'id': args.student_id,
        'supervisor': args.supervisor
    }
    
    # Case 1: Generate test data
    if args.generate_test or (not args.input and not os.path.exists('data')):
        test_dir = os.path.join(args.outdir, 'synthetic_test_data')
        files = generate_test_data(test_dir, fs=args.fs, duration=args.test_duration)
        if not args.input:
            args.input = test_dir  # Automatically switch to processing the generated directory
            
    # Case 2: Process input file or folder
    if args.input:
        results = []
        if os.path.isdir(args.input):
            # Batch process directory
            print(f"Batch processing directory: {args.input}")
            for filename in os.listdir(args.input):
                if filename.endswith('.csv') or filename.endswith('.txt'):
                    filepath = os.path.join(args.input, filename)
                    res = process_ecg_file(filepath, args.fs, args.outdir, student_info, args.correction)
                    if res:
                        results.append(res)
        else:
            # Process single file
            res = process_ecg_file(args.input, args.fs, args.outdir, student_info, args.correction)
            if res:
                results.append(res)
                
        # Export a collective summary JSON
        if results:
            summary_path = os.path.join(args.outdir, 'batch_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\n--- Batch Process Finished! Summary exported to: {summary_path} ---")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
