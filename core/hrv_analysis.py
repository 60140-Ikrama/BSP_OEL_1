import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.interpolate import interp1d

class HRVAnalyzer:
    """
    Computes Time-Domain, Frequency-Domain, and Non-Linear HRV metrics from RR intervals.
    Includes ectopic beat detection, correction, and physiological interpretation logic.
    """
    def __init__(self, fs_ecg=250):
        self.fs_ecg = fs_ecg

    def detect_ectopic_beats(self, rr_intervals, threshold_pct=0.20):
        """
        Detects ectopic beats in RR intervals using a rolling median window and global limits.
        An interval is flagged as ectopic if:
         1. It is outside physiological limits (300 ms to 2000 ms).
         2. It deviates from the rolling median of 15 beats by more than threshold_pct (20%).
        Returns:
            ectopic_mask: boolean np.array (True where RR interval is ectopic)
        """
        rr = np.array(rr_intervals)
        n = len(rr)
        ectopic_mask = np.zeros(n, dtype=bool)
        
        # 1. Global physiological limit checks
        ectopic_mask |= (rr < 300.0) | (rr > 2000.0)
        
        # 2. Local rolling median check
        window_size = 15
        half_win = window_size // 2
        
        for i in range(n):
            # Define local window boundaries
            start = max(0, i - half_win)
            end = min(n, i + half_win + 1)
            
            # Compute median excluding the current point if possible (or just local median)
            local_median = np.median(rr[start:end])
            
            # Check percentage deviation
            if np.abs(rr[i] - local_median) > (threshold_pct * local_median):
                ectopic_mask[i] = True
                
        # 3. Check for sudden successive outliers (typical of PVCs)
        # Standard deviation deviation check
        global_median = np.median(rr)
        global_mad = np.median(np.abs(rr - global_median))
        # Avoid division by zero
        mad_thresh = max(global_mad * 3.5, 50.0)
        ectopic_mask |= np.abs(rr - global_median) > mad_thresh
        
        return ectopic_mask

    def correct_ectopic_beats(self, rr_intervals, ectopic_mask, method='spline'):
        """
        Corrects detected ectopic beats by interpolating over their positions.
        Methods:
         - 'linear': Linear interpolation between neighboring normal beats.
         - 'spline': Cubic spline interpolation.
        Returns:
            corrected_rr: np.array of corrected RR intervals.
        """
        rr = np.array(rr_intervals).copy()
        indices = np.arange(len(rr))
        
        # Valid (normal) indices
        normal_idx = indices[~ectopic_mask]
        ectopic_idx = indices[ectopic_mask]
        
        # If no ectopic beats or not enough normal beats to interpolate, return original
        if len(ectopic_idx) == 0 or len(normal_idx) < 4:
            return rr
            
        if method == 'linear':
            f_interp = interp1d(normal_idx, rr[normal_idx], kind='linear', fill_value='extrapolate')
            rr[ectopic_idx] = f_interp(ectopic_idx)
        elif method == 'spline':
            # Use cubic spline interpolation
            f_interp = interp1d(normal_idx, rr[normal_idx], kind='cubic', fill_value='extrapolate')
            rr[ectopic_idx] = f_interp(ectopic_idx)
            
        # Ensure physiological sanity after interpolation
        rr = np.clip(rr, 300.0, 2000.0)
        return rr

    def compute_time_domain(self, rr_intervals):
        """
        Calculates Time-Domain HRV features:
         - Mean RR (ms)
         - Mean Heart Rate (BPM)
         - SDNN (ms) - Standard Deviation of NN intervals
         - RMSSD (ms) - Root Mean Square of Successive Differences
         - NN50 (count) - Number of successive differences > 50ms
         - pNN50 (%) - Percentage of successive differences > 50ms
        """
        rr = np.array(rr_intervals)
        n = len(rr)
        
        if n < 2:
            return {'mean_rr': 0, 'mean_hr': 0, 'sdnn': 0, 'rmssd': 0, 'nn50': 0, 'pnn50': 0}
            
        mean_rr = np.mean(rr)
        mean_hr = 60000.0 / mean_rr
        sdnn = np.std(rr, ddof=1)
        
        # Successive differences
        diff_rr = np.diff(rr)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        
        nn50 = np.sum(np.abs(diff_rr) > 50.0)
        pnn50 = (nn50 / len(diff_rr)) * 100.0
        
        return {
            'mean_rr': mean_rr,
            'mean_hr': mean_hr,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'nn50': int(nn50),
            'pnn50': pnn50
        }

    def compute_frequency_domain(self, rr_intervals, resampling_fs=4.0):
        """
        Calculates Frequency-Domain HRV features using Welch's PSD.
        Steps:
         1. Resample unevenly spaced RR intervals to a uniform time-series at 4.0 Hz.
         2. Apply Welch Periodogram to obtain Power Spectral Density.
         3. Integrate spectral power in standard bands:
            - VLF: 0.00 - 0.04 Hz
            - LF:  0.04 - 0.15 Hz (mixed sympathetic/parasympathetic activity)
            - HF:  0.15 - 0.40 Hz (pure parasympathetic vagal modulation)
         4. Calculate LF/HF ratio.
        """
        rr = np.array(rr_intervals)
        
        # Time coordinates of the RR peaks (cumulative sum of intervals converted to seconds)
        rr_time = np.cumsum(rr) / 1000.0
        # Start time at 0
        rr_time = rr_time - rr_time[0]
        
        # Create a uniform time grid for resampling
        resample_time = np.arange(0, rr_time[-1], 1.0 / resampling_fs)
        
        if len(resample_time) < 16:
            # Not enough data
            return {'vlf': 0, 'lf': 0, 'hf': 0, 'lf_nu': 0, 'hf_nu': 0, 'ratio': 0, 'psd_f': np.array([]), 'psd_p': np.array([])}
            
        # Resample using cubic spline
        f_resample = interp1d(rr_time, rr, kind='cubic', fill_value='extrapolate')
        rr_resampled = f_resample(resample_time)
        
        # Remove mean (detrending)
        rr_detrended = rr_resampled - np.mean(rr_resampled)
        
        # Compute Welch PSD
        # Window size is chosen to resolve low frequency components (e.g. 256 or 512 points)
        nperseg = min(256, len(rr_detrended))
        f, pxx = welch(rr_detrended, fs=resampling_fs, nperseg=nperseg, scaling='density')
        
        # Integrate power in bands using custom trapezoidal integration for NumPy 2.0+ compatibility
        def trapz(y, x):
            if len(y) < 2:
                return 0.0
            return np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x))

        # VLF (0 - 0.04 Hz)
        vlf_mask = (f >= 0.00) & (f < 0.04)
        vlf_power = trapz(pxx[vlf_mask], f[vlf_mask])
        
        # LF (0.04 - 0.15 Hz)
        lf_mask = (f >= 0.04) & (f < 0.15)
        lf_power = trapz(pxx[lf_mask], f[lf_mask])
        
        # HF (0.15 - 0.40 Hz)
        hf_mask = (f >= 0.15) & (f <= 0.40)
        hf_power = trapz(pxx[hf_mask], f[hf_mask])
        
        total_power = vlf_power + lf_power + hf_power
        
        # Normalized units (nu)
        lf_nu = (lf_power / (total_power - vlf_power)) * 100.0 if (total_power - vlf_power) > 0 else 0.0
        hf_nu = (hf_power / (total_power - vlf_power)) * 100.0 if (total_power - vlf_power) > 0 else 0.0
        
        ratio = lf_power / hf_power if hf_power > 0 else 0.0
        
        return {
            'vlf': vlf_power,
            'lf': lf_power,
            'hf': hf_power,
            'lf_nu': lf_nu,
            'hf_nu': hf_nu,
            'ratio': ratio,
            'total_power': total_power,
            'psd_f': f,
            'psd_p': pxx
        }

    def compute_nonlinear(self, rr_intervals):
        """
        Calculates Non-Linear HRV features:
         - Poincaré Plot parameters (SD1, SD2)
         - Sample Entropy (SampEn)
        """
        rr = np.array(rr_intervals)
        n = len(rr)
        
        if n < 5:
            return {'sd1': 0, 'sd2': 0, 'ratio': 0, 'sampen': 0}
            
        # Poincaré plot elements: x = RR_i, y = RR_{i+1}
        x = rr[:-1]
        y = rr[1:]
        
        # SD1: standard deviation perpendicular to line-of-identity (y = x)
        # Represents short-term HRV / parasympathetic influence
        sd1 = np.sqrt(0.5 * np.var(x - y, ddof=1))
        
        # SD2: standard deviation along the line-of-identity
        # Represents long-term HRV / combined sympathetic & parasympathetic
        sd2 = np.sqrt(2 * np.var(rr, ddof=1) - 0.5 * np.var(x - y, ddof=1))
        ratio = sd1 / sd2 if sd2 > 0 else 0.0
        
        # Compute Sample Entropy (SampEn)
        sampen = self._calculate_sample_entropy(rr, m=2, r=0.2 * np.std(rr, ddof=1))
        
        return {
            'sd1': sd1,
            'sd2': sd2,
            'ratio': ratio,
            'sampen': sampen
        }

    def _calculate_sample_entropy(self, data, m=2, r=0.2):
        """
        Calculates Sample Entropy (SampEn) of a time series.
        m: template length (default 2)
        r: tolerance threshold (default 0.2 * standard deviation)
        """
        n = len(data)
        if n < m + 1:
            return 0.0
            
        # Helper function to count templates
        def _count_templates(dim):
            # Create templates
            templates = np.array([data[i : i + dim] for i in range(n - dim + 1)])
            # Compute pairwise distances using Chebychev (Max) distance
            count = 0
            num_templates = len(templates)
            for i in range(num_templates):
                # Subtract template i from all templates, take max distance across dim
                diffs = np.max(np.abs(templates - templates[i]), axis=1)
                # Count points that fall within tolerance r, excluding self-comparison
                count += np.sum(diffs < r) - 1
            return count
            
        # Count for length m and m+1
        count_m = _count_templates(m)
        count_m1 = _count_templates(m + 1)
        
        if count_m1 == 0 or count_m == 0:
            return 0.0
            
        return -np.log(count_m1 / count_m)

    def generate_clinical_interpretation(self, time_metrics, freq_metrics, nonlinear_metrics):
        """
        Generates a professional academic interpretation of the HRV results.
        Discusses sympathovagal balance, parasympathetic tone, and overall complexity.
        """
        rmssd = time_metrics['rmssd']
        sdnn = time_metrics['sdnn']
        ratio = freq_metrics['ratio']
        sd1 = nonlinear_metrics['sd1']
        sd2 = nonlinear_metrics['sd2']
        sampen = nonlinear_metrics['sampen']
        
        # 1. Evaluate Vagal Tone (Parasympathetic)
        if rmssd > 50.0:
            vagal_status = "High / Robust vagal (parasympathetic) tone. This indicates good cardiovascular fitness, high vagal control, and healthy autonomic recovery."
        elif rmssd > 25.0:
            vagal_status = "Moderate / Balanced vagal tone. Autonomic nervous system shows normal parasympathetic modulation."
        else:
            vagal_status = "Depressed vagal (parasympathetic) tone. Reduced RMSSD is a marker of stress, fatigue, or autonomic imbalance, commonly observed in cardiovascular conditions or high stress states."
            
        # 2. Evaluate Autonomic Balance (Sympathovagal)
        if ratio > 2.0:
            balance_status = f"Sympathetic dominance (LF/HF = {ratio:.2f} > 2.0). Indicates a shift towards fight-or-flight mechanisms, high physical/mental stress, or sympathetic hyperactivation."
        elif ratio < 0.5:
            balance_status = f"Parasympathetic dominance (LF/HF = {ratio:.2f} < 0.5). Typically associated with deep rest, athletic recovery, or high vagal drive."
        else:
            balance_status = f"Balanced autonomic state (LF/HF = {ratio:.2f}). Reflects healthy homeostatic equilibrium between the sympathetic and parasympathetic branches."

        # 3. Evaluate Signal Complexity (Sample Entropy)
        if sampen > 1.2:
            complexity_status = "High complexity and healthy physiological adaptability. The cardiac pacemaker behaves non-linearly, showing normal chaotic fluctuations that allow rapid adaptation to external stimuli."
        elif sampen > 0.7:
            complexity_status = "Moderate complexity. The heartbeat intervals display standard physiological variability."
        else:
            complexity_status = "Low complexity and high signal regularity. Reduced sample entropy indicates rigid physiological control or rhythm pacing, often seen in heart failure, high age, or pathological arrhythmia."
            
        interpretation_text = (
            f"**Autonomic Vagal Assessment (RMSSD = {rmssd:.1f} ms, SD1 = {sd1:.1f} ms):**\n"
            f"{vagal_status}\n\n"
            f"**Sympathovagal Balance (LF/HF Ratio = {ratio:.2f}):**\n"
            f"{balance_status}\n\n"
            f"**Non-Linear Adaptability (Sample Entropy = {sampen:.3f}, SD2 = {sd2:.1f} ms):**\n"
            f"{complexity_status}"
        )
        
        return interpretation_text

# Test script
if __name__ == '__main__':
    from synthetic_data import SyntheticECGGenerator
    gen = SyntheticECGGenerator(fs=250)
    t, sig, r_peaks, ectopics = gen.generate_signal(duration_sec=60, rhythm='PVC')
    
    # Simple peak detector to test HRV
    from signal_processing import ECGProcessor
    proc = ECGProcessor(fs=250)
    sig_clean, _ = proc.remove_baseline_wander_median(sig)
    sig_filt = proc.apply_bandpass(sig_clean, lowcut=0.5, highcut=40.0)
    detected_peaks, _ = proc.pan_tompkins_detector(sig_filt)
    
    # Calculate RR intervals in ms
    rr = np.diff(detected_peaks) / 250.0 * 1000.0
    
    analyzer = HRVAnalyzer(fs_ecg=250)
    # Detect and correct
    mask = analyzer.detect_ectopic_beats(rr)
    corrected_rr = analyzer.correct_ectopic_beats(rr, mask, method='spline')
    
    # Analyze
    time_m = analyzer.compute_time_domain(corrected_rr)
    freq_m = analyzer.compute_frequency_domain(corrected_rr)
    nonl_m = analyzer.compute_nonlinear(corrected_rr)
    
    print(f"Ectopics detected: {np.sum(mask)} ({np.sum(mask)/len(rr)*100:.1f}%)")
    print("Time domain:", time_m)
    print("Freq ratio:", freq_m['ratio'])
    print("Nonlinear SD1/SD2:", nonl_m['sd1'], nonl_m['sd2'], "SampEn:", nonl_m['sampen'])
