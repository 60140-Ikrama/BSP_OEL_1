import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
import neurokit2 as nk

class ECGProcessor:
    """
    Handles biomedical signal processing operations for ECG analysis.
    Implements baseline wander removal, bandpass filtering, noise suppression,
    and R-peak detection via the Pan-Tompkins algorithm.
    """
    def __init__(self, fs=250):
        self.fs = fs

    def detect_peaks_nk(self, data, method='neurokit'):
        """
        Detects R-peaks using NeuroKit2 algorithms.
        Supported methods: 'neurokit', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'engzee2012', 'kalidas2016'
        """
        try:
            # nk.ecg_peaks expects a 1D numeric array of signal values
            signals, info = nk.ecg_peaks(data, sampling_rate=self.fs, method=method)
            r_peaks = info['ECG_R_Peaks']
            
            # Align R-peaks with local maximum in data
            aligned_r_peaks = []
            search_win = int(0.04 * self.fs)  # 40ms window
            for rp in r_peaks:
                start = max(0, rp - search_win)
                end = min(len(data), rp + search_win)
                local_max_idx = start + np.argmax(np.abs(data[start:end]))
                aligned_r_peaks.append(local_max_idx)
            return np.unique(aligned_r_peaks)
        except Exception as e:
            # Fallback to custom Pan-Tompkins on failure
            print(f"NeuroKit2 QRS detection failed ({e}). Falling back to custom Pan-Tompkins.")
            peaks, _ = self.pan_tompkins_detector(data)
            return peaks

    def butter_bandpass(self, lowcut=0.5, highcut=45.0, order=3):
        """
        Designs a Butterworth bandpass filter.
        """
        nyq = 0.5 * self.fs
        # Constrain lowcut and highcut to be within Nyquist limits (0 < Wn < 1)
        lowcut_val = max(0.01, min(lowcut, nyq - 1.0))
        highcut_val = max(lowcut_val + 1.0, min(highcut, nyq - 1.0))
        
        low = lowcut_val / nyq
        high = highcut_val / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass(self, data, lowcut=0.5, highcut=45.0, order=3):
        """
        Applies a zero-phase Butterworth bandpass filter using filtfilt to avoid phase shift.
        """
        b, a = self.butter_bandpass(lowcut, highcut, order)
        return filtfilt(b, a, data)

    def remove_baseline_wander_median(self, data):
        """
        Removes baseline wander using a dual median filter approach.
        A 200ms median filter removes QRS complexes and T-waves.
        A 600ms median filter removes P-waves.
        The result is the baseline drift, which is subtracted from the raw signal.
        """
        # Convert ms to samples
        win1 = int(0.2 * self.fs)  # 200ms
        win2 = int(0.6 * self.fs)  # 600ms

        # Ensure windows are odd
        if win1 % 2 == 0: win1 += 1
        if win2 % 2 == 0: win2 += 1

        # Use scipy's fast median filter implementation (handles boundaries)
        from scipy.ndimage import median_filter
        
        # Step 1: Remove QRS and T waves
        base1 = median_filter(data, size=win1)
        # Step 2: Remove P waves (smooths base1 further to find baseline)
        baseline = median_filter(base1, size=win2)

        # Step 3: Subtract baseline from raw signal
        corrected_data = data - baseline
        return corrected_data, baseline

    def remove_noise_savgol(self, data, window_ms=80, polyorder=2):
        """
        Suppresses high-frequency noise (e.g., muscle noise) using a Savitzky-Golay filter.
        It fits local polynomials to smooth signal details while preserving sharp R-peaks.
        """
        window_size = int((window_ms / 1000.0) * self.fs)
        if window_size % 2 == 0:
            window_size += 1
        if window_size < 3:
            window_size = 3
            
        # Ensure window_size is strictly smaller than the signal length to prevent SciPy's polyfit crash
        n = len(data)
        if window_size >= n:
            window_size = n - 1 if n % 2 == 0 else n - 2
            if window_size < 3:
                # Signal is too short for Savitzky-Golay filtering, return original signal
                return data
        return savgol_filter(data, window_length=window_size, polyorder=polyorder)

    def pan_tompkins_detector(self, data):
        """
        Implements the classic Pan-Tompkins QRS detection algorithm.
        Steps:
         1. Bandpass filter (5-15 Hz) to isolate QRS energy.
         2. Differentiation to highlight the steep slope of the QRS complex.
         3. Squaring function to intensify QRS complex and make all points positive.
         4. Moving Window Integration (approx 150ms window) to obtain QRS envelopes.
         5. Adaptive Thresholding with back-search and refractory period.
        Returns:
            r_peaks: np.array of R-peak sample indices.
            stages: dict containing intermediate signals for visualization.
        """
        # --- Stage 1: Bandpass Filtering (5 - 15 Hz) ---
        nyq = 0.5 * self.fs
        b, a = butter(3, [5.0 / nyq, 15.0 / nyq], btype='band')
        filtered = filtfilt(b, a, data)

        # --- Stage 2: Derivative Filter ---
        # Equation: y[n] = (1/8) * (2*x[n] + x[n-1] - x[n-3] - 2*x[n-4])
        # Implemented using a convolution filter
        derivative_kernel = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) / 8.0
        derived = np.convolve(filtered, derivative_kernel, mode='same')

        # --- Stage 3: Squaring ---
        squared = derived ** 2

        # --- Stage 4: Moving Integration Window (150ms) ---
        window_size = int(0.150 * self.fs)
        integration_kernel = np.ones(window_size) / window_size
        integrated = np.convolve(squared, integration_kernel, mode='same')

        # --- Stage 5: Adaptive Thresholding ---
        # Find all local peaks in the integrated signal
        peaks, _ = find_peaks(integrated, distance=int(0.200 * self.fs)) # Refractory limit of 200ms
        
        r_peaks = []
        
        # Adaptive Threshold state variables
        spki = np.max(integrated) * 0.25  # Signal Peak Level
        npki = np.mean(integrated) * 0.5   # Noise Peak Level
        
        avg_rr = int(0.8 * self.fs)  # Initial guess of 800ms
        refractory_samples = int(0.200 * self.fs)  # 200ms refractory period
        
        last_r_peak = -refractory_samples
        
        i = 0
        while i < len(peaks):
            peak_idx = peaks[i]
            peak_val = integrated[peak_idx]
            
            # Threshold calculations
            thr1 = npki + 0.25 * (spki - npki)
            thr2 = 0.5 * thr1  # Threshold for back-search
            
            # Check if it exceeds signal threshold and respects refractory period
            if peak_val > thr1:
                if peak_idx - last_r_peak > refractory_samples:
                    r_peaks.append(peak_idx)
                    spki = 0.125 * peak_val + 0.875 * spki
                    # Update average RR interval
                    if len(r_peaks) > 1:
                        avg_rr = int(np.mean(np.diff(r_peaks[-8:])))
                    last_r_peak = peak_idx
                else:
                    # Double-triggering check (if current is larger than last detected within refractory)
                    if len(r_peaks) > 0 and peak_val > integrated[r_peaks[-1]]:
                        # Replace last peak if this one is higher
                        r_peaks[-1] = peak_idx
                        spki = 0.125 * peak_val + 0.875 * spki
                        last_r_peak = peak_idx
            else:
                # Noise peak
                npki = 0.125 * peak_val + 0.875 * npki
                
            # --- Back-Search Mechanism ---
            # If no QRS is detected in 1.66 times the average RR interval
            if len(r_peaks) > 0 and (peak_idx - last_r_peak) > int(1.66 * avg_rr):
                # Search back in the window for skipped peaks
                search_start = last_r_peak + refractory_samples
                search_end = peak_idx
                
                # Find local peaks in the missed region
                missed_peaks, _ = find_peaks(integrated[search_start:search_end], distance=int(0.200 * self.fs))
                missed_peaks = missed_peaks + search_start  # adjust indices
                
                for mp in missed_peaks:
                    if integrated[mp] > thr2:
                        r_peaks.append(mp)
                        spki = 0.25 * integrated[mp] + 0.75 * spki
                        last_r_peak = mp
                        avg_rr = int(np.mean(np.diff(r_peaks[-8:])))
                        break # add first valid back-search peak
            
            i += 1

        # R-peaks detected in integrated signal are slightly delayed (~75ms window center shift).
        # We align the detected peaks with the local maximum in the original bandpass-filtered ECG.
        aligned_r_peaks = []
        search_win = int(0.1 * self.fs) # 100ms search window
        
        for rp in r_peaks:
            start = max(0, rp - search_win)
            end = min(len(data), rp + int(0.05 * self.fs))
            # Find the peak in the bandpass-filtered or baseline-wander-removed signal
            local_max_idx = start + np.argmax(np.abs(data[start:end]))
            aligned_r_peaks.append(local_max_idx)
            
        aligned_r_peaks = np.unique(aligned_r_peaks)  # Remove duplicates

        stages = {
            'filtered': filtered,
            'derived': derived,
            'squared': squared,
            'integrated': integrated
        }

        return aligned_r_peaks, stages

# Test code
if __name__ == '__main__':
    # Simple verification
    from synthetic_data import SyntheticECGGenerator
    gen = SyntheticECGGenerator(fs=250)
    t, sig, true_peaks, _ = gen.generate_signal(duration_sec=10, rhythm='NSR')
    
    proc = ECGProcessor(fs=250)
    # Remove baseline wander
    sig_clean, baseline = proc.remove_baseline_wander_median(sig)
    # Apply bandpass
    sig_filt = proc.apply_bandpass(sig_clean, lowcut=0.5, highcut=40.0)
    # Detect peaks
    detected_peaks, stages = proc.pan_tompkins_detector(sig_filt)
    
    print(f"True Peaks: {len(true_peaks)}, Detected: {len(detected_peaks)}")
