import numpy as np

class SyntheticECGGenerator:
    """
    Generates synthetic ECG signals with various rhythm types and noise profiles.
     Rhythms:
      - NSR (Normal Sinus Rhythm) with Respiratory Sinus Arrhythmia (RSA)
      - AFib (Atrial Fibrillation) with highly irregular beats and absent P-waves
      - PVC (Premature Ventricular Contraction) incorporating ectopic beats and compensatory pauses
      - VTach (Ventricular Tachycardia) characterized by rapid, wide-complex waveforms
     Noise sources:
      - Baseline Wander (low-frequency breathing artifact)
      - Powerline Interference (50 Hz or 60 Hz)
      - High-frequency electromyographic (EMG) muscle noise
    """
    def __init__(self, fs=250):
        self.fs = fs  # Sampling rate in Hz

    def _generate_single_beat(self, beat_len, is_pvc=False, is_afib=False):
        """
        Generates a single ECG beat of a given length (in samples).
        Models P, Q, R, S, T waves as Gaussian functions.
        """
        t = np.linspace(0, 1, beat_len)
        signal = np.zeros(beat_len)

        # Standard parameters for Gaussian waves: (amplitude, center_phase, width)
        # Phase goes from 0 (start of beat) to 1 (end of beat)
        if is_pvc:
            # Ectopic PVC beat: wide, tall/distorted QRS, no P-wave, inverted T-wave
            wave_params = {
                'Q': (-0.1, 0.25, 0.03),
                'R': (1.8, 0.35, 0.07),  # Wider and taller R-peak
                'S': (-0.6, 0.42, 0.05),
                'T': (-0.4, 0.65, 0.15)  # Inverted T-wave
            }
        else:
            # Normal beat
            wave_params = {
                'P': (0.12, 0.15, 0.03) if not is_afib else (0, 0, 0),  # No P-wave in AFib
                'Q': (-0.08, 0.32, 0.015),
                'R': (1.0, 0.35, 0.015),  # Sharp R-peak
                'S': (-0.25, 0.38, 0.015),
                'T': (0.22, 0.65, 0.065)
            }
            # AFib adds small chaotic f-waves instead of a flat baseline
            if is_afib:
                f_freq = 6.0 + 2.0 * np.random.rand()  # 6-8 Hz f-waves
                signal += 0.08 * np.sin(2 * np.pi * f_freq * (t * beat_len / self.fs))

        for wave, (amp, pos, width) in wave_params.items():
            if amp == 0:
                continue
            signal += amp * np.exp(-((t - pos) ** 2) / (2 * width ** 2))

        return signal

    def generate_signal(self, duration_sec=60, rhythm='NSR', noise_config=None):
        """
        Generates a continuous ECG signal.
        rhythm options: 'NSR', 'AFib', 'PVC', 'VTach'
        noise_config: dict with keys 'baseline_wander' (float amp), 'powerline' (float amp), 'emg' (float amp)
        Returns:
            t: time array
            signal: raw ECG signal
            r_peaks: indices of true R-peaks
            ectopic_indices: indices of ectopic R-peaks (for PVC)
        """
        total_samples = int(duration_sec * self.fs)
        
        # 1. Generate RR intervals based on rhythm type
        rr_intervals = []
        ectopic_mask = []  # True if the beat starting at this interval is ectopic
        
        if rhythm == 'NSR':
            # Nominal HR = 70 bpm (mean RR = 857 ms)
            mean_rr = 0.85  # seconds
            num_beats = int(duration_sec / mean_rr) + 5
            # Add RSA: Respiratory Sinus Arrhythmia (~0.25 Hz breathing cycle)
            respir_t = np.arange(num_beats) * mean_rr
            rsa_mod = 0.06 * np.sin(2 * np.pi * 0.25 * respir_t)
            # Add slight physiological HRV noise
            hrv_noise = 0.02 * np.random.randn(num_beats)
            rr_intervals = mean_rr + rsa_mod + hrv_noise
            ectopic_mask = [False] * len(rr_intervals)
            
        elif rhythm == 'AFib':
            # Highly irregular RR intervals (random walk/gamma-like distribution)
            # Mean HR = 90 bpm (mean RR = 0.67s)
            mean_rr = 0.67
            num_beats = int(duration_sec / mean_rr) + 10
            # Generate irregular intervals with standard deviation of ~120ms
            rr_intervals = np.random.gamma(shape=25, scale=mean_rr/25, size=num_beats)
            ectopic_mask = [False] * len(rr_intervals)
            
        elif rhythm == 'PVC':
            # Normal rhythm with occasional PVCs (ectopic beats)
            # A PVC comes early (short RR) and is followed by a compensatory pause (long RR)
            mean_rr = 0.85
            num_beats = int(duration_sec / mean_rr) + 5
            respir_t = np.arange(num_beats) * mean_rr
            rsa_mod = 0.05 * np.sin(2 * np.pi * 0.22 * respir_t)
            rr_intervals = list(mean_rr + rsa_mod + 0.015 * np.random.randn(num_beats))
            ectopic_mask = [False] * len(rr_intervals)
            
            # Inject PVCs every 8-12 beats
            idx = 8
            while idx < num_beats - 5:
                # Ectopic beat comes early (e.g. 60% of normal interval)
                orig_rr = rr_intervals[idx]
                pvc_rr = orig_rr * 0.55
                # Compensatory pause makes the next beat late
                pause_rr = orig_rr * 1.45
                
                rr_intervals[idx] = pvc_rr
                rr_intervals[idx+1] = pause_rr
                ectopic_mask[idx] = True
                
                idx += np.random.randint(9, 14)  # interval between PVCs
                
        elif rhythm == 'VTach':
            # Rapid, regular wide complexes. HR = 150 bpm (mean RR = 0.40s)
            mean_rr = 0.40
            num_beats = int(duration_sec / mean_rr) + 5
            rr_intervals = mean_rr + 0.008 * np.random.randn(num_beats)
            ectopic_mask = [False] * len(rr_intervals)
            
        else:
            raise ValueError(f"Unknown rhythm: {rhythm}")
            
        # 2. Assemble the continuous signal from single beats
        signal = np.zeros(total_samples)
        r_peaks = []
        ectopic_indices = []
        
        current_sample = 0
        for i, rr in enumerate(rr_intervals):
            beat_len = int(rr * self.fs)
            if current_sample + beat_len > total_samples:
                break
                
            is_pvc_beat = ectopic_mask[i] or (rhythm == 'VTach')  # VTach uses wide PVC-like shapes
            is_afib_beat = (rhythm == 'AFib')
            
            beat_sig = self._generate_single_beat(beat_len, is_pvc=is_pvc_beat, is_afib=is_afib_beat)
            signal[current_sample : current_sample + beat_len] = beat_sig
            
            # The R-peak is at 35% of the single beat length
            r_peak_pos = current_sample + int(0.35 * beat_len)
            r_peaks.append(r_peak_pos)
            if ectopic_mask[i]:
                ectopic_indices.append(r_peak_pos)
                
            current_sample += beat_len
            
        t = np.arange(total_samples) / self.fs
        
        # 3. Add noise sources if requested
        if noise_config:
            if 'baseline_wander' in noise_config and noise_config['baseline_wander'] > 0:
                # Low freq breathing (e.g. 0.15 Hz and 0.3 Hz)
                amp = noise_config['baseline_wander']
                wander = amp * np.sin(2 * np.pi * 0.12 * t) + (amp * 0.5) * np.sin(2 * np.pi * 0.28 * t + 1.2)
                signal += wander
                
            if 'powerline' in noise_config and noise_config['powerline'] > 0:
                # 50 Hz powerline interference
                amp = noise_config['powerline']
                signal += amp * np.sin(2 * np.pi * 50.0 * t)
                
            if 'emg' in noise_config and noise_config['emg'] > 0:
                # White noise for EMG muscle artifact
                amp = noise_config['emg']
                signal += amp * np.random.randn(len(signal))
                
        return t, signal, np.array(r_peaks), np.array(ectopic_indices)

# Simple test generation script
if __name__ == '__main__':
    # Test generation of a 10-second ECG signal
    gen = SyntheticECGGenerator(fs=250)
    t, sig, r_peaks, ectopics = gen.generate_signal(
        duration_sec=10, 
        rhythm='PVC', 
        noise_config={'baseline_wander': 0.15, 'powerline': 0.05, 'emg': 0.02}
    )
    print(f"Generated {len(sig)} samples, {len(r_peaks)} R-peaks, {len(ectopics)} ectopic beats.")
