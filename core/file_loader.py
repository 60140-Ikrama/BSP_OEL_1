import os
import numpy as np
import pandas as pd
import scipy.io as sio

def load_ecg_file(filepath, fs=250):
    """
    Loads an ECG signal from a file.
    Supported formats:
      - .csv / .txt: Comma/space/tab-separated text files.
      - .mat: MATLAB files (finds the largest 1D/2D array).
      - .dat: Text-based or flat binary files.
      - .edf: European Data Format (parsed using pure Python).
    Returns:
        t: time array (seconds)
        sig: ECG voltage array (mV or normalized)
        detected_fs: sampling frequency (if found in file, otherwise defaults to fs)
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.csv', '.txt']:
        t, sig, file_fs = _load_text_file(filepath, fs)
    elif ext == '.mat':
        t, sig, file_fs = _load_mat_file(filepath, fs)
    elif ext == '.dat':
        t, sig, file_fs = _load_dat_file(filepath, fs)
    elif ext == '.edf':
        t, sig, file_fs = _load_edf_file(filepath, fs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Clean NaNs and infs from the signal to prevent filter divergence (e.g. in filtfilt)
    t = np.asarray(t, dtype=np.float64).copy()
    sig = np.asarray(sig, dtype=np.float64).copy()
    non_finite_mask = ~np.isfinite(sig)
    if non_finite_mask.any():
        finite_mask = ~non_finite_mask
        if finite_mask.any():
            # Interpolate non-finite values using nearest finite points
            sig[non_finite_mask] = np.interp(
                np.flatnonzero(non_finite_mask), 
                np.flatnonzero(finite_mask), 
                sig[finite_mask]
            )
        else:
            # Fallback if entire signal is non-finite
            sig[:] = 0.0

    return t, sig, file_fs

def _load_text_file(filepath, default_fs):
    """
    Parses CSV and TXT files. Looks for time and voltage columns.
    Handles semicolon delimiters, decimal commas, and files with or without headers.
    """
    # Skip comments/metadata and detect delimiter
    first_line = ""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_strip = line.strip()
            if line_strip and not line_strip.startswith(('#', '%')):
                first_line = line
                break
                
    if not first_line:
        raise ValueError("Empty file or only comments/metadata found.")
    
    delim = ','
    if '\t' in first_line:
        delim = '\t'
    elif ';' in first_line:
        delim = ';'
    elif ' ' in first_line and ',' not in first_line:
        delim = ' '
        
    # Split first line by delimiter to check if it's header or data
    tokens = [t.strip() for t in first_line.split(delim) if t.strip()]
    is_header = False
    if tokens:
        for tok in tokens:
            cleaned_tok = tok.replace(',', '.')
            try:
                float(cleaned_tok)
            except ValueError:
                # Found a non-numeric token, so it's likely a header
                is_header = True
                break
                
    try:
        # Load using pandas and make a copy to ensure writeable blocks
        sep_param = r'\s+' if delim == ' ' else delim
        header_param = 0 if is_header else None
        df = pd.read_csv(filepath, sep=sep_param, header=header_param, comment='#').copy()
    except Exception:
        # Fallback to numpy loadtxt
        try:
            try:
                sig = np.loadtxt(filepath, comments=('#', '%'))
            except Exception:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().replace(',', '.')
                from io import StringIO
                sig = np.loadtxt(StringIO(content), comments=('#', '%'))
                
            t = np.arange(len(sig)) / default_fs
            return t, sig, default_fs
        except Exception as e:
            raise ValueError(f"Failed to parse text file: {e}")
            
    # Clean columns: replace commas with dots for decimals and cast to float
    for c in df.columns:
        if df[c].dtype == object:
            try:
                cleaned_col = df[c].astype(str).str.replace(',', '.', regex=False).str.strip()
                # Remove common units/characters if any
                cleaned_col = cleaned_col.str.replace(r'[^\d\.\-\+eE]', '', regex=True)
                converted = pd.to_numeric(cleaned_col, errors='coerce')
                if converted.notna().sum() > 0.5 * len(df):
                    df[c] = converted.copy()
            except Exception:
                pass

    # Find columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns found in text file.")
        
    # Look for voltage/signal columns
    voltage_col = None
    time_col = None
    
    for col in numeric_cols:
        col_str = str(col).lower()
        if 'volt' in col_str or 'ecg' in col_str or 'sig' in col_str or 'lead' in col_str or 'ekg' in col_str:
            voltage_col = col
            break
            
    for col in numeric_cols:
        col_str = str(col).lower()
        if 'time' in col_str or 'sec' in col_str or 't' == col_str:
            time_col = col
            break
            
    if voltage_col is None:
        # If we have multiple columns and one is identified as time, choose the first other column
        other_cols = [c for c in numeric_cols if c != time_col]
        if other_cols:
            voltage_col = other_cols[0]
        else:
            voltage_col = numeric_cols[0]
        
    sig = df[voltage_col].values.copy()
    
    if time_col is not None:
        t = df[time_col].values.copy()
        # Remove any rows where either t or sig is NaN to keep them aligned
        valid_mask = np.isfinite(t) & np.isfinite(sig)
        if not valid_mask.all():
            t = t[valid_mask]
            sig = sig[valid_mask]
        # Estimate sampling frequency from time difference
        if len(t) > 1:
            diffs = np.diff(t)
            mean_diff = np.mean(diffs[diffs > 0])
            detected_fs = int(round(1.0 / mean_diff))
            return t, sig, detected_fs
    else:
        # Clean sig if it has NaNs
        valid_mask = np.isfinite(sig)
        if not valid_mask.all():
            sig = sig[valid_mask]
        t = np.arange(len(sig)) / default_fs
        
    return t, sig, default_fs

def _load_mat_file(filepath, default_fs):
    """
    Loads MATLAB .mat files. Scans variables to find the ECG array.
    """
    try:
        mat = sio.loadmat(filepath)
    except Exception as e:
        raise ValueError(f"Failed to parse MATLAB file: {e}")
        
    # Scan keys to find numeric vectors
    candidate_key = None
    candidate_len = 0
    
    for key, val in mat.items():
        if key.startswith('__'):
            continue
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
            # Flatten or find largest dimension
            val_len = val.size
            if val_len > candidate_len and val_len > 100:
                candidate_key = key
                candidate_len = val_len
                
    if candidate_key is None:
        raise ValueError("No numeric vector found in MATLAB file.")
        
    sig = mat[candidate_key].flatten()
    t = np.arange(len(sig)) / default_fs
    
    return t, sig, default_fs

def _load_dat_file(filepath, default_fs):
    """
    Loads .dat files (commonly text format or flat binary).
    Uses a robust range heuristic to detect if the binary is 16-bit integers (standard) or float32.
    """
    try:
        # Try loading as text first
        return _load_text_file(filepath, default_fs)
    except Exception:
        # Try loading as flat binary (float32 or int16)
        try:
            # MIT-BIH and other standard ECG DAT files are almost always 16-bit signed integers (int16).
            # Let's read both and check signal properties to decide.
            sig_int = np.fromfile(filepath, dtype=np.int16)
            sig_float = np.fromfile(filepath, dtype=np.float32)
            
            sig = sig_int  # default
            
            if len(sig_float) >= 100:
                max_val_float = np.max(np.abs(sig_float))
                # Heuristics for float32:
                # - shouldn't contain NaNs/Infs
                # - shouldn't be extremely large (>1e6) or extremely close to 0 (<1e-6, unless pure 0)
                has_nan_inf = np.isnan(sig_float).any() or np.isinf(sig_float).any()
                is_extreme = max_val_float > 1e6 or (max_val_float < 1e-6 and max_val_float > 0)
                
                if not has_nan_inf and not is_extreme:
                    sig = sig_float
            
            if len(sig) < 100:
                raise ValueError("Binary read returned too few samples.")
                
            t = np.arange(len(sig)) / default_fs
            return t, sig, default_fs
        except Exception as e:
            raise ValueError(f"Failed to parse binary .dat file: {e}")

def _load_edf_file(filepath, default_fs):
    """
    Pure Python parser for European Data Format (EDF) files.
    Avoids external C-libraries and provides robust extraction.
    """
    try:
        with open(filepath, 'rb') as f:
            # --- Parse Header ---
            # 8 bytes version
            version = f.read(8).decode('ascii').strip()
            # Patient and recording ID
            patient_id = f.read(80).decode('ascii').strip()
            recording_id = f.read(80).decode('ascii').strip()
            # Start date/time
            start_date = f.read(8).decode('ascii').strip()
            start_time = f.read(8).decode('ascii').strip()
            
            # Number of bytes in header
            header_bytes = int(f.read(8).decode('ascii').strip())
            # Reserved 44 bytes
            f.read(44)
            # Number of data records
            num_records = int(f.read(8).decode('ascii').strip())
            # Duration of data record in seconds
            record_duration = float(f.read(8).decode('ascii').strip())
            # Number of signals (channels)
            ns = int(f.read(4).decode('ascii').strip())
            
            # Channel details
            labels = [f.read(16).decode('ascii').strip() for _ in range(ns)]
            transducers = [f.read(80).decode('ascii').strip() for _ in range(ns)]
            phys_dims = [f.read(8).decode('ascii').strip() for _ in range(ns)]
            phys_mins = [float(f.read(8).decode('ascii').strip()) for _ in range(ns)]
            phys_maxs = [float(f.read(8).decode('ascii').strip()) for _ in range(ns)]
            dig_mins = [float(f.read(8).decode('ascii').strip()) for _ in range(ns)]
            dig_maxs = [float(f.read(8).decode('ascii').strip()) for _ in range(ns)]
            prefilters = [f.read(80).decode('ascii').strip() for _ in range(ns)]
            samples_per_record = [int(f.read(8).decode('ascii').strip()) for _ in range(ns)]
            # Reserved 32 bytes per signal
            for _ in range(ns):
                f.read(32)
                
            # --- Select ECG Signal ---
            ecg_idx = 0
            for idx, label in enumerate(labels):
                lbl_lower = label.lower()
                if 'ecg' in lbl_lower or 'ekg' in lbl_lower or 'volt' in lbl_lower or 'lead' in lbl_lower:
                    ecg_idx = idx
                    break
            
            # Sampling frequency of the selected signal
            signal_fs = int(round(samples_per_record[ecg_idx] / record_duration))
            
            # --- Read Data Blocks ---
            # Total samples for selected signal
            total_samples = num_records * samples_per_record[ecg_idx]
            sig_data = np.zeros(total_samples)
            
            # Seek to start of data
            f.seek(header_bytes)
            
            # Read records sequentially
            for r in range(num_records):
                for s in range(ns):
                    # Read 2-byte signed integers
                    num_bytes = samples_per_record[s] * 2
                    raw_bytes = f.read(num_bytes)
                    
                    if s == ecg_idx:
                        # Convert bytes to numpy 16-bit signed ints
                        record_data = np.frombuffer(raw_bytes, dtype=np.int16)
                        start_pos = r * samples_per_record[ecg_idx]
                        sig_data[start_pos : start_pos + len(record_data)] = record_data
                        
            # --- Calibrate to physical values ---
            p_min = phys_mins[ecg_idx]
            p_max = phys_maxs[ecg_idx]
            d_min = dig_mins[ecg_idx]
            d_max = dig_maxs[ecg_idx]
            
            # Formula: phys = phys_min + (dig - dig_min) * (phys_max - phys_min) / (dig_max - dig_min)
            calibrated_sig = p_min + (sig_data - d_min) * (p_max - p_min) / (d_max - d_min)
            
            t = np.arange(len(calibrated_sig)) / signal_fs
            return t, calibrated_sig, signal_fs
            
    except Exception as e:
        raise ValueError(f"Failed to parse European Data Format (EDF) file: {e}")
