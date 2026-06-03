# ECG-HRV Analysis System

This repository contains a professional-grade **ECG Signal Processing & Heart Rate Variability (HRV) Analysis System** implemented in Python. It is designed for multi-stage digital filtering, R-peak detection, autonomic physiological analysis, statistical feature extraction, and automated report generation.

---

## 📋 System Features

### Signal Processing & Preprocessing
*   **Baseline Wander Removal**: Implements dual median filtering (200ms and 600ms windows) to estimate and subtract respiratory drifts without attenuating QRS amplitudes or introducing phase distortion.
*   **Denoising (Bandpass Filter)**: Employs a 3rd-order Butterworth bandpass filter (0.5–45 Hz) applied via zero-phase double-filtering (`filtfilt`) to suppress high-frequency muscle artifacts and 50 Hz powerline interference.
*   **QRS R-Peak Detection**: Implements the step-by-step classic Pan-Tompkins algorithm including derivative filtering, squaring, moving integration (150ms window), and adaptive double-thresholding with back-search and refractory lockouts.

### Feature Extraction, Physiological Assessment & Synthesis
*   **Ectopic Beat Handling**: Detects outlier beats (PVCs) using localized median percentage deviation (20% threshold) and restores missing intervals using cubic spline interpolation.
*   **Linear HRV (Time-Domain)**: Computes $Mean\ RR$, $Mean\ HR$, $SDNN$, $RMSSD$, $NN50$, and $pNN50$.
*   **Linear HRV (Frequency-Domain)**: Resamples unevenly spaced RR intervals to 4 Hz and applies the Welch Periodogram to integrate power in Very Low (VLF, 0.00–0.04 Hz), Low (LF, 0.04–0.15 Hz), and High (HF, 0.15–0.40 Hz) frequency bands to compute the sympathovagal balance ($LF/HF$ Ratio).
*   **Non-Linear HRV**: Calculates Poincaré plot geometry ($SD1$ and $SD2$) and computes rhythm complexity via Sample Entropy ($SampEn$).
*   **Reporting**: Automatically compiles clinical tables, physiological discussion, and 7 figures into printable PDF and editable Word DOCX reports.

---

## ⚙️ Mathematical Formulations & Filter Design

### 1. Baseline Wander Removal (Dual Median Filter)
Respiration causes a low-frequency wander $x_{drift}[n]$ added to the true ECG $x_{true}[n]$. It is estimated non-linearly:
$$x_{drift}[n] = \text{Median}_{600\text{ms}}\left( \text{Median}_{200\text{ms}}\left( x_{raw}[n] \right) \right)$$
$$x_{clean}[n] = x_{raw}[n] - x_{drift}[n]$$

### 2. Butterworth Bandpass Filter
A 3rd-order Butterworth filter blocks frequencies outside the $0.5 \text{ Hz} \le f \le 45 \text{ Hz}$ range. To achieve zero phase shift, we pass the signal forward and then backward through the filter:
$$y_{f}[n] = \sum_{i=0}^{N} b_i x[n-i] - \sum_{j=1}^{M} a_j y_{f}[n-j]$$
$$y_{final}[n] = y_{f, reversed}[n] \ast h_{butter}$$
This ensures QRS peaks are not shifted in time (a critical requirement for calculating exact RR intervals).

### 3. Pan-Tompkins QRS Detection
1.  **Derivative Filter**: Highlights the high-frequency slope of the QRS complex:
    $$y[n] = \frac{1}{8} \left( 2x[n] + x[n-1] - x[n-3] - 2x[n-4] \right)$$
2.  **Squaring**: Magnifies the QRS slopes and makes all values positive:
    $$y[n] = x[n]^2$$
3.  **Moving Integration**: Evaluates the energy envelope of the QRS complex using a window size $N$ (~150ms):
    $$y[n] = \frac{1}{N} \sum_{i=0}^{N-1} x[n-i]$$
4.  **Adaptive Thresholding**:
    R-peaks are identified by scanning the integrated envelope. Two thresholds are maintained adaptively:
    $$THR_1 = NPKI + 0.25 \times (SPKI - NPKI)$$
    Where $SPKI$ is the running estimate of the signal peak level, and $NPKI$ is the running estimate of the noise peak level. A back-search is triggered if no peaks occur within $1.66 \times$ the running average RR interval, utilizing a lower threshold:
    $$THR_2 = 0.5 \times THR_1$$

### 4. Ectopic Beat Correction
Ventricular ectopic beats (like PVCs) introduce artificially short RR intervals followed by long compensatory pauses. They are detected using a rolling median window:
$$\text{If } |RR[i] - \text{Median}_{15}(RR)| > 0.20 \times \text{Median}_{15}(RR) \implies RR[i] \text{ is ectopic}$$
Ectopic indices are replaced using cubic spline interpolation fitted to normal neighboring beats.

### 5. Poincaré Non-Linear Parameters
SD1 represents short-term variation (dominated by parasympathetic vagal activity). SD2 represents long-term overall variation:
$$SD1 = \sqrt{\frac{1}{2} \text{Var}(RR_n - RR_{n+1})}$$
$$SD2 = \sqrt{2 \text{Var}(RR_n) - \frac{1}{2} \text{Var}(RR_n - RR_{n+1})}$$

---

## 🛠️ System Architecture

*   **`core/synthetic_data.py`**: Generates realistic multi-file ECG waveforms (NSR, AFib, PVC, VTach) at 250 Hz with customizable respiratory drift and high-frequency noise.
*   **`core/signal_processing.py`**: Implements Butterworth filters, dual median filters, Savitzky-Golay filters, and the Pan-Tompkins QRS R-peak detector.
*   **`core/hrv_analysis.py`**: Implements time-domain, frequency-domain (Welch PSD), non-linear HRV, and clinical interpretation text engines.
*   **`core/report_generator.py`**: Compiles tables and matplotlib plots into PDF reports (via `reportlab`) and Word documents (via `python-docx`).
*   **`app.py`**: Web-based Streamlit GUI allowing users to upload data, adjust filters, check signal envelopes, examine Poincaré plots, and download PDF/DOCX reports.
*   **`cli.py`**: Command-line batch script to process folders of ECG files.

---

## 🚀 Installation & Execution

### 1. Install Dependencies
Make sure Python is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Web Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
This opens a modern, responsive web dashboard where you can generate synthetic cardiac signals, upload custom patient logs, visualize the filtering steps, inspect ectopic corrections, and download report templates.

### 3. Run Batch Processing via CLI
To process a folder of ECG files at once:
1.  **Generate Test Data**:
    ```bash
    python cli.py --generate-test --outdir my_results
    ```
    This creates multiple CSV files representing NSR, AFib, PVC, and VTach signals in `my_results/synthetic_test_data/`.
2.  **Run Pipeline & Compile Reports**:
    ```bash
    python cli.py --input my_results/synthetic_test_data --outdir my_results
    ```
    This processes all CSV files in the folder, plots the 7 mandatory figures, saves them under `my_results/`, and compiles detailed PDF and DOCX reports for each rhythm.
