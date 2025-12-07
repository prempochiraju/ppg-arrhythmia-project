"""
create_ecg_ppg_dataset.py

Build a beat-level ECG+PPG dataset using peak-to-peak mapping.

- Uses ECG R-peaks (from annotations) and finds the nearest PPG systolic peak.
- Extracts fixed-length windows from ECG and PPG centered on the R-peak.
- Assigns 5 arrhythmia classes based on annotation symbols.
- Saves x_ecg, x_ppg, y into an .npz file.

You need:
    pip install wfdb scipy numpy

And you must set:
    RECORDS = [...]   # list of record names from your PhysioNet dataset
    DATA_DIR = "path/to/wfdb/files"
"""

import os
import numpy as np
import wfdb
import scipy.signal as sps
from typing import Dict, List, Tuple


def bandpass_filter(x: np.ndarray, fs: float,
                    low: float = 0.5, high: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """Simple Butterworth band-pass filter."""
    nyq = 0.5 * fs
    b, a = sps.butter(order, [low / nyq, high / nyq], btype="band")
    return sps.filtfilt(b, a, x)


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    mu = np.mean(x)
    sd = np.std(x) + 1e-8
    return (x - mu) / sd


def find_ecg_rpeaks(ecg: np.ndarray,
                    fs: float,
                    distance_sec: float = 0.25) -> np.ndarray:
    """
    Very simple R-peak detector using scipy.signal.find_peaks.
    For serious use, replace with Pan-Tompkins or annotation-based peaks.
    """
    distance = int(distance_sec * fs)
    # invert if R is negative in your data
    peaks, _ = sps.find_peaks(ecg, distance=distance)
    return peaks


def find_ppg_systolic_peaks(ppg: np.ndarray,
                            fs: float,
                            distance_sec: float = 0.4) -> np.ndarray:
    """Detect main PPG systolic peaks."""
    distance = int(distance_sec * fs)
    peaks, _ = sps.find_peaks(ppg, distance=distance)
    return peaks


def match_ecg_to_ppg_peaks(r_peaks: np.ndarray,
                           ppg_peaks: np.ndarray,
                           fs: float,
                           max_delay_sec: float = 0.7) -> List[Tuple[int, int]]:
    """
    For each ECG R-peak, find the nearest *later* PPG peak
    within [0, max_delay_sec]. Returns list of (r_idx, ppg_idx).
    """
    matched = []
    max_delay = int(max_delay_sec * fs)

    ppg_idx = 0
    for r in r_peaks:
        # move ppg_idx until peak is after R
        while ppg_idx < len(ppg_peaks) and ppg_peaks[ppg_idx] < r:
            ppg_idx += 1
        if ppg_idx >= len(ppg_peaks):
            break
        # check delay
        if 0 <= ppg_peaks[ppg_idx] - r <= max_delay:
            matched.append((r, ppg_peaks[ppg_idx]))
    return matched

"""
Example 5-class mapping (modify for your project):

0 = Normal      (N, L, R, etc.)
1 = PVC         (V, !)
2 = PAC / SVT   (S, A, supraventricular types)
3 = AFib        (annotation "AFIB" or 'F' beats if available)
4 = Other       (all remaining beats we decide to keep)

You can change this mapping easily.
"""


def build_symbol_to_class() -> Dict[str, int]:
    sym2cls = {}

    # Normal beats
    for sym in ["N", "L", "R", "e", "j"]:
        sym2cls[sym] = 0

    # PVC
    for sym in ["V", "!"]:
        sym2cls[sym] = 1

    # PAC / supraventricular
    for sym in ["A", "S"]:
        sym2cls[sym] = 2

    # AFib beats (depends on dataset; often 'F' is AFib beat)
    sym2cls["F"] = 3

    # everything else -> 4 (Other) will be handled later
    return sym2cls


def build_examples_for_record(record_name: str,
                              data_dir: str,
                              ecg_channel: int = 0,
                              ppg_channel: int = 1,
                              fs_override: float = None,
                              win_len_sec: float = 2.0
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create ECG+PPG beat windows for a single WFDB record.

    Returns:
        x_ecg: (num_beats, win_len) float32
        x_ppg: (num_beats, win_len) float32
        y:     (num_beats,) int64
    """
    rec_path = os.path.join(data_dir, record_name)

    # ---- load signals ----
    record = wfdb.rdsamp(rec_path)
    sig = record.p_signals
    fs = fs_override if fs_override is not None else record.fs

    ecg = sig[:, ecg_channel]
    ppg = sig[:, ppg_channel]

    # ---- filtering + normalization ----
    ecg_f = bandpass_filter(ecg, fs, low=0.5, high=40.0)
    ppg_f = bandpass_filter(ppg, fs, low=0.3, high=8.0)  # narrower band

    ecg_n = standardize(ecg_f)
    ppg_n = standardize(ppg_f)

    # ---- detect peaks ----
    r_peaks = find_ecg_rpeaks(ecg_n, fs)
    ppg_peaks = find_ppg_systolic_peaks(ppg_n, fs)

    matched_peaks = match_ecg_to_ppg_peaks(r_peaks, ppg_peaks, fs)

    if len(matched_peaks) == 0:
        print(f"[{record_name}] Warning: no matched peaks.")
        return np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0,), dtype=np.int64)

    # ---- load beat annotations to get symbols ----
    ann = wfdb.rdann(rec_path, "atr")  # change extension if needed
    # ann.sample: index, ann.symbol: list of symbols
    ann_samples = np.asarray(ann.sample)
    ann_syms = np.asarray(ann.symbol)

    sym2cls = build_symbol_to_class()
    win_len = int(win_len_sec * fs)
    half_win = win_len // 2

    ecg_beats = []
    ppg_beats = []
    labels = []

    for r_idx, ppg_idx in matched_peaks:
        # find the annotation closest to this R-peak
        j = np.argmin(np.abs(ann_samples - r_idx))
        sym = ann_syms[j]

        # map symbol → class id (default 4 = Other)
        cls = sym2cls.get(sym, 4)

        # center windows on R-peak (use same window for ECG & PPG)
        start = r_idx - half_win
        end = r_idx + half_win

        if start < 0 or end > len(ecg_n):
            continue  # skip incomplete window at edges

        ecg_seg = ecg_n[start:end]
        ppg_seg = ppg_n[start:end]

        if len(ecg_seg) != win_len or len(ppg_seg) != win_len:
            continue

        ecg_beats.append(ecg_seg)
        ppg_beats.append(ppg_seg)
        labels.append(cls)

    if len(labels) == 0:
        print(f"[{record_name}] Warning: no valid beat windows.")
        return np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0,), dtype=np.int64)

    x_ecg = np.stack(ecg_beats).astype(np.float32)
    x_ppg = np.stack(ppg_beats).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    print(f"[{record_name}] built {len(y)} beats.")
    return x_ecg, x_ppg, y


def main():
    # TODO: set these for your dataset
    DATA_DIR = "path/to/physionet_dataset"  # folder containing .dat, .hea, .atr
    RECORDS = [
        # e.g., "s20011", "s20021", ...
        # put your record base names here (without extension)
    ]

    all_ecg = []
    all_ppg = []
    all_labels = []

    for rec in RECORDS:
        x_ecg, x_ppg, y = build_examples_for_record(
            record_name=rec,
            data_dir=DATA_DIR,
            ecg_channel=0,    # adjust if your ECG channel index is different
            ppg_channel=1,    # adjust if your PPG channel index is different
            fs_override=None, # or set if you want to force a specific fs
            win_len_sec=2.0   # 2-second beat window; change if needed
        )

        if len(y) == 0:
            continue

        all_ecg.append(x_ecg)
        all_ppg.append(x_ppg)
        all_labels.append(y)

    if len(all_labels) == 0:
        print("No beats collected. Check DATA_DIR, RECORDS, and channel indices.")
        return

    x_ecg = np.concatenate(all_ecg, axis=0)
    x_ppg = np.concatenate(all_ppg, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print("\nFinal dataset shapes:")
    print("  x_ecg:", x_ecg.shape)
    print("  x_ppg:", x_ppg.shape)
    print("  y:    ", y.shape)
    print("  class distribution:", np.bincount(y))

    # Save as a single NPZ
    out_path = os.path.join(DATA_DIR, "ecg_ppg_beats_5class.npz")
    np.savez(out_path, x_ecg=x_ecg, x_ppg=x_ppg, y=y)
    print(f"\nSaved dataset to: {out_path}")


if __name__ == "__main__":
    main()
