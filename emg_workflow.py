"""
Standalone EMG workflow script tailored for four-channel DC-preserving recordings.

Key features
- Loads Excel files with TIME + EMGA–EMGD columns and converts Excel times to seconds starting at 0.
- Keeps a raw copy; builds DC trend "ladder" (<0.1 / <0.2 / <0.5 Hz), movement-band envelope, and canonical EMG band.
- Uses MNE for filtering/PSD/spectrograms and NeuroKit2 for envelope estimation on band-passed EMG.
- Optional change-point-based segmentation (ruptures) to split baseline/transition/post based on DC trend shifts.
- Saves QC plots and a summary CSV with drift metrics, cross-channel correlations, PCA stats, and EMG bandpowers.

Customization knobs live in the CONFIG dictionary near the top of the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns
from mne.time_frequency import psd_array_welch
from sklearn.decomposition import PCA

try:  # Optional dependency for change-point detection
    import ruptures as rpt
except ImportError:  # pragma: no cover - optional
    rpt = None

# -----------------------------------------------------------------------------
# Customization knobs
# -----------------------------------------------------------------------------
CONFIG = {
    "data_path": "/Users/nipungorantla/Desktop/Oddball_DOC001/NGControlBalls.xlsx",  # Path to the Excel file
    "output_dir": "./Users/nipungorantla/Desktop/Oddball_DOC001/emgOutputs",  # Folder for plots/CSV
    "fs": 2048.0,  # Sampling rate in Hz
    "time_column": "TIME",
    "channel_columns": ["EMGA", "EMGB", "EMGC", "EMGD"],
    # DC/ultra-slow ladder (Hz)
    "trend_cutoffs": [0.1, 0.2, 0.5],
    # Low-frequency movement-like band (Hz)
    "movement_band": (0.5, 5.0),
    # Canonical EMG band for activation metrics (Hz)
    "emg_band": (20.0, 250.0),
    # Envelope smoothing window (seconds)
    "envelope_smooth_sec": 0.200,
    # Spectrogram parameters
    "spectrogram_nperseg": 512,
    "spectrogram_noverlap": 256,
    # Change-point detection parameters (Pelt in ruptures)
    "use_changepoint": True,
    "changepoint_penalty": 5.0,
    # Transition window (seconds) around change-point
    "transition_half_window": 1.0,
    # Welch PSD parameters
    "psd_nperseg": 1024,
    "psd_noverlap": 512,
}

BAND_DEFINITIONS = {
    "emg_total": CONFIG["emg_band"],
    "low_emg": (20.0, 80.0),
    "mid_emg": (80.0, 150.0),
    "high_emg": (150.0, 250.0),
}


@dataclass
class ChannelDecomposition:
    name: str
    raw: np.ndarray
    trends: Dict[float, np.ndarray]
    movement_band: np.ndarray
    emg_band: np.ndarray
    envelope: np.ndarray
    psd_freqs: np.ndarray
    psd_values: np.ndarray
    spectrogram: Dict[str, np.ndarray]
    bandpowers: Dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# I/O and time handling
# -----------------------------------------------------------------------------
def load_excel(filepath: Path, time_column: str, channel_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Excel EMG data and convert TIME to seconds starting at zero.

    Handles Excel time objects by converting to timedeltas. Any absolute
    datetimes are also converted by subtracting the first timestamp.
    """

    df = pd.read_excel(filepath)
    if time_column not in df:
        raise ValueError(f"Expected TIME column '{time_column}' not found.")

    time_raw = df[time_column]
    if np.issubdtype(time_raw.dtype, np.number):
        # Excel stores time as fraction of day; convert to seconds
        time_sec = time_raw.astype(float) * 24 * 3600
    else:
        # Robust conversion via pandas timedeltas
        try:
            time_sec = pd.to_timedelta(time_raw).dt.total_seconds()
        except Exception:  # pragma: no cover - defensive
            time_sec = pd.to_datetime(time_raw).astype("int64") / 1e9

    time_sec = time_sec - time_sec.iloc[0]
    data = df[channel_columns].to_numpy(dtype=float)
    return time_sec.to_numpy(), data


# -----------------------------------------------------------------------------
# Signal processing helpers (MNE / NeuroKit2)
# -----------------------------------------------------------------------------
def lowpass(signal: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    """Extract DC/ultra-slow trend via MNE low-pass filter (zero-phase FIR)."""

    return mne.filter.filter_data(signal, sfreq=fs, l_freq=None, h_freq=cutoff, method="fir", verbose=False)


def bandpass(signal: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray:
    """Band-pass filter using MNE without touching DC outside the band."""

    l_freq, h_freq = band
    return mne.filter.filter_data(signal, sfreq=fs, l_freq=l_freq, h_freq=h_freq, method="fir", verbose=False)


def compute_envelope(emg_band: np.ndarray, fs: float, smooth_sec: float) -> np.ndarray:
    """Rectify and smooth EMG amplitude using NeuroKit2 (preserves low-frequency envelopes)."""

    return nk.emg_amplitude(emg_band, sampling_rate=fs, method="smooth", window_size=int(smooth_sec * fs))


def compute_psd(segment: np.ndarray, fs: float, nperseg: int, noverlap: int) -> Tuple[np.ndarray, np.ndarray]:
    freqs, psd = psd_array_welch(
        segment,
        sfreq=fs,
        n_fft=nperseg,
        n_per_seg=nperseg,
        n_overlap=noverlap,
        average="mean",
        verbose=False,
    )
    return freqs, psd


def compute_bandpowers(freqs: np.ndarray, psd: np.ndarray, bands: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    bandpowers: Dict[str, float] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        bandpowers[name] = np.trapz(psd[mask], freqs[mask]) if np.any(mask) else np.nan
    return bandpowers


def compute_spectrogram(signal: np.ndarray, fs: float, nperseg: int, noverlap: int) -> Dict[str, np.ndarray]:
    freqs, times, Sxx = mne.time_frequency.spectrogram(
        signal,
        sfreq=fs,
        n_per_seg=nperseg,
        n_overlap=noverlap,
        verbose=False,
    )
    return {"freqs": freqs, "times": times, "power": Sxx}


# -----------------------------------------------------------------------------
# Segmentation based on DC trend change-point
# -----------------------------------------------------------------------------
def detect_changepoint(trend: np.ndarray, fs: float, penalty: float) -> Optional[int]:
    """Locate a single change-point in the dominant DC trend using ruptures if available."""

    if rpt is None:
        return None
    algo = rpt.Pelt(model="rbf").fit(trend)
    cps = algo.predict(pen=penalty)
    if not cps:
        return None
    # ruptures returns end indices for each segment; take first change
    cp_idx = cps[0]
    return cp_idx if 0 < cp_idx < len(trend) else None


def segment_indices(n_samples: int, fs: float, changepoint: Optional[int], half_window_sec: float) -> Dict[str, slice]:
    """Generate baseline/transition/post slices around a change-point (or fallback mid-point)."""

    if changepoint is None:
        changepoint = n_samples // 2
    half_window = int(half_window_sec * fs)
    start_trans = max(changepoint - half_window, 0)
    end_trans = min(changepoint + half_window, n_samples)
    return {
        "baseline": slice(0, start_trans),
        "transition": slice(start_trans, end_trans),
        "post": slice(end_trans, n_samples),
    }


# -----------------------------------------------------------------------------
# Metrics and PCA
# -----------------------------------------------------------------------------
def trend_slope_and_range(trend: np.ndarray, times: np.ndarray) -> Tuple[float, float]:
    if trend.size < 2:
        return np.nan, np.nan
    coef = np.polyfit(times, trend, 1)
    slope = coef[0]
    drift_range = trend.max() - trend.min()
    return slope, drift_range


def trend_correlation(trends: np.ndarray) -> float:
    if trends.shape[1] < 2:
        return np.nan
    corr_matrix = np.corrcoef(trends.T)
    upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    return float(np.nanmean(upper))


def pca_common_mode(trends: np.ndarray) -> Tuple[float, np.ndarray]:
    pca = PCA(n_components=min(3, trends.shape[1]))
    transformed = pca.fit_transform(trends)
    pc1_var = float(pca.explained_variance_ratio_[0]) if pca.explained_variance_ratio_.size else np.nan
    # Reconstruct PC1 contribution and residuals
    pc1 = np.outer(transformed[:, 0], pca.components_[0])
    residuals = trends - pc1
    return pc1_var, residuals


# -----------------------------------------------------------------------------
# Main processing pipeline
# -----------------------------------------------------------------------------
def decompose_channels(time_sec: np.ndarray, data: np.ndarray, config: dict) -> Dict[str, ChannelDecomposition]:
    fs = config["fs"]
    n_channels = data.shape[1]
    decompositions: Dict[str, ChannelDecomposition] = {}
    for idx in range(n_channels):
        name = config["channel_columns"][idx]
        raw = data[:, idx]
        trends = {cut: lowpass(raw, fs, cut) for cut in config["trend_cutoffs"]}
        movement_band = bandpass(raw, fs, config["movement_band"])
        emg_band = bandpass(raw, fs, config["emg_band"])
        envelope = compute_envelope(emg_band, fs, config["envelope_smooth_sec"])
        freqs, psd_vals = compute_psd(raw, fs, config["psd_nperseg"], config["psd_noverlap"])
        bandpowers = compute_bandpowers(freqs, psd_vals, BAND_DEFINITIONS)
        spectro = compute_spectrogram(raw, fs, config["spectrogram_nperseg"], config["spectrogram_noverlap"])
        decompositions[name] = ChannelDecomposition(
            name=name,
            raw=raw,
            trends=trends,
            movement_band=movement_band,
            emg_band=emg_band,
            envelope=envelope,
            psd_freqs=freqs,
            psd_values=psd_vals,
            spectrogram=spectro,
            bandpowers=bandpowers,
        )
    return decompositions


def build_summary(
    time_sec: np.ndarray,
    decomps: Dict[str, ChannelDecomposition],
    config: dict,
    changepoint: Optional[int],
) -> pd.DataFrame:
    fs = config["fs"]
    segments = segment_indices(len(time_sec), fs, changepoint, config["transition_half_window"])
    trend_key = max(config["trend_cutoffs"])  # richest DC ladder level

    # Stack trends for correlation/PCA
    trend_matrix = np.column_stack([decomps[ch].trends[trend_key] for ch in config["channel_columns"]])
    pc1_var, residuals = pca_common_mode(trend_matrix)

    rows = []
    for seg_name, slc in segments.items():
        seg_times = time_sec[slc]
        seg_trends = trend_matrix[slc]
        if seg_trends.shape[0] < 2:
            mean_corr = np.nan
            seg_pc1_var = np.nan
            seg_resid = np.full_like(seg_trends, np.nan)
        else:
            mean_corr = trend_correlation(seg_trends)
            seg_pc1_var, seg_resid = pca_common_mode(seg_trends)
        for idx, ch in enumerate(config["channel_columns"]):
            ch_trend = seg_trends[:, idx]
            slope, drift_range = trend_slope_and_range(ch_trend, seg_times)
            ch_emg_seg = decomps[ch].emg_band[slc]
            if ch_emg_seg.size < 2:
                bandpowers = {f"bandpower_{k}": np.nan for k in BAND_DEFINITIONS}
            else:
                nperseg = min(ch_emg_seg.size, config["psd_nperseg"])
                noverlap = min(config["psd_noverlap"], max(nperseg // 2, 1))
                freqs, psd_vals = compute_psd(ch_emg_seg, fs, nperseg, noverlap)
                bandpowers = {f"bandpower_{k}": v for k, v in compute_bandpowers(freqs, psd_vals, BAND_DEFINITIONS).items()}
            rows.append(
                {
                    "channel": ch,
                    "segment": seg_name,
                    "trend_cutoff": trend_key,
                    "dc_slope_uV_per_s": slope,
                    "dc_range_uV": drift_range,
                    "mean_trend_corr": mean_corr,
                    "pc1_variance_explained": seg_pc1_var,
                    "residual_trend_std": float(np.std(seg_resid[:, idx])),
                    **{f"bandpower_{k}": v for k, v in bandpowers.items()},
                }
            )
    summary = pd.DataFrame(rows)
    summary["pc1_variance_explained_full"] = pc1_var
    return summary


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def plot_raw_with_trend(time_sec: np.ndarray, decomps: Dict[str, ChannelDecomposition], config: dict, output_dir: Path) -> None:
    sns.set(style="whitegrid")
    trend_key = max(config["trend_cutoffs"])
    fig, axes = plt.subplots(len(decomps), 1, figsize=(12, 2 * len(decomps)), sharex=True)
    if len(decomps) == 1:
        axes = [axes]
    for ax, (name, decomp) in zip(axes, decomps.items()):
        ax.plot(time_sec, decomp.raw, color="gray", linewidth=0.6, label="Raw")
        ax.plot(time_sec, decomp.trends[trend_key], color="red", linewidth=1.2, label=f"Trend <{trend_key} Hz")
        ax.set_ylabel(name)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw EMG with DC Trend Overlay")
    fig.tight_layout()
    fig.savefig(output_dir / "raw_with_trend.png", dpi=200)
    plt.close(fig)


def plot_trend_ladder(time_sec: np.ndarray, decomps: Dict[str, ChannelDecomposition], config: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(len(decomps), 1, figsize=(12, 2 * len(decomps)), sharex=True)
    if len(decomps) == 1:
        axes = [axes]
    for ax, (name, decomp) in zip(axes, decomps.items()):
        for cut, trend in decomp.trends.items():
            ax.plot(time_sec, trend, label=f"<{cut} Hz")
        ax.set_ylabel(name)
        ax.legend(loc="upper right", ncol=len(decomp.trends))
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("DC/Ultra-Slow Trend Ladder")
    fig.tight_layout()
    fig.savefig(output_dir / "trend_ladder.png", dpi=200)
    plt.close(fig)


def plot_psd(decomps: Dict[str, ChannelDecomposition], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, decomp in decomps.items():
        ax.semilogy(decomp.psd_freqs, decomp.psd_values, label=name)
    ax.set_xlim(0, max(CONFIG["emg_band"][1], 300))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V^2/Hz)")
    ax.legend()
    ax.set_title("Power Spectral Density")
    fig.tight_layout()
    fig.savefig(output_dir / "psd.png", dpi=200)
    plt.close(fig)


def plot_spectrogram(time_sec: np.ndarray, decomps: Dict[str, ChannelDecomposition], output_dir: Path) -> None:
    for name, decomp in decomps.items():
        spec = decomp.spectrogram
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.pcolormesh(spec["times"], spec["freqs"], 10 * np.log10(spec["power"]), shading="auto")
        ax.set_title(f"Spectrogram - {name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Power (dB)")
        fig.tight_layout()
        fig.savefig(output_dir / f"spectrogram_{name}.png", dpi=200)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def main(config: dict = CONFIG) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    time_sec, data = load_excel(Path(config["data_path"]), config["time_column"], config["channel_columns"])
    decomps = decompose_channels(time_sec, data, config)

    # Common trend (use highest cutoff)
    trend_key = max(config["trend_cutoffs"])
    common_trend = np.mean([decomps[ch].trends[trend_key] for ch in config["channel_columns"]], axis=0)

    changepoint = None
    if config["use_changepoint"]:
        changepoint = detect_changepoint(common_trend, config["fs"], config["changepoint_penalty"])

    summary = build_summary(time_sec, decomps, config, changepoint)
    summary_path = output_dir / "emg_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plots
    plot_raw_with_trend(time_sec, decomps, config, output_dir)
    plot_trend_ladder(time_sec, decomps, config, output_dir)
    plot_psd(decomps, output_dir)
    plot_spectrogram(time_sec, decomps, output_dir)

    # Save metadata
    metadata = {
        "config": config,
        "changepoint_index": changepoint,
        "summary_path": str(summary_path),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Processed data saved to {output_dir}")
    if changepoint is None:
        print("Change-point detection unavailable or no change found; using midpoint for transition.")
    else:
        print(f"Detected change-point at sample {changepoint} (~{changepoint / config['fs']:.2f} s).")


if __name__ == "__main__":
    main()
