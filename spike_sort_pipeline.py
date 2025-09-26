from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hdbscan import HDBSCAN
from scipy import signal
from scipy.signal import argrelmin, firwin, lfilter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MS = 1000
RAW_SCALE = 10.0

DEFAULT_CHANNELS = [
    "ch21", "ch31", "ch41", "ch51", "ch61", "ch71", "ch12",
    "ch22", "ch32", "ch42", "ch52", "ch62", "ch72", "ch82", "ch13",
    "ch23", "ch33", "ch43", "ch53", "ch63", "ch73", "ch83", "ch14",
    "ch24", "ch34", "ch44", "ch54", "ch64", "ch74", "ch84", "ch15",
    "ch25", "ch35", "ch45", "ch55", "ch65", "ch75", "ch85", "ch16",
    "ch26", "ch36", "ch46", "ch56", "ch66", "ch76", "ch86", "ch17",
    "ch27", "ch37", "ch47", "ch57", "ch67", "ch77", "ch87", "ch28",
    "ch38", "ch48", "ch58", "ch68", "ch78",
]

COLOR = [
    "b", "chartreuse", "r", "c", "m", "y", "k", "Brown", "ForestGreen",
    "darkcyan", "maroon", "orange", "green", "steelblue", "purple",
    "gold", "navy", "gray", "indigo", "black", "darkgoldenrod",
]


@dataclass
class SortingConfig:
    fs: int = 20_000
    band_bottom: int = 300
    band_top: int = 3_000
    band_numtaps: int = 255
    spike_threshold_sd: float = 4.0
    spike_order: int = 15
    spike_polarity: int = -1
    window_before_ms: float = 1.0
    window_after_ms: float = 2.0
    cut_area: int = 13
    pca_components: int = 2
    cluster_min_size: int = 2_000
    cluster_min_samples: int = 250
    template_merge_score: int = 115
    noise_reassign_score: int = 72
    ch_array: np.ndarray = field(default_factory=lambda: np.array(DEFAULT_CHANNELS, dtype=str))


@dataclass
class ChannelMeta:
    data_name: str
    channel_label: str
    figure_dirs: dict[str, Path]
    h5_path: Path
    log_path: Path
    sampling_rate: int


def load_raw_matrix(path: Path | str, n_channels: int, dtype: str = "h", scale: float = RAW_SCALE) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(path, dtype=dtype)
    if data.size == 0:
        raise ValueError(f"RAW file {path} is empty")
    if data.size % n_channels:
        raise ValueError(f"RAW size {data.size} not divisible by n_channels={n_channels}")
    matrix = data.reshape(-1, n_channels).astype(np.float32)
    if scale:
        matrix = matrix / np.float32(scale)
    return matrix


def extract_channel(raw_matrix: np.ndarray, ch_label: str, ch_array: Sequence[str]) -> np.ndarray:
    if raw_matrix.ndim != 2:
        raise ValueError("raw_matrix must be 2D")
    ch_array = np.asarray(ch_array)
    if raw_matrix.shape[1] != ch_array.size:
        raise ValueError(
            f"Mismatch between raw_matrix columns ({raw_matrix.shape[1]}) and channel list ({ch_array.size})"
        )
    mask = ch_array == ch_label
    if not np.any(mask):
        raise ValueError(f"{ch_label} is not present in the provided channel list")
    return raw_matrix[:, mask][:, 0]


def BandPassFilter(
    wave_raw: np.ndarray,
    bottom: float,
    top: float,
    sampling_rate: int,
    numtaps: int,
) -> np.ndarray:
    if wave_raw.size == 0:
        return np.empty(0, dtype=np.float32)
    nyq = sampling_rate / 2
    cutoff = np.array([bottom, top]) / nyq
    taps = firwin(numtaps, cutoff, pass_zero=False)
    filtered = lfilter(taps, 1, wave_raw)
    delay = int((numtaps - 1) / 2)
    if delay:
        filtered = filtered[delay:]
    return filtered.astype(np.float32)


def SpikeDetection(wave_filtered: np.ndarray, sd_thr: float, order: int, spike: int) -> np.ndarray:
    if wave_filtered.size == 0:
        return np.empty(0, dtype=int)
    peaks = argrelmin(-1 * spike * wave_filtered, order=order)[0]
    if peaks.size == 0:
        return np.empty(0, dtype=int)
    median = np.median(wave_filtered)
    mad = np.median(np.abs(wave_filtered - median)) / 0.6745
    threshold = median - sd_thr * mad
    spike_index = peaks[wave_filtered[peaks] < threshold]
    return spike_index.astype(int)


def GetWaveShape(
    spike_index: np.ndarray,
    wave_filtered: np.ndarray,
    area_before_peak_ms: float,
    area_after_peak_ms: float,
    sampling_rate: int,
    ms: int = MS,
) -> tuple[np.ndarray, np.ndarray]:
    if spike_index.size == 0:
        return np.empty((0, 0), dtype=np.float32), spike_index
    before = int(area_before_peak_ms * sampling_rate / ms)
    after = int(area_after_peak_ms * sampling_rate / ms)
    windows = np.column_stack((spike_index - before, spike_index + after))
    valid_mask = (windows[:, 0] >= 0) & (windows[:, 1] < wave_filtered.size)
    if not np.any(valid_mask):
        return np.empty((0, 0), dtype=np.float32), spike_index[:0]
    windows = windows[valid_mask]
    spike_index = spike_index[valid_mask]
    waveforms = np.array(
        [wave_filtered[start:end] for start, end in windows],
        dtype=np.float32,
    )
    return waveforms, spike_index


def CutWaveShape(spike_shape: np.ndarray, area: int) -> np.ndarray:
    if spike_shape.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    length = spike_shape.shape[1]
    center = length // 3
    start = max(0, center - area)
    stop = min(length, center + 2 * area + 1)
    roi = np.arange(start, stop)
    return spike_shape[:, roi]


def DimensionalityReductionWithDiffs(waveforms_roi: np.ndarray, n_comp: int) -> tuple[np.ndarray, np.ndarray]:
    if waveforms_roi.size == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32)
    diff1 = np.diff(waveforms_roi, n=1, axis=1)
    if waveforms_roi.shape[1] >= 3:
        diff2 = np.diff(waveforms_roi, n=2, axis=1)
        features_diff = np.concatenate([diff1, diff2], axis=1)
    else:
        features_diff = diff1
    features_diff = features_diff.astype(np.float32, copy=False)
    n_components = int(min(max(1, n_comp), features_diff.shape[0], features_diff.shape[1]))
    if n_components == 0:
        return np.empty((features_diff.shape[0], 0), dtype=np.float32), np.empty(0, dtype=np.float32)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(features_diff)
    variance = pca.explained_variance_ratio_
    return X_pca.astype(np.float32), variance.astype(np.float32)


def ClusteringWithHDBSCAN(
    spike_feature: np.ndarray,
    clu_size: int,
    min_sam: int,
    cor_num: int = 1,
    lea_siz: int = 100,
) -> np.ndarray:
    n_samples = spike_feature.shape[0]
    if n_samples == 0:
        return np.empty(0, dtype=int)
    if n_samples < 2 or spike_feature.shape[1] == 0:
        return np.full(n_samples, -1, dtype=int)
    kwargs = dict(
        min_cluster_size=clu_size,
        min_samples=min_sam,
        leaf_size=lea_siz,
        cluster_selection_method="leaf",
        core_dist_n_jobs=cor_num,
    )
    try:
        labels = HDBSCAN(**kwargs).fit_predict(spike_feature)
        if np.unique(labels).size == 1:
            raise ValueError("single cluster")
        return labels
    except ValueError:
        try:
            fallback_kwargs = dict(
                min_cluster_size=max(10, min(clu_size, max(2, n_samples // 2))),
                min_samples=max(5, min(min_sam, max(2, n_samples // 2))),
                leaf_size=lea_siz,
                core_dist_n_jobs=cor_num,
                allow_single_cluster=True,
                cluster_selection_method="leaf",
            )
            labels = HDBSCAN(**fallback_kwargs).fit_predict(spike_feature)
            if np.unique(labels).size == 1:
                raise ValueError("still single cluster")
            return labels
        except Exception:
            return np.full(n_samples, -1, dtype=int)
    except Exception:
        return np.full(n_samples, -1, dtype=int)


def MakeTemplates(clu: int, result: np.ndarray, wave_shape: np.ndarray) -> np.ndarray:
    waves = wave_shape[result == clu]
    if waves.size == 0:
        return np.zeros((2, wave_shape.shape[1]), dtype=np.float32)
    template = waves.mean(axis=0)
    template_sd = waves.std(axis=0)
    return np.vstack([template, template_sd]).astype(np.float32)


def CheckTemplate(template: np.ndarray, wave: np.ndarray) -> np.ndarray:
    temp_late, temp_late_sd = template
    temp_lower = wave > (temp_late - temp_late_sd)
    temp_upper = wave < (temp_late + temp_late_sd)
    centre_slice = slice(11, 14)
    temp_index = int(np.sum([temp_upper[centre_slice], temp_lower[centre_slice]]))
    if temp_index == 6:
        score = int(np.sum([temp_lower, temp_upper]))
        return np.array([score, temp_index]).astype(int)
    return np.array([0, temp_index]).astype(int)


def ReclustNoise(noise_wave: np.ndarray, templates: Sequence[np.ndarray], thr_score: int = 72) -> int:
    if not templates:
        return -1
    clus_score = np.array([CheckTemplate(template, noise_wave) for template in templates])
    max_index = int(np.argmax(clus_score[:, 0]))
    score, overlap = clus_score[max_index]
    if score > thr_score and overlap == 6:
        return max_index
    return -1


def MargeCluster_TM(cluster: np.ndarray, wave_shape: np.ndarray, thr_marge: int = 115) -> np.ndarray:
    if cluster.size == 0 or wave_shape.size == 0:
        return cluster
    positive_labels = np.array([label for label in np.unique(cluster) if label >= 0], dtype=int)
    if positive_labels.size <= 1:
        return cluster
    templates = [MakeTemplates(int(label), cluster, wave_shape) for label in positive_labels]
    scores = np.zeros((positive_labels.size, positive_labels.size), dtype=int)
    for col, template in enumerate(templates):
        scores[:, col] = np.array([CheckTemplate(tmp, template[0])[0] for tmp in templates])
        scores[: col + 1, col] = 0
    merge_pairs = np.argwhere(scores >= thr_marge)
    new_cluster = cluster.copy()
    for src_idx, dst_idx in merge_pairs[::-1]:
        src_label = positive_labels[src_idx]
        dst_label = positive_labels[dst_idx]
        if src_label == dst_label:
            continue
        new_cluster[new_cluster == src_label] = dst_label
    return new_cluster


def RescueNoise(cluster: np.ndarray, wave_shape: np.ndarray, thr_noise: int = 72) -> np.ndarray:
    if cluster.size == 0 or wave_shape.size == 0:
        return cluster
    noise_index = np.where(cluster == -1)[0]
    if noise_index.size == 0:
        return cluster
    positive_labels = np.array([label for label in np.unique(cluster) if label >= 0], dtype=int)
    if positive_labels.size == 0:
        return cluster
    templates = [MakeTemplates(int(label), cluster, wave_shape) for label in positive_labels]
    noise_waves = wave_shape[noise_index]
    reassign = np.array([ReclustNoise(wave, templates, thr_noise) for wave in noise_waves], dtype=int)
    new_cluster = cluster.copy()
    valid_mask = reassign >= 0
    if valid_mask.any():
        new_cluster[noise_index[valid_mask]] = positive_labels[reassign[valid_mask]]
    return new_cluster


def CalcACR(spike_time: np.ndarray) -> np.ndarray:
    window_auto = 1000
    bin_width = 1
    bin_num = int(((window_auto * 2) / bin_width) + 1)
    hist_auto = np.zeros(bin_num)
    for mid_search in spike_time:
        left_end = mid_search - window_auto
        right_end = mid_search + window_auto
        index_search = np.where((spike_time >= left_end) & (spike_time <= right_end))[0]
        temp_spike_time = spike_time[index_search] - mid_search
        hist_auto += np.histogram(temp_spike_time, bins=bin_num, range=(-window_auto, window_auto))[0]
    return hist_auto


def JudgeAcr(x_axis: np.ndarray, acr: np.ndarray) -> float:
    all_mask = (x_axis >= -200) & (x_axis <= 200)
    all_acr = acr[all_mask]
    search_positions = []
    for offset in (-2, -1, 1, 2):
        idx = np.where(x_axis == offset)[0]
        if idx.size:
            search_positions.append(idx[0])
    if not search_positions or np.sum(all_acr) == 0:
        return 0.0
    search_acr = acr[search_positions]
    return float(np.sum(search_acr) / np.sum(all_acr) * 100)


def CalcPOW(acr: np.ndarray) -> np.ndarray:
    sampling_rate = 1000
    freq, power = signal.periodogram(x=acr, fs=sampling_rate)
    return np.column_stack((freq, power))


def process_channel(raw_wave: np.ndarray, cfg: SortingConfig) -> dict[str, np.ndarray]:
    filtered = BandPassFilter(
        raw_wave,
        bottom=cfg.band_bottom,
        top=cfg.band_top,
        sampling_rate=cfg.fs,
        numtaps=cfg.band_numtaps,
    )
    spike_idx = SpikeDetection(
        filtered,
        sd_thr=cfg.spike_threshold_sd,
        order=cfg.spike_order,
        spike=cfg.spike_polarity,
    )
    waveforms, spike_idx = GetWaveShape(
        spike_idx,
        filtered,
        area_before_peak_ms=cfg.window_before_ms,
        area_after_peak_ms=cfg.window_after_ms,
        sampling_rate=cfg.fs,
    )
    if spike_idx.size == 0 or waveforms.size == 0:
        empty_variance = np.zeros(cfg.pca_components, dtype=np.float32)
        return {
            "filtered": filtered,
            "spike_idx": spike_idx,
            "waveforms": np.empty((0, 0), dtype=np.float32),
            "waveforms_roi": np.empty((0, 0), dtype=np.float32),
            "features": np.empty((0, cfg.pca_components), dtype=np.float32),
            "variance": empty_variance,
            "labels": np.empty(0, dtype=int),
            "spike_times_ms": np.array([], dtype=np.float32),
            "isi": np.empty((0, 4), dtype=np.float32),
        }
    waveforms_roi = CutWaveShape(waveforms, area=cfg.cut_area)
    if waveforms_roi.size == 0:
        x_pca = np.empty((waveforms.shape[0], 0), dtype=np.float32)
        variance = np.zeros(cfg.pca_components, dtype=np.float32)
    else:
        x_pca, variance = DimensionalityReductionWithDiffs(waveforms_roi, cfg.pca_components)
    if x_pca.size == 0 or x_pca.shape[1] == 0:
        features = np.empty((waveforms_roi.shape[0], 0), dtype=np.float32)
        clusters = np.full(spike_idx.shape, -1, dtype=int)
    elif x_pca.shape[0] >= 2 and x_pca.shape[1] >= 1:
        features = StandardScaler().fit_transform(x_pca)
        clusters = ClusteringWithHDBSCAN(features, cfg.cluster_min_size, cfg.cluster_min_samples)
    else:
        features = x_pca - x_pca.mean(axis=0, keepdims=True)
        clusters = np.full(spike_idx.shape, -1, dtype=int)
    merged = MargeCluster_TM(clusters, waveforms, cfg.template_merge_score)
    refined = RescueNoise(merged, waveforms_roi, cfg.noise_reassign_score)
    spike_times_ms = spike_idx / (cfg.fs / MS)
    isi = np.c_[
        np.arange(1, spike_times_ms.size + 1),
        spike_times_ms,
        np.diff(np.r_[0.0, spike_times_ms]),
        refined,
    ]
    return {
        "filtered": filtered,
        "spike_idx": spike_idx,
        "waveforms": waveforms,
        "waveforms_roi": waveforms_roi,
        "features": features,
        "variance": variance,
        "labels": refined,
        "spike_times_ms": spike_times_ms,
        "isi": isi,
    }


def visualize_and_save(meta: ChannelMeta, result: dict[str, np.ndarray]) -> dict[str, int]:
    filtered = result["filtered"]
    spike_idx = result["spike_idx"]
    waveforms = result["waveforms"]
    waveforms_roi = result["waveforms_roi"]
    features = result["features"]
    labels = result["labels"]
    spike_times_ms = result["spike_times_ms"]
    variance = result["variance"]
    isi = result["isi"]

    total_spikes = int(spike_idx.size)
    if labels.size:
        unique_labels, counts = np.unique(labels, return_counts=True)
    else:
        unique_labels = np.array([], dtype=int)
        counts = np.array([], dtype=int)
    positive_labels = unique_labels[unique_labels >= 0]
    log_lines = [
        f"[{meta.channel_label}]",
        f"total_spikes: {total_spikes}",
        f"detected_clusters: {int(positive_labels.size)}",
    ]
    if variance.size and total_spikes:
        log_lines.append(
            "pca_variance: " + ", ".join(f"{v:.4f}" for v in np.atleast_1d(variance))
        )
    if unique_labels.size:
        log_lines.append("cluster_counts:")
        for clu, count in zip(unique_labels, counts):
            label_name = f"cluster {int(clu)}" if clu >= 0 else "noise"
            log_lines.append(f"  {label_name}: {int(count)}")
    else:
        log_lines.append("cluster_counts: none")
    acr_logs: list[str] = []

    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.arange(filtered.size) / meta.sampling_rate
    ax.plot(t, filtered, lw=0.5, color="steelblue")
    if spike_idx.size:
        ax.plot(spike_idx / meta.sampling_rate, filtered[spike_idx], "r.", ms=3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Filtered (uV)")
    ax.set_title(f"{meta.data_name} {meta.channel_label} filtered")
    fig.tight_layout()
    fig.savefig(meta.figure_dirs["spike_detect"] / f"{meta.data_name}_{meta.channel_label}_spike_detect.png")
    plt.close(fig)

    if features.size and features.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(features[:, 0], features[:, 1], s=5, c="gray", alpha=0.5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA scatter {meta.data_name}_{meta.channel_label}")
        fig.tight_layout()
        fig.savefig(meta.figure_dirs["pca"] / f"{meta.data_name}_{meta.channel_label}_pca_raw.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        for clu, color in zip(np.unique(labels), itertools.cycle(COLOR)):
            mask = labels == clu
            if not mask.any():
                continue
            ax.scatter(features[mask, 0], features[mask, 1], s=8, alpha=0.7, label=f"clu {clu}", c=color)
        ax.legend(fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA clustered {meta.data_name}_{meta.channel_label}")
        fig.tight_layout()
        fig.savefig(meta.figure_dirs["pca"] / f"{meta.data_name}_{meta.channel_label}_pca_cluster.png")
        plt.close(fig)

    for clu, color in zip(np.unique(labels), itertools.cycle(COLOR)):
        mask = labels == clu
        if clu < 0 or not mask.any():
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(waveforms[mask].T, color=color, alpha=0.2, lw=0.5)
        ax.plot(np.median(waveforms[mask], axis=0), color=color, lw=2)
        ax.set_title(f"{meta.channel_label} cluster {clu} (n={mask.sum()})")
        ax.set_xlabel("Samples")
        ax.set_ylabel("uV")
        fig.tight_layout()
        fig.savefig(meta.figure_dirs["sorting_cluster"] / f"{meta.data_name}_{meta.channel_label}_cluster{clu}_waveforms.png")
        plt.close(fig)

    x_axis = np.arange(-1000, 1001)
    for clu, color in zip(np.unique(labels), itertools.cycle(COLOR)):
        mask = labels == clu
        if clu < 0 or not mask.any():
            continue
        acr = CalcACR(spike_times_ms[mask])
        fire_index = JudgeAcr(x_axis, acr)
        status = "PASS" if fire_index <= 1.0 else "FAIL"
        acr_logs.append(
            f"  cluster {int(clu)}: n={int(mask.sum())}, fire_index={fire_index:.3f}% ({status})"
        )

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x_axis, acr, color="black", lw=1)
        ax.set_xlim(-200, 200)
        ax.set_xlabel("Time lag [ms]")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"{meta.channel_label} cluster {clu} autocorrelogram")
        fig.tight_layout()
        fig.savefig(meta.figure_dirs["auto_correlo"] / f"{meta.data_name}_{meta.channel_label}_cluster{clu}_acr.png")
        plt.close(fig)

        freqP = CalcPOW(acr)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(freqP[:, 0], freqP[:, 1], color="black", lw=1)
        ax.set_xlim(0, 80)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power/frequency")
        ax.set_title(f"{meta.channel_label} cluster {clu} power spectrum")
        fig.tight_layout()
        fig.savefig(meta.figure_dirs["auto_correlo"] / f"{meta.data_name}_{meta.channel_label}_cluster{clu}_power.png")
        plt.close(fig)

    datasets = {
        "labels": labels.astype(int, copy=False),
        "spike_idx": spike_idx.astype(int, copy=False),
        "spike_times_ms": spike_times_ms.astype(np.float32, copy=False),
        "waveforms": waveforms.astype(np.float32, copy=False),
        "waveforms_roi": waveforms_roi.astype(np.float32, copy=False),
        "variance": variance.astype(np.float32, copy=False),
        "isi": isi.astype(np.float32, copy=False),
    }
    with h5py.File(meta.h5_path, "a") as h5:
        group = h5.require_group(meta.channel_label)
        for name, data in datasets.items():
            if name in group:
                del group[name]
            if data.size:
                group.create_dataset(name, data=data, compression="gzip", compression_opts=4)
            else:
                group.create_dataset(name, shape=data.shape, dtype=data.dtype)

    if acr_logs:
        log_lines.append("acr_tests:")
        log_lines.extend(acr_logs)
    else:
        log_lines.append("acr_tests: n/a")

    with meta.log_path.open("a", encoding="utf-8") as fp:
        fp.write("\n".join(log_lines) + "\n\n")

    return {
        "total_spikes": total_spikes,
        "positive_clusters": int(positive_labels.size),
        "unique_labels": unique_labels.tolist(),
    }


def prepare_output(base_dir: Path) -> dict[str, Path]:
    figure_dirs = {
        "spike_detect": base_dir / "spike_detect",
        "pca": base_dir / "pca",
        "sorting_cluster": base_dir / "sorting_cluster",
        "auto_correlo": base_dir / "auto_correlo",
    }
    for path in figure_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return figure_dirs


def run_spike_sorting(
    raw_path: Path | str,
    channels: Optional[Sequence[str]] = None,
    output_root: Optional[Path | str] = None,
    cfg: Optional[SortingConfig] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    selected_channels = channels or []
    channel_array = np.array(selected_channels if selected_channels else DEFAULT_CHANNELS, dtype=str)
    if cfg is None:
        cfg = SortingConfig(ch_array=channel_array)
    else:
        cfg = replace(cfg, ch_array=channel_array)

    data_name = raw_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = Path(output_root) if output_root else raw_path.parent / "temp_results"
    base_dir = base_root / f"{data_name}_spike_sort_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    figure_dirs = prepare_output(base_dir)
    h5_path = base_dir / f"{data_name}_spike_sort.h5"
    if h5_path.exists():
        h5_path.unlink()

    log_path = base_dir / f"{data_name}_spike_sort.log"
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(f"Spike sorting log for {data_name} (created {datetime.now().isoformat()})\n\n")

    raw_matrix = load_raw_matrix(raw_path, channel_array.size)
    summary = []

    for ch_label in channel_array:
        if progress_callback:
            progress_callback(f"[{ch_label}] sorting started")
        wave = extract_channel(raw_matrix, ch_label, ch_array=channel_array)
        result = process_channel(wave, cfg)
        meta = ChannelMeta(data_name, ch_label, figure_dirs, h5_path, log_path, cfg.fs)
        channel_summary = visualize_and_save(meta, result)
        summary.append({"channel": ch_label, **channel_summary})
        if progress_callback:
            progress_callback(
                f"[{ch_label}] completed: spikes={channel_summary['total_spikes']} clusters={channel_summary['positive_clusters']}"
            )

    if progress_callback:
        progress_callback(f"Sorting finished. Results saved to {base_dir}")

    return {
        "base_dir": base_dir,
        "log_path": log_path,
        "h5_path": h5_path,
        "channels": channel_array.tolist(),
        "summary": summary,
    }
