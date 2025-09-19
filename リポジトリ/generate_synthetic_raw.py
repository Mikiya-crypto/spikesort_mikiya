# -*- coding: utf-8 -*-
"""
Synthetic RAW generator for extracellular recordings.

Format: int16 little-endian, interleaved by channel (C-order), no header.
Default: mono channel (n_channels=1).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class SynthConfig:
    fs: int = 20000           # Sampling rate [Hz]
    duration_s: float = 10.0  # Duration [s]
    n_channels: int = 1       # Number of channels
    noise_std_uV: float = 10.0  # Baseline noise std in microvolts
    spike_rate_hz: float = 3.0  # Poisson spike rate per second per unit
    n_units: int = 3          # Number of synthetic units
    spike_amp_uV: float = 120.0  # Peak amplitude of spikes
    refractory_ms: float = 1.5   # Refractory window for spike trains
    seed: Optional[int] = 13


def _make_spike_waveform(fs: int, amp_uV: float) -> np.ndarray:
    """Return a canonical biphasic spike (~1.2 ms) sampled at fs."""
    dur_ms = 1.2
    n = int(fs * dur_ms / 1000.0)
    t = np.linspace(0, dur_ms / 1000.0, n, endpoint=False, dtype=np.float32)
    # Simple difference of Gaussians to imitate biphasic shape
    w = -np.exp(-((t - 0.3e-3) ** 2) / (2 * (0.12e-3) ** 2))
    w += 0.6 * np.exp(-((t - 0.7e-3) ** 2) / (2 * (0.18e-3) ** 2))
    w = w / np.max(np.abs(w)) * amp_uV
    return w.astype(np.float32)


def _poisson_spike_train(rate_hz: float, T: float, refractory_s: float, rng: np.random.Generator) -> np.ndarray:
    """Generate Poisson spikes with absolute refractory period."""
    spikes = []
    t = 0.0
    if rate_hz <= 0:
        return np.array([], dtype=np.float64)
    while t < T:
        t += rng.exponential(1.0 / rate_hz)
        if spikes and (t - spikes[-1]) < refractory_s:
            continue
        if t < T:
            spikes.append(t)
    return np.array(spikes, dtype=np.float64)


def generate_synthetic_raw(cfg: SynthConfig) -> np.ndarray:
    """Return (n_samples, n_channels) float32 microvolt signal with spikes + noise."""
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.fs * cfg.duration_s)
    data = rng.normal(0.0, cfg.noise_std_uV, size=(n, cfg.n_channels)).astype(np.float32)

    w = _make_spike_waveform(cfg.fs, cfg.spike_amp_uV)
    wlen = len(w)

    for ch in range(cfg.n_channels):
        for _ in range(cfg.n_units):
            st = _poisson_spike_train(cfg.spike_rate_hz, cfg.duration_s, cfg.refractory_ms/1000.0, rng)
            idxs = (st * cfg.fs).astype(int)
            for i in idxs:
                j = i + wlen
                if j < n:
                    data[i:j, ch] += w

    return data


def save_raw_int16_le(path: str | Path, data_uv: np.ndarray) -> None:
    """Save (n_samples, n_channels) uV float array to int16 little-endian .raw (no header)."""
    scaled = np.clip(np.round(data_uv), -32768, 32767).astype(np.int16)
    scaled.tofile(path)
