# -*- coding: utf-8 -*-
"""
Utility to inspect synthetic RAW files produced by launch_synthetic_raw.pyw.

Example
-------
python plot_synthetic_raw.py --seconds 0.05 --channels 0 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_synthetic_raw import SynthConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a segment of an int16 RAW file")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to the RAW file (omit to choose via dialog)",
    )
    parser.add_argument("--fs", type=int, default=SynthConfig.fs, help="Sampling rate [Hz]")
    parser.add_argument(
        "--n-channels",
        type=int,
        default=None,
        help="Total number of interleaved channels in the file (omit to infer from RAW size)",
    )
    parser.add_argument(
        "--duration-hint",
        type=float,
        default=None,
        help="Expected recording duration [s] to guide inference (optional)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="*",
        help="Subset of channel indices (0-based) to plot; default = all channels",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Duration [s] to visualize from the given offset (omit for full length)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Starting time [s] of the segment to plot",
    )
    parser.add_argument(
        "--no-channel-gui",
        action="store_true",
        help="Do not open the channel selection dialog when inference is ambiguous",
    )
    return parser.parse_args()


def _reshape(data: np.ndarray, n_channels: int) -> np.ndarray:
    if data.size % n_channels != 0:
        raise ValueError("Data size is not divisible by the number of channels")
    return data.reshape(-1, n_channels)


def _select_channels(ch_idxs: Sequence[int] | None, n_channels: int) -> list[int]:
    if not ch_idxs:
        return list(range(n_channels))
    selected = []
    for idx in ch_idxs:
        if idx < 0 or idx >= n_channels:
            raise ValueError(f"Channel index {idx} out of range (0..{n_channels - 1})")
        selected.append(idx)
    return selected


def _ask_raw_path(initial: Path | None = None) -> Path:
    """Open a file dialog to select a RAW file and return its path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(
            "Tkinter is required for file selection; pass --path to skip the dialog."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    initial_dir = str((initial or SCRIPT_DIR).resolve())
    filetypes = [("RAW files", "*.raw"), ("All files", "*.*")]
    selected = filedialog.askopenfilename(
        title="Select RAW file",
        filetypes=filetypes,
        initialdir=initial_dir,
    )
    root.destroy()

    if not selected:
        raise RuntimeError("No RAW file selected; aborting.")
    return Path(selected)


def _candidate_channels(total_samples: int, fs: int) -> list[tuple[int, float]]:
    candidates: list[tuple[int, float]] = []
    max_channels = min(4096, total_samples)
    for n_ch in range(1, max_channels + 1):
        if total_samples % n_ch != 0:
            continue
        samples_per_channel = total_samples // n_ch
        duration = samples_per_channel / fs
        if duration <= 0.0:
            continue
        candidates.append((n_ch, duration))
    return candidates


def _score_candidates(
    candidates: list[tuple[int, float]],
    duration_hint: float | None,
) -> list[tuple[int, float, tuple[float, float, float]]]:
    target_duration = (
        duration_hint
        if duration_hint and duration_hint > 0
        else (SynthConfig.duration_s if SynthConfig.duration_s > 0 else None)
    )
    default_channels = SynthConfig.n_channels if SynthConfig.n_channels > 0 else None

    scored: list[tuple[int, float, tuple[float, float, float]]] = []
    for n_ch, duration in candidates:
        duration_penalty = 0.0 if target_duration is None else abs(duration - target_duration)
        channel_penalty = 0.0 if default_channels is None else abs(n_ch - default_channels) * 0.05
        scored.append((n_ch, duration, (duration_penalty, channel_penalty, -n_ch)))
    scored.sort(key=lambda item: item[2])
    return scored


def _prompt_channel_selection(
    scored_candidates: list[tuple[int, float, tuple[float, float, float]]],
    fs: int,
    raw_path: Path,
) -> int | None:
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception:
        return None

    # Use only the first few for display to keep UI compact
    display_candidates = scored_candidates[:10]

    root = tk.Tk()
    root.title("Select channel count")
    root.resizable(False, False)

    info = ttk.Label(
        root,
        text=(
            f"File: {raw_path.name}\n"
            f"Sampling rate: {fs} Hz\n"
            "Select a channel count or enter manually"
        ),
        justify=tk.LEFT,
        padding=10,
    )
    info.grid(row=0, column=0, columnspan=2, sticky=tk.W)

    listbox = tk.Listbox(root, height=len(display_candidates), width=32, exportselection=False)
    for idx, (n_ch, duration, _) in enumerate(display_candidates, start=1):
        label = f"{n_ch} channels (~{duration:.2f} s)"
        listbox.insert(tk.END, label)
    listbox.grid(row=1, column=0, columnspan=2, padx=10, pady=4, sticky=tk.EW)
    if display_candidates:
        listbox.selection_set(0)

    ttk.Label(root, text="Manual override:").grid(row=2, column=0, padx=10, pady=(8, 2), sticky=tk.W)
    manual_var = tk.StringVar()
    entry = ttk.Entry(root, textvariable=manual_var, width=10)
    entry.grid(row=2, column=1, padx=10, pady=(8, 2), sticky=tk.W)

    result: dict[str, int | None] = {"value": None}

    def on_ok() -> None:
        selection = listbox.curselection()
        if selection:
            result["value"] = display_candidates[selection[0]][0]
        manual = manual_var.get().strip()
        if manual:
            try:
                value = int(manual)
            except ValueError:
                messagebox.showerror("Invalid input", "Channel count must be an integer")
                return
            if value <= 0:
                messagebox.showerror("Invalid input", "Channel count must be positive")
                return
            result["value"] = value
        if result["value"] is None:
            messagebox.showinfo("Selection required", "Please choose a channel count.")
            return
        root.destroy()

    def on_cancel() -> None:
        result["value"] = None
        root.destroy()

    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
    ttk.Button(btn_frame, text="OK", command=on_ok, width=10).pack(side=tk.LEFT, padx=6)
    ttk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=6)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return result["value"]


def _infer_channels(
    total_samples: int,
    fs: int,
    duration_hint: float | None,
    allow_gui: bool,
    raw_path: Path,
) -> tuple[int, float, list[tuple[int, float]]]:
    if fs <= 0:
        raise ValueError("Sampling rate must be positive to infer channel count")

    candidates = _candidate_channels(total_samples, fs)
    if not candidates:
        raise ValueError("Could not determine channel count from RAW size")

    scored = _score_candidates(candidates, duration_hint)
    best_n_ch, best_duration, _ = scored[0]

    chosen = best_n_ch
    if allow_gui and len(scored) > 1:
        selection = _prompt_channel_selection(scored, fs, raw_path)
        if selection:
            chosen = selection
            # recompute duration for selected value
            samples_per_channel = total_samples // chosen
            best_duration = samples_per_channel / fs

    alternatives = [(n_ch, duration) for n_ch, duration, _ in scored[1:6]]
    return chosen, best_duration, alternatives

def main() -> None:
    args = _parse_args()
    raw_path = args.path
    if raw_path is None:
        raw_path = _ask_raw_path()
    raw_path = raw_path.expanduser()
    raw = np.fromfile(raw_path, dtype="<i2")  # int16 little-endian
    total_samples = raw.size

    if args.n_channels is None:
        n_channels, duration_s, alternatives = _infer_channels(
            total_samples,
            args.fs,
            args.duration_hint,
            allow_gui=not args.no_channel_gui,
            raw_path=raw_path,
        )
        msg = (
            f"Selected {n_channels} channels (approx duration {duration_s:.3f} s). "
            "Override with --n-channels if this looks wrong."
        )
        if alternatives:
            alts = ", ".join(f"{n} ch ~ {d:.2f} s" for n, d in alternatives)
            msg += f"\nOther plausible combinations: {alts}."
            if args.duration_hint is None and args.no_channel_gui:
                msg += "\nProvide --duration-hint or omit --no-channel-gui to pick manually."
        print(msg)
    else:
        n_channels = args.n_channels
        duration_s = total_samples / (args.fs * n_channels)
        print(f"Using explicit {n_channels} channels (approx duration {duration_s:.3f} s).")

    data = _reshape(raw, n_channels).astype(np.float32)

    start_idx = int(args.offset * args.fs)
    if start_idx < 0:
        raise ValueError("offset must be >= 0")
    if start_idx >= data.shape[0]:
        raise ValueError("offset is beyond the data length")

    if args.seconds is None:
        stop_idx = data.shape[0]
    else:
        n_samples = int(args.seconds * args.fs)
        if n_samples <= 0:
            raise ValueError("seconds must be positive")
        stop_idx = min(start_idx + n_samples, data.shape[0])

    seg = data[start_idx:stop_idx]
    time = np.arange(start_idx, stop_idx) / args.fs

    channels = _select_channels(args.channels, n_channels)

    plt.figure(figsize=(10, 2 * len(channels)))
    for i, ch in enumerate(channels, start=1):
        ax = plt.subplot(len(channels), 1, i)
        ax.plot(time, seg[:, ch], lw=0.8)
        ax.set_ylabel(f"Ch {ch} (uV)")
        if i == len(channels):
            ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
