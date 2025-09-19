# -*- coding: utf-8 -*-
"""
GUI launcher to generate synthetic extracellular RAW files without using a console.

Usage:
- Double-click this file (launch_synthetic_raw.pyw) on Windows.
- Adjust parameters and click "Generate".

Output format:
- int16 little-endian, interleaved by channel, no header (.raw)
"""

from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional


# Ensure project root is importable when this file is double-clicked from the script directory
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from リポジトリ.generate_synthetic_raw import (
        SynthConfig,
        generate_synthetic_raw,
        save_raw_int16_le,
    )
except Exception as exc:  # pragma: no cover
    tk.Tk().withdraw()
    messagebox.showerror(
        title="Import Error",
        message=(
            "Could not import required module 'リポジトリ.generate_synthetic_raw'.\n\n"
            f"Project root tried: {PROJECT_ROOT}\n\n"
            f"Error: {exc}"
        ),
    )
    raise


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Synthetic RAW Generator")
        self.geometry("560x460")
        self.minsize(520, 420)

        # Variables
        self.var_fs = tk.StringVar(value=str(SynthConfig.fs))
        self.var_duration = tk.StringVar(value=str(SynthConfig.duration_s))
        self.var_channels = tk.StringVar(value=str(SynthConfig.n_channels))
        self.var_noise = tk.StringVar(value=str(SynthConfig.noise_std_uV))
        self.var_rate = tk.StringVar(value=str(SynthConfig.spike_rate_hz))
        self.var_units = tk.StringVar(value=str(SynthConfig.n_units))
        self.var_amp = tk.StringVar(value=str(SynthConfig.spike_amp_uV))
        self.var_refractory = tk.StringVar(value=str(SynthConfig.refractory_ms))
        self.var_seed = tk.StringVar(value="13")
        self.var_out = tk.StringVar(value=str((PROJECT_ROOT / "synthetic.raw").resolve()))
        self.var_open_folder = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Grid for parameters
        grid = ttk.LabelFrame(frm, text="Parameters")
        grid.pack(fill=tk.X, expand=False, **pad)

        def add_row(r: int, label: str, var: tk.StringVar, suffix: str = "") -> None:
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky=tk.W, **pad)
            entry = ttk.Entry(grid, textvariable=var, width=18)
            entry.grid(row=r, column=1, sticky=tk.W, **pad)
            ttk.Label(grid, text=suffix).grid(row=r, column=2, sticky=tk.W, **pad)

        add_row(0, "Sampling rate", self.var_fs, "Hz")
        add_row(1, "Duration", self.var_duration, "s")
        add_row(2, "Channels", self.var_channels)
        add_row(3, "Noise std", self.var_noise, "μV")
        add_row(4, "Spike rate", self.var_rate, "Hz / unit")
        add_row(5, "Units", self.var_units)
        add_row(6, "Spike amplitude", self.var_amp, "μV")
        add_row(7, "Refractory", self.var_refractory, "ms")
        add_row(8, "Seed (blank = None)", self.var_seed)

        # Output path chooser
        out_frame = ttk.LabelFrame(frm, text="Output")
        out_frame.pack(fill=tk.X, expand=False, **pad)

        ttk.Label(out_frame, text="RAW file path").grid(row=0, column=0, sticky=tk.W, **pad)
        ent_out = ttk.Entry(out_frame, textvariable=self.var_out, width=48)
        ent_out.grid(row=0, column=1, sticky=tk.EW, **pad)
        out_frame.columnconfigure(1, weight=1)
        ttk.Button(out_frame, text="Browse...", command=self._browse).grid(row=0, column=2, **pad)

        ttk.Checkbutton(out_frame, text="Open folder after save", variable=self.var_open_folder).grid(
            row=1, column=1, sticky=tk.W, **pad
        )

        # Actions
        btn_frame = ttk.Frame(frm)
        btn_frame.pack(fill=tk.X, expand=False, **pad)

        self.btn_generate = ttk.Button(btn_frame, text="Generate", command=self._on_generate)
        self.btn_generate.pack(side=tk.LEFT, padx=4)

        self.progress = ttk.Label(btn_frame, text="")
        self.progress.pack(side=tk.LEFT, padx=12)

        # Notes
        note = ttk.Label(
            frm,
            text=(
                "Output format: int16 little-endian, interleaved by channel, no header\n"
                "Size (bytes) ≈ fs × duration × channels × 2"
            ),
            justify=tk.LEFT,
        )
        note.pack(fill=tk.X, expand=False, **pad)

    def _browse(self) -> None:
        initial = Path(self.var_out.get()).expanduser()
        if not initial.parent.exists():
            initial = PROJECT_ROOT / "synthetic.raw"
        path = filedialog.asksaveasfilename(
            title="Save RAW file",
            defaultextension=".raw",
            filetypes=[("RAW files", "*.raw"), ("All files", "*.*")],
            initialfile=initial.name,
            initialdir=str(initial.parent),
        )
        if path:
            self.var_out.set(path)

    def _on_generate(self) -> None:
        try:
            cfg = self._read_config_from_ui()
        except ValueError as e:
            messagebox.showerror("Invalid parameter", str(e))
            return

        out_path = Path(self.var_out.get()).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.btn_generate.config(state=tk.DISABLED)
        self.progress.config(text="Generating...")
        self.update_idletasks()

        try:
            data_uv = generate_synthetic_raw(cfg)
            save_raw_int16_le(out_path, data_uv)

        except Exception as e:  # pragma: no cover
            messagebox.showerror("Generation failed", str(e))
            self.progress.config(text="")
            self.btn_generate.config(state=tk.NORMAL)
            return

        self.progress.config(text="Done")
        self.btn_generate.config(state=tk.NORMAL)

        size_bytes = data_uv.shape[0] * data_uv.shape[1] * 2
        msg = (
            f"Saved: {out_path}\n"
            f"Shape: {tuple(data_uv.shape)} (n_samples, n_channels)\n"
            f"Approx size: {size_bytes:,} bytes"
        )
        messagebox.showinfo("Success", msg)

        if self.var_open_folder.get():
            try:
                # Open the folder containing the file (Windows)
                import os

                os.startfile(out_path.parent)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _read_config_from_ui(self) -> SynthConfig:
        try:
            fs = int(self.var_fs.get())
            duration_s = float(self.var_duration.get())
            n_channels = int(self.var_channels.get())
            noise_std_uV = float(self.var_noise.get())
            spike_rate_hz = float(self.var_rate.get())
            n_units = int(self.var_units.get())
            spike_amp_uV = float(self.var_amp.get())
            refractory_ms = float(self.var_refractory.get())
        except Exception as e:
            raise ValueError(f"Failed to parse parameters: {e}")

        seed_str = self.var_seed.get().strip()
        seed: Optional[int]
        if seed_str == "":
            seed = None
        else:
            try:
                seed = int(seed_str)
            except ValueError:
                raise ValueError("Seed must be an integer or left blank")

        if fs <= 0:
            raise ValueError("Sampling rate must be positive")
        if duration_s <= 0:
            raise ValueError("Duration must be positive")
        if n_channels <= 0:
            raise ValueError("Channels must be positive")
        if n_units < 0:
            raise ValueError("Units must be >= 0")
        if spike_rate_hz < 0:
            raise ValueError("Spike rate must be >= 0")
        if refractory_ms < 0:
            raise ValueError("Refractory must be >= 0")

        return SynthConfig(
            fs=fs,
            duration_s=duration_s,
            n_channels=n_channels,
            noise_std_uV=noise_std_uV,
            spike_rate_hz=spike_rate_hz,
            n_units=n_units,
            spike_amp_uV=spike_amp_uV,
            refractory_ms=refractory_ms,
            seed=seed,
        )


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()


