from __future__ import annotations

import queue
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from spike_sort_pipeline import DEFAULT_CHANNELS, run_spike_sorting


class LauncherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spike Sort Launcher")
        self.geometry("960x620")
        self.minsize(860, 560)

        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.raw_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Idle")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._process_queue)

    def _build_ui(self) -> None:
        for col in range(2):
            self.columnconfigure(col, weight=1)
        self.rowconfigure(1, weight=1)

        file_frame = ttk.LabelFrame(self, text="入力")
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="RAW ファイル:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        raw_entry = ttk.Entry(file_frame, textvariable=self.raw_path_var)
        raw_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.raw_button = ttk.Button(file_frame, text="参照...", command=self._browse_raw)
        self.raw_button.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(file_frame, text="出力フォルダ:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        out_entry = ttk.Entry(file_frame, textvariable=self.output_dir_var)
        out_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.output_button = ttk.Button(file_frame, text="参照...", command=self._browse_output)
        self.output_button.grid(row=1, column=2, padx=5, pady=5)

        channel_frame = ttk.LabelFrame(self, text="チャンネル選択")
        channel_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        channel_frame.columnconfigure(0, weight=1)
        channel_frame.rowconfigure(1, weight=1)

        info = "Ctrl/Shift クリックで複数選択。未選択の場合は既定の全チャンネルを処理します。"
        ttk.Label(channel_frame, text=info).grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.channel_listbox = tk.Listbox(channel_frame, selectmode=tk.MULTIPLE, exportselection=False)
        scrollbar = ttk.Scrollbar(channel_frame, orient=tk.VERTICAL, command=self.channel_listbox.yview)
        self.channel_listbox.configure(yscrollcommand=scrollbar.set)
        self.channel_listbox.grid(row=1, column=0, sticky="nsew", padx=(5, 0), pady=5)
        scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 5), pady=5)

        for ch in DEFAULT_CHANNELS:
            self.channel_listbox.insert(tk.END, ch)
        if DEFAULT_CHANNELS:
            self.channel_listbox.selection_set(0, tk.END)

        log_frame = ttk.LabelFrame(self, text="ログ")
        log_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(log_frame, state="disabled", wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        button_frame = ttk.Frame(self)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))
        button_frame.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(button_frame, text="スパイクソーティング開始", command=self._run_clicked)
        self.run_button.grid(row=0, column=0, padx=5)

        ttk.Label(button_frame, textvariable=self.status_var).grid(row=0, column=1, sticky="w")

        ttk.Button(button_frame, text="閉じる", command=self._on_close).grid(row=0, column=2, padx=5)

    def _browse_raw(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("RAW", "*.raw"), ("All files", "*.*")])
        if path:
            self.raw_path_var.set(path)
            default_output = Path(path).parent / "temp_results"
            if not self.output_dir_var.get():
                self.output_dir_var.set(str(default_output))

    def _browse_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir_var.set(path)

    def _get_selected_channels(self) -> Sequence[str]:
        selection = self.channel_listbox.curselection()
        if not selection:
            return list(DEFAULT_CHANNELS)
        return [self.channel_listbox.get(i) for i in selection]

    def _run_clicked(self) -> None:
        raw_path_str = self.raw_path_var.get().strip()
        if not raw_path_str:
            messagebox.showwarning("入力不足", "RAW ファイルを選択してください。")
            return
        raw_path = Path(raw_path_str)
        if not raw_path.exists():
            messagebox.showerror("ファイルが見つかりません", f"{raw_path} は存在しません。")
            return

        output_str = self.output_dir_var.get().strip()
        output_root = Path(output_str) if output_str else raw_path.parent / "temp_results"
        output_root.mkdir(parents=True, exist_ok=True)

        channels = self._get_selected_channels()
        self.append_log(f"RAW: {raw_path}")
        self.append_log(f"出力先: {output_root}")
        self.append_log(f"対象チャンネル数: {len(channels)}")

        self._set_running_state(True)
        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(raw_path, output_root, channels),
            daemon=True,
        )
        thread.start()

    def _run_pipeline_thread(self, raw_path: Path, output_root: Path, channels: Sequence[str]) -> None:
        try:
            result = run_spike_sorting(
                raw_path=raw_path,
                channels=channels,
                output_root=output_root,
                progress_callback=lambda msg: self.queue.put(("info", msg)),
            )
            self.queue.put(("done", result))
        except Exception as exc:  # noqa: BLE001
            detail = traceback.format_exc()
            self.queue.put(("error", (str(exc), detail)))

    def _process_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "info":
                    self.append_log(str(payload))
                elif kind == "done":
                    self._handle_done(payload)
                elif kind == "error":
                    self._handle_error(payload)  # type: ignore[arg-type]
        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_queue)

    def _handle_done(self, result: dict[str, object]) -> None:
        self._set_running_state(False)
        base_dir: Path = result["base_dir"]  # type: ignore[assignment]
        log_path: Path = result["log_path"]  # type: ignore[assignment]
        summary = result.get("summary", [])
        self.append_log("ソーティングが完了しました。")
        if isinstance(summary, list) and summary:
            for entry in summary:
                channel = entry.get("channel")
                spikes = entry.get("total_spikes")
                clusters = entry.get("positive_clusters")
                self.append_log(f"  {channel}: spikes={spikes}, clusters={clusters}")
        self.append_log(f"成果物: {base_dir}")
        self.append_log(f"ログ: {log_path}")
        messagebox.showinfo(
            "完了",
            f"処理が完了しました。\n出力先: {base_dir}\nログ: {log_path}",
        )

    def _handle_error(self, payload: tuple[str, str]) -> None:
        self._set_running_state(False)
        message, detail = payload
        self.append_log(f"エラー: {message}")
        messagebox.showerror("エラー", f"{message}\n\n{detail}")

    def _set_running_state(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.run_button.configure(state=state)
        self.raw_button.configure(state=state)
        self.output_button.configure(state=state)
        self.channel_listbox.configure(state=state)
        self.status_var.set("Running" if running else "Idle")

    def append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        if self.run_button["state"] == tk.DISABLED:
            if not messagebox.askyesno("確認", "処理中です。終了しますか？"):
                return
        self.destroy()


def main() -> None:
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
