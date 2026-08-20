#!/usr/bin/env python3
"""Small, standalone ARIB STD-B25 batch decoder.

The program deliberately knows nothing about EDCB or Mirakurun.  It only
renames an input recording, runs arib-b25-stream-test, and checks the output
duration with ffprobe.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from urllib.parse import unquote, urlparse
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


DEFAULT_DECODER = os.environ.get("B25_DECODER", "arib-b25-stream-test")
DEFAULT_FFPROBE = os.environ.get("FFPROBE", "ffprobe")
DURATION_TOLERANCE_SECONDS = 30.0


class ProcessingError(RuntimeError):
    """An expected, user-facing processing error."""


@dataclass(frozen=True)
class StagedPaths:
    """The original, staging, and final names for one input recording."""

    input_path: Path
    staging_path: Path
    output_path: Path
    already_staged: bool


@dataclass(frozen=True)
class JobResult:
    """Result returned for one file."""

    input_path: Path
    status: str
    message: str
    staging_path: Path | None = None
    output_path: Path | None = None
    temporary_path: Path | None = None


def _remove_final_ts(path: Path) -> Path:
    """Remove exactly one final ``.ts`` suffix while preserving case."""

    return Path(str(path)[:-3])


def plan_paths(path: Path) -> StagedPaths:
    """Validate a recording extension and calculate safe working names.

    A regular ``foo.ts``/``foo.m2ts`` input is renamed to ``*.ts`` and the
    decoded output is written back to the original name.  A name ending in
    ``.ts.ts`` or ``.m2ts.ts`` is accepted as a previously staged input so a
    failed run can be resumed without appending another suffix.
    """

    name = path.name
    lower_name = name.lower()

    if lower_name.endswith(".ts.ts.ts") or lower_name.endswith(".m2ts.ts.ts"):
        raise ProcessingError(
            "既に .ts が複数回付いています。対象を確認してください: "
            f"{path}"
        )

    if lower_name.endswith(".m2ts.ts") or lower_name.endswith(".ts.ts"):
        return StagedPaths(
            input_path=path,
            staging_path=path,
            output_path=_remove_final_ts(path),
            already_staged=True,
        )

    if lower_name.endswith(".m2ts") or lower_name.endswith(".ts"):
        return StagedPaths(
            input_path=path,
            staging_path=Path(str(path) + ".ts"),
            output_path=path,
            already_staged=False,
        )

    raise ProcessingError(
        f"拡張子が対応外です（.ts または .m2ts が必要）: {path}"
    )


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f} 秒"


def probe_duration(path: Path, ffprobe: str = DEFAULT_FFPROBE) -> float:
    """Read a media duration in seconds using ffprobe."""

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ProcessingError(f"ffprobeを起動できません: {exc}") from exc

    if completed.returncode != 0:
        detail = _stderr_tail(completed.stderr)
        suffix = f" ({detail})" if detail else ""
        raise ProcessingError(
            f"ffprobeで長さを取得できません: {path}{suffix}"
        )

    value_text = completed.stdout.strip()
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ProcessingError(
            f"ffprobeの長さが数値ではありません: {path}: {value_text!r}"
        ) from exc

    if not math.isfinite(value) or value <= 0:
        raise ProcessingError(f"ffprobeの長さが不正です: {path}: {value_text!r}")
    return value


def _stderr_tail(stderr: str | bytes | None, limit: int = 8) -> str:
    """Make decoder/ffprobe diagnostics compact and readable."""

    if not stderr:
        return ""
    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    else:
        text = stderr
    # arib-b25-stream-test commonly updates one progress line with CR.
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    return " / ".join(lines[-limit:])


def _ensure_regular_file(path: Path) -> None:
    if not path.exists():
        raise ProcessingError(f"入力ファイルが見つかりません: {path}")
    if not path.is_file():
        raise ProcessingError(f"入力が通常ファイルではありません: {path}")


def _entry_exists(path: Path) -> bool:
    """Return true for regular files, directories, and dangling symlinks."""

    return os.path.lexists(str(path))


def _make_temporary_path(output_path: Path) -> Path:
    """Reserve a same-directory temporary path without creating output."""

    try:
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.decoding-",
            suffix=".part",
            dir=str(output_path.parent),
            delete=False,
        )
    except OSError as exc:
        raise ProcessingError(
            f"一時ファイルを作成できません: {output_path.parent}: {exc}"
        ) from exc
    temporary_path = Path(handle.name)
    handle.close()
    return temporary_path


def decode_one(
    input_path: Path,
    *,
    decoder: str = DEFAULT_DECODER,
    ffprobe: str = DEFAULT_FFPROBE,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> JobResult:
    """Process one recording, retaining every user-visible file on failure."""

    def notify(message: str) -> None:
        if progress is not None:
            progress(message)

    input_path = Path(input_path).expanduser()
    try:
        # Extension validation intentionally comes before any filesystem
        # mutation or decoder launch.
        planned = plan_paths(input_path)
        _ensure_regular_file(planned.input_path)

        # For a regular input, output_path is the input's current name, so it
        # necessarily exists.  It is a conflict only when processing a
        # previously staged name (where input_path and output_path differ).
        if (
            planned.output_path != planned.input_path
            and _entry_exists(planned.output_path)
        ):
            return JobResult(
                input_path=input_path,
                status="skipped",
                message=f"SKIP: 出力先が既に存在します: {planned.output_path}",
                staging_path=planned.staging_path,
                output_path=planned.output_path,
            )

        if not planned.already_staged and _entry_exists(planned.staging_path):
            return JobResult(
                input_path=input_path,
                status="skipped",
                message=(
                    "SKIP: リネーム先が既に存在するため変更しません: "
                    f"{planned.staging_path}"
                ),
                staging_path=planned.staging_path,
                output_path=planned.output_path,
            )

        if dry_run:
            return JobResult(
                input_path=input_path,
                status="dry-run",
                message=(
                    f"DRY-RUN: {planned.input_path} -> {planned.staging_path} "
                    f"-> {planned.output_path}"
                ),
                staging_path=planned.staging_path,
                output_path=planned.output_path,
            )

        # Probe before renaming.  This catches a missing/broken ffprobe
        # installation without leaving the input in its staged name.
        old_duration = probe_duration(planned.input_path, ffprobe)
        notify(f"長さを確認: {planned.input_path} ({_format_seconds(old_duration)})")

        if not planned.already_staged:
            try:
                planned.input_path.rename(planned.staging_path)
            except OSError as exc:
                raise ProcessingError(
                    f"入力ファイルのリネームに失敗しました: "
                    f"{planned.input_path} -> {planned.staging_path}: {exc}"
                ) from exc
            notify(f"リネーム完了: {planned.input_path} -> {planned.staging_path}")

        _ensure_regular_file(planned.staging_path)
        temporary_path = _make_temporary_path(planned.output_path)
        notify(f"復号開始: {planned.staging_path}")

        try:
            with planned.staging_path.open("rb") as source, temporary_path.open(
                "wb"
            ) as destination:
                completed = subprocess.run(
                    [decoder],
                    stdin=source,
                    stdout=destination,
                    stderr=subprocess.PIPE,
                    check=False,
                )
        except OSError as exc:
            raise ProcessingError(
                f"arib-b25-stream-testを起動できません: {exc}"
            ) from exc

        diagnostics = _stderr_tail(completed.stderr)
        if completed.returncode != 0:
            detail = f" ({diagnostics})" if diagnostics else ""
            raise ProcessingError(
                f"復号に失敗しました（終了コード {completed.returncode}）: "
                f"{planned.staging_path}{detail}"
            )

        try:
            temporary_size = temporary_path.stat().st_size
        except OSError as exc:
            raise ProcessingError(
                f"復号出力のサイズを確認できません: {temporary_path}: {exc}"
            ) from exc
        if temporary_size <= 0:
            raise ProcessingError(
                f"復号出力が空です。途中ファイルを残しました: {temporary_path}"
            )
        # Do not overwrite an existing user file.  The normal conflict check
        # above covers ordinary use; this second check protects a race.
        if _entry_exists(planned.output_path):
            raise ProcessingError(
                f"復号中に出力先が作成されたため公開しません。"
                f"途中ファイルを残しました: {temporary_path}"
            )
        try:
            temporary_path.rename(planned.output_path)
        except OSError as exc:
            raise ProcessingError(
                f"復号出力を完成ファイル名へ移動できません: "
                f"{temporary_path} -> {planned.output_path}: {exc}"
            ) from exc
        notify(f"復号出力を作成: {planned.output_path}")

        try:
            new_duration = probe_duration(planned.output_path, ffprobe)
        except ProcessingError as exc:
            raise ProcessingError(
                f"出力は削除せず残しています。{exc}"
            ) from exc
        difference = abs(new_duration - old_duration)
        if difference > DURATION_TOLERANCE_SECONDS:
            raise ProcessingError(
                "長さの差が許容範囲を超えました。ファイルは削除せず残しています: "
                f"旧 {_format_seconds(old_duration)}, "
                f"新 {_format_seconds(new_duration)}, "
                f"差 {_format_seconds(difference)}"
            )

        return JobResult(
            input_path=input_path,
            status="success",
            message=(
                f"OK: {planned.output_path} "
                f"（旧 {_format_seconds(old_duration)} / "
                f"新 {_format_seconds(new_duration)} / "
                f"差 {_format_seconds(difference)}）"
            ),
            staging_path=planned.staging_path,
            output_path=planned.output_path,
        )
    except ProcessingError as exc:
        return JobResult(
            input_path=input_path,
            status="error",
            message=f"ERROR: {exc}",
            staging_path=locals().get("planned", None).staging_path
            if "planned" in locals()
            else None,
            output_path=locals().get("planned", None).output_path
            if "planned" in locals()
            else None,
            temporary_path=locals().get("temporary_path"),
        )


def process_many(
    paths: Iterable[Path],
    *,
    decoder: str = DEFAULT_DECODER,
    ffprobe: str = DEFAULT_FFPROBE,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[JobResult]:
    """Process files sequentially, preserving the order supplied by caller."""

    results: list[JobResult] = []
    for path in paths:
        result = decode_one(
            Path(path),
            decoder=decoder,
            ffprobe=ffprobe,
            dry_run=dry_run,
            progress=progress,
        )
        results.append(result)
        if progress is not None:
            progress(result.message)
    return results


def _cli_main(
    paths: Sequence[str],
    *,
    decoder: str,
    ffprobe: str,
    dry_run: bool,
) -> int:
    if not paths:
        print("入力ファイルがありません。--help を確認してください。", file=sys.stderr)
        return 2

    results = process_many(
        [Path(path) for path in paths],
        decoder=decoder,
        ffprobe=ffprobe,
        dry_run=dry_run,
        progress=lambda message: print(message, flush=True),
    )
    successes = sum(result.status == "success" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    dry_runs = sum(result.status == "dry-run" for result in results)
    failures = sum(result.status == "error" for result in results)
    print(
        f"完了: {len(results)} 件（成功 {successes}, スキップ {skipped}, "
        f"確認のみ {dry_runs}, エラー {failures}）",
        flush=True,
    )
    return 1 if failures else 0


class DecoderApp:
    """Tkinter front end; files supplied by a desktop launcher are queued."""

    def __init__(
        self,
        root,
        initial_paths: Sequence[str],
        *,
        decoder: str,
        ffprobe: str,
    ) -> None:
        import tkinter as tk
        from tkinter import filedialog
        from tkinter import ttk

        self.tk = tk
        self.filedialog = filedialog
        self.ttk = ttk
        self.root = root
        self.decoder = decoder
        self.ffprobe = ffprobe
        self.zenity = shutil.which("zenity")
        self.paths: list[Path] = []
        first_path = (
            _drop_value_to_path(initial_paths[0]) if initial_paths else None
        )
        self.last_directory = (
            first_path.parent
            if first_path is not None
            else Path.cwd()
        )
        self.running = False
        self.messages: queue.Queue[object] = queue.Queue()

        root.title("OoDecodeCLI B25")
        root.geometry("760x500")
        root.minsize(620, 360)

        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text=(
                "TS / M2TSを追加して「復号開始」を押してください。\n"
                "入力は .ts.ts にリネームされ、元の名前に復号結果を作成します。"
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.listbox.yview
        )
        scrollbar.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=scrollbar.set)

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=(8, 4))
        self.add_button = ttk.Button(
            button_frame, text="ファイル追加…", command=self.add_files
        )
        self.add_button.pack(side="left")
        self.remove_button = ttk.Button(
            button_frame, text="選択削除", command=self.remove_selected
        )
        self.remove_button.pack(side="left", padx=(6, 0))
        self.clear_button = ttk.Button(
            button_frame, text="一覧をクリア", command=self.clear_files
        )
        self.clear_button.pack(side="left", padx=(6, 0))
        self.start_button = ttk.Button(
            button_frame, text="復号開始", command=self.start_processing
        )
        self.start_button.pack(side="right")
        self.close_button = ttk.Button(
            button_frame, text="終了", command=root.destroy
        )
        self.close_button.pack(side="right", padx=(0, 6))

        ttk.Label(frame, text="ログ").pack(anchor="w", pady=(8, 2))
        self.log_text = tk.Text(frame, height=8, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=False)

        for value in initial_paths:
            path = _drop_value_to_path(value)
            if path is not None:
                self.add_path(path)
        if initial_paths:
            self.write_log(f"起動時に {len(initial_paths)} 件を追加しました。")
        self._poll_messages()

    def add_path(self, path: Path) -> None:
        path = path.expanduser()
        if path in self.paths:
            return
        self.paths.append(path)
        self.listbox.insert(self.tk.END, str(path))

    def add_files(self) -> None:
        paths = self._select_files_with_zenity()
        if paths is None:
            paths = self.filedialog.askopenfilenames(
                title="復号するTS/M2TSを選択",
                initialdir=str(self.last_directory),
                filetypes=[
                    ("TS / M2TS", "*.ts *.TS *.m2ts *.M2TS"),
                    ("すべてのファイル", "*"),
                ],
            )
        for path in paths:
            self.add_path(Path(path))
        if paths:
            self.last_directory = Path(paths[0]).expanduser().parent

    def _select_files_with_zenity(self) -> tuple[str, ...] | None:
        """Use GNOME's native GTK picker, avoiding Tk's X11 menu bug.

        ``None`` means Zenity could not be used and the caller should fall
        back to Tk's dialog.  An empty tuple means the user cancelled.
        """

        if self.zenity is None:
            return None

        initial_name = str(self.last_directory) + os.sep
        command = [
            self.zenity,
            "--file-selection",
            "--multiple",
            "--title=復号するTS/M2TSを選択",
            "--separator=\n",
            f"--filename={initial_name}",
            "--file-filter=TS / M2TS | *.ts *.TS *.m2ts *.M2TS",
            "--file-filter=すべてのファイル | *",
        ]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            self.write_log(
                f"Zenityを起動できないためTk選択画面を使用します: {exc}"
            )
            self.zenity = None
            return None

        if completed.returncode == 0:
            return tuple(path for path in completed.stdout.splitlines() if path)
        if completed.returncode == 1:
            return ()

        detail = _stderr_tail(completed.stderr)
        self.write_log(
            "Zenityのファイル選択に失敗したためTk選択画面を使用します"
            + (f": {detail}" if detail else "。")
        )
        return None

    def remove_selected(self) -> None:
        selected = list(self.listbox.curselection())
        for index in reversed(selected):
            self.listbox.delete(index)
            del self.paths[index]

    def clear_files(self) -> None:
        self.listbox.delete(0, self.tk.END)
        self.paths.clear()

    def write_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(self.tk.END, message + "\n")
        self.log_text.see(self.tk.END)
        self.log_text.configure(state="disabled")

    def _poll_messages(self) -> None:
        try:
            while True:
                message = self.messages.get_nowait()
                if (
                    isinstance(message, tuple)
                    and len(message) == 2
                    and message[0] == "finished"
                ):
                    self._processing_finished(message[1])
                else:
                    self.write_log(str(message))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _set_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in (
            self.add_button,
            self.remove_button,
            self.clear_button,
            self.start_button,
            self.close_button,
        ):
            button.configure(state=state)

    def start_processing(self) -> None:
        if self.running:
            return
        if not self.paths:
            from tkinter import messagebox

            messagebox.showinfo("OoDecodeCLI B25", "ファイルを追加してください。")
            return

        paths = list(self.paths)
        self.running = True
        self._set_controls(False)
        self.write_log(f"{len(paths)} 件を順番に処理します。")

        def worker() -> None:
            try:
                results = process_many(
                    paths,
                    decoder=self.decoder,
                    ffprobe=self.ffprobe,
                    progress=self.messages.put,
                )
                self.messages.put(("finished", results))
            except Exception as exc:  # keep the GUI usable on an unexpected error
                self.messages.put(("finished", [
                    JobResult(
                        input_path=path,
                        status="error",
                        message=f"ERROR: 予期しない例外: {exc}",
                    )
                    for path in paths
                ]))

        threading.Thread(target=worker, daemon=True).start()

    def _processing_finished(self, results: Sequence[JobResult]) -> None:
        self.running = False
        self._set_controls(True)
        failures = sum(result.status == "error" for result in results)
        self.write_log(
            f"処理完了: {len(results)} 件（エラー {failures} 件）。"
        )
        from tkinter import messagebox

        if failures:
            messagebox.showerror(
                "OoDecodeCLI B25",
                f"{failures} 件でエラーが発生しました。ログとファイルを確認してください。",
            )
        else:
            messagebox.showinfo(
                "OoDecodeCLI B25",
                "処理が完了しました。入力ファイルと出力ファイルは削除していません。",
            )


def _drop_value_to_path(value: str) -> Path | None:
    """Convert a file-manager DnD URI or plain path into a local path."""

    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.startswith("#"):
        return None
    parsed = urlparse(value)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            return None
        return Path(unquote(parsed.path))
    if parsed.scheme:
        return None
    return Path(value)


class GtkDecoderApp:
    """GTK3 front end with native file-manager drag-and-drop support."""

    def __init__(self, Gtk, Gdk, GLib, initial_paths, *, decoder, ffprobe):
        self.Gtk = Gtk
        self.Gdk = Gdk
        self.GLib = GLib
        self.decoder = decoder
        self.ffprobe = ffprobe
        self.paths: list[Path] = []
        self.running = False
        self.messages: queue.Queue[object] = queue.Queue()
        first_path = (
            _drop_value_to_path(initial_paths[0]) if initial_paths else None
        )
        self.last_directory = (
            first_path.parent
            if first_path is not None
            else Path.cwd()
        )

        self.window = Gtk.Window(title="OoDecodeCLI B25")
        self.window.set_default_size(760, 500)
        self.window.set_border_width(12)
        self.window.connect("destroy", Gtk.main_quit)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.drop_area = Gtk.EventBox()
        self.window.add(self.drop_area)
        self.drop_area.add(outer)
        self.drop_label = Gtk.Label(
            label=(
                "TS / M2TSをここへドラッグ＆ドロップするか、"
                "「ファイル追加…」を押してください。\n"
                "入力は .ts.ts にリネームされ、元の名前に復号結果を作成します。"
            )
        )
        self.drop_label.set_xalign(0)
        self.drop_label.set_line_wrap(True)
        outer.pack_start(self.drop_label, False, False, 0)

        self.store = Gtk.ListStore(str)
        self.tree = Gtk.TreeView(model=self.store)
        self.tree.set_headers_visible(False)
        selection = self.tree.get_selection()
        selection.set_mode(Gtk.SelectionMode.MULTIPLE)
        renderer = Gtk.CellRendererText()
        column = Gtk.TreeViewColumn("対象ファイル", renderer, text=0)
        self.tree.append_column(column)
        self.tree_scroll = Gtk.ScrolledWindow()
        self.tree_scroll.set_policy(
            Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC
        )
        self.tree_scroll.add(self.tree)
        outer.pack_start(self.tree_scroll, True, True, 0)

        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        outer.pack_start(buttons, False, False, 0)
        self.add_button = Gtk.Button(label="ファイル追加…")
        self.add_button.connect("clicked", self.add_files)
        buttons.pack_start(self.add_button, False, False, 0)
        self.remove_button = Gtk.Button(label="選択削除")
        self.remove_button.connect("clicked", self.remove_selected)
        buttons.pack_start(self.remove_button, False, False, 0)
        self.clear_button = Gtk.Button(label="一覧をクリア")
        self.clear_button.connect("clicked", self.clear_files)
        buttons.pack_start(self.clear_button, False, False, 0)
        self.close_button = Gtk.Button(label="終了")
        self.close_button.connect("clicked", lambda *_: self.window.destroy())
        buttons.pack_end(self.close_button, False, False, 0)
        self.start_button = Gtk.Button(label="復号開始")
        self.start_button.connect("clicked", self.start_processing)
        buttons.pack_end(self.start_button, False, False, 0)

        self.log_buffer = Gtk.TextBuffer()
        self.log_view = Gtk.TextView(buffer=self.log_buffer)
        self.log_view.set_editable(False)
        self.log_view.set_cursor_visible(False)
        self.log_view.set_wrap_mode(Gtk.WrapMode.WORD)
        log_scroll = Gtk.ScrolledWindow()
        log_scroll.set_min_content_height(110)
        log_scroll.add(self.log_view)
        outer.pack_start(log_scroll, False, True, 0)

        # Gtk's XDND handling accepts the standard URI list emitted by GNOME
        # Files/Nautilus, as well as plain text paths from other file managers.
        for widget in (
            self.drop_area,
            self.drop_label,
            self.tree,
            self.tree_scroll,
            self.log_view,
            self.add_button,
            self.remove_button,
            self.clear_button,
            self.close_button,
            self.start_button,
        ):
            self._enable_drop(widget)

        for value in initial_paths:
            path = _drop_value_to_path(value)
            if path is not None:
                self.add_path(path)
        if initial_paths:
            self.write_log(f"起動時に {len(initial_paths)} 件を追加しました。")
        GLib.timeout_add(100, self._poll_messages)

    def _enable_drop(self, widget) -> None:
        targets = [
            self.Gtk.TargetEntry.new("text/uri-list", 0, 0),
            self.Gtk.TargetEntry.new("text/plain", 0, 1),
            self.Gtk.TargetEntry.new("x-special/gnome-copied-files", 0, 2),
        ]
        widget.drag_dest_set(
            self.Gtk.DestDefaults.ALL,
            targets,
            self.Gdk.DragAction.COPY,
        )
        widget.connect("drag-data-received", self._on_drag_data_received)

    def _on_drag_data_received(
        self, widget, context, x, y, selection_data, info, timestamp
    ) -> None:
        if self.running:
            context.finish(False, False, timestamp)
            return
        values = selection_data.get_uris() or []
        if values and values[0] in {"copy", "cut"}:
            values = values[1:]
        if not values:
            text = selection_data.get_text() or ""
            values = text.splitlines()
            if values and values[0] in {"copy", "cut"}:
                values = values[1:]
        added = 0
        for value in values:
            path = _drop_value_to_path(value)
            if path is None:
                continue
            before = len(self.paths)
            self.add_path(path)
            added += int(len(self.paths) > before)
        context.finish(added > 0, False, timestamp)
        if added:
            self.write_log(f"ドラッグ＆ドロップで {added} 件を追加しました。")
        else:
            self.write_log("ドロップされたデータからファイルを認識できませんでした。")

    def add_path(self, path: Path) -> None:
        path = path.expanduser()
        if path in self.paths:
            return
        self.paths.append(path)
        self.store.append([str(path)])

    def add_files(self, *_args) -> None:
        dialog = self.Gtk.FileChooserDialog(
            title="復号するTS/M2TSを選択",
            parent=self.window,
            action=self.Gtk.FileChooserAction.OPEN,
            buttons=(
                "キャンセル",
                self.Gtk.ResponseType.CANCEL,
                "追加",
                self.Gtk.ResponseType.ACCEPT,
            ),
        )
        dialog.set_select_multiple(True)
        if self.last_directory.is_dir():
            dialog.set_current_folder(str(self.last_directory))
        media_filter = self.Gtk.FileFilter()
        media_filter.set_name("TS / M2TS")
        for pattern in ("*.ts", "*.TS", "*.m2ts", "*.M2TS"):
            media_filter.add_pattern(pattern)
        dialog.add_filter(media_filter)
        all_filter = self.Gtk.FileFilter()
        all_filter.set_name("すべてのファイル")
        all_filter.add_pattern("*")
        dialog.add_filter(all_filter)
        response = dialog.run()
        paths = dialog.get_filenames() if response == self.Gtk.ResponseType.ACCEPT else []
        dialog.destroy()
        for path in paths:
            self.add_path(Path(path))
        if paths:
            self.last_directory = Path(paths[0]).expanduser().parent

    def remove_selected(self, *_args) -> None:
        selection = self.tree.get_selection()
        model, rows = selection.get_selected_rows()
        indices = sorted(
            (row.get_indices()[0] for row in rows), reverse=True
        )
        for index in indices:
            tree_iter = model.get_iter(self.Gtk.TreePath.new_from_indices([index]))
            model.remove(tree_iter)
            del self.paths[index]

    def clear_files(self, *_args) -> None:
        self.store.clear()
        self.paths.clear()

    def write_log(self, message: str) -> None:
        end = self.log_buffer.get_end_iter()
        self.log_buffer.insert(end, message + "\n")
        mark = self.log_buffer.create_mark(None, self.log_buffer.get_end_iter(), False)
        self.log_view.scroll_to_mark(mark, 0.0, True, 0.0, 1.0)
        self.log_buffer.delete_mark(mark)

    def _poll_messages(self) -> bool:
        try:
            while True:
                message = self.messages.get_nowait()
                if (
                    isinstance(message, tuple)
                    and len(message) == 2
                    and message[0] == "finished"
                ):
                    self._processing_finished(message[1])
                else:
                    self.write_log(str(message))
        except queue.Empty:
            pass
        return True

    def _set_controls(self, enabled: bool) -> None:
        for button in (
            self.add_button,
            self.remove_button,
            self.clear_button,
            self.start_button,
            self.close_button,
        ):
            button.set_sensitive(enabled)

    def start_processing(self, *_args) -> None:
        if self.running:
            return
        if not self.paths:
            self.write_log("ファイルを追加してください。")
            return
        paths = list(self.paths)
        self.running = True
        self._set_controls(False)
        self.write_log(f"{len(paths)} 件を順番に処理します。")

        def worker() -> None:
            try:
                results = process_many(
                    paths,
                    decoder=self.decoder,
                    ffprobe=self.ffprobe,
                    progress=self.messages.put,
                )
                self.messages.put(("finished", results))
            except Exception as exc:
                self.messages.put(
                    (
                        "finished",
                        [
                            JobResult(
                                input_path=path,
                                status="error",
                                message=f"ERROR: 予期しない例外: {exc}",
                            )
                            for path in paths
                        ],
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def _processing_finished(self, results: Sequence[JobResult]) -> None:
        self.running = False
        self._set_controls(True)
        failures = sum(result.status == "error" for result in results)
        self.write_log(f"処理完了: {len(results)} 件（エラー {failures} 件）。")
        message_type = (
            self.Gtk.MessageType.ERROR if failures else self.Gtk.MessageType.INFO
        )
        text = (
            f"{failures} 件でエラーが発生しました。ログとファイルを確認してください。"
            if failures
            else "処理が完了しました。入力ファイルと出力ファイルは削除していません。"
        )
        dialog = self.Gtk.MessageDialog(
            transient_for=self.window,
            modal=True,
            message_type=message_type,
            buttons=self.Gtk.ButtonsType.OK,
            text=text,
        )
        dialog.run()
        dialog.destroy()


def _gtk_gui_main(
    paths: Sequence[str], *, decoder: str, ffprobe: str
) -> int | None:
    """Run the GTK GUI, or return None when GTK3 is unavailable."""

    try:
        import gi

        gi.require_version("Gdk", "3.0")
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gdk, GLib, Gtk
    except (ImportError, ValueError):
        return None

    try:
        initialized = Gtk.init_check()
        if initialized is False or (
            isinstance(initialized, tuple)
            and initialized
            and not initialized[0]
        ):
            return None
        app = GtkDecoderApp(
            Gtk, Gdk, GLib, paths, decoder=decoder, ffprobe=ffprobe
        )
        app.window.show_all()
        Gtk.main()
    except Exception as exc:
        print(f"GTK GUIを起動できません: {exc}", file=sys.stderr)
        return None
    return 0


def _tk_gui_main(
    paths: Sequence[str], *, decoder: str, ffprobe: str
) -> int:
    try:
        import tkinter as tk
    except ImportError:
        print(
            "Tkinterが利用できません。python3-tkをインストールするか、"
            "--cli を使用してください。",
            file=sys.stderr,
        )
        return 2

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(
            f"GUIを起動できません（DISPLAY/Tkinterを確認してください）: {exc}",
            file=sys.stderr,
        )
        return 2
    DecoderApp(root, paths, decoder=decoder, ffprobe=ffprobe)
    root.mainloop()
    return 0


def _gui_main(
    paths: Sequence[str], *, decoder: str, ffprobe: str
) -> int:
    gtk_result = _gtk_gui_main(paths, decoder=decoder, ffprobe=ffprobe)
    if gtk_result is not None:
        return gtk_result
    return _tk_gui_main(paths, decoder=decoder, ffprobe=ffprobe)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立したARIB STD-B25復号ツール（EDCB/Mirakurun非依存）"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="入力する .ts / .m2ts（複数可。GUI起動時は一覧に追加）",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="GUIを使わず、処理結果を端末に表示",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="リネーム・復号をせず、拡張子と競合を確認",
    )
    parser.add_argument(
        "--decoder",
        default=DEFAULT_DECODER,
        help=f"復号コマンド（既定: {DEFAULT_DECODER}）",
    )
    parser.add_argument(
        "--ffprobe",
        default=DEFAULT_FFPROBE,
        help=f"ffprobeの実行ファイル（既定: {DEFAULT_FFPROBE}）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cli or args.dry_run:
        return _cli_main(
            args.files,
            decoder=args.decoder,
            ffprobe=args.ffprobe,
            dry_run=args.dry_run,
        )
    return _gui_main(args.files, decoder=args.decoder, ffprobe=args.ffprobe)


if __name__ == "__main__":
    raise SystemExit(main())
