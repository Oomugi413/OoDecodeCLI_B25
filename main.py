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
        self.last_directory = (
            Path(initial_paths[0]).expanduser().parent
            if initial_paths
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

        for path in initial_paths:
            self.add_path(Path(path))
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


def _gui_main(
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
