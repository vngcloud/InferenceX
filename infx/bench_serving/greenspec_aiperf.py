"""GreenSpec on aiperf: subset selection, the raw_payload fallback, and the artifact guard.

GreenSpec is an internal speculative-decoding bench cut from production chat
traffic. A release ships every subset in one flat ``greenspec.jsonl``: each line
is a fixed set of metadata keys (``META_KEYS``) followed by a complete
chat-completions body with no ``model`` and no ``stream``. ``run_greenspec_aiperf``
(``benchmarks/benchmark_lib.sh``) drives aiperf over one subset and uses this
module for three steps:

``pool``     count the rows a selection keeps (fail fast on an empty filter).
``prepare``  write the selection as an aiperf ``raw_payload`` file, for an aiperf
             without the ``greenspec_*`` presets or a selection no preset covers.
             It builds each body exactly like aiperf's ``GreenSpecLoader``.
``console``  pass aiperf's console through to the job log, minus what can echo
             server error text (the Error Summary table and ERROR records),
             with each progress bar printed at most once per interval.
``guard``    run after the client, before any upload. This repo is public and
             its artifacts are downloadable, so nothing derived from row content
             may leave the runner: keep only ``ALLOWLIST`` in the artifact dir,
             strip error messages from the aiperf export (a server error can
             echo request content), withhold the server log, then fail if any
             ``question_id`` of the release, or any file over ``MAX_BYTES``,
             is still in what gets uploaded.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

META_KEYS = (
    "question_id",
    "category",
    "bucket",
    "lite",
    "isl",
    "osl",
    "thinking_src",
    "recorded",
)
MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens")
SUBSETS = ("diverse", "agentic_coding", "agentic_tools", "llm_task")
ALLOWLIST = (
    "profile_export_aiperf.json",
    "result.json",
    "spec_metrics_before.json",
    "spec_metrics_after.json",
    "benchmark_command.txt",
)
MAX_BYTES = 2 * 1024 * 1024
# aiperf's tqdm bars: "{desc}: {n}/{total} |{bar}| {pct}% [{elapsed}<{remaining}]", or
# "{desc}: {pct}% |{bar}| [{elapsed}<{remaining}]" when the total is unknown.
BAR_RE = re.compile(
    r"(?P<desc>[A-Za-z][A-Za-z _-]{0,30}): +(?:(?P<n>[\d,]+)/(?P<total>[\d,]+) +)?"
    r"(?:(?P<pct0>\d+)% +)?\|[^|]*\| *(?:(?P<pct>\d+)% +)?"
    r"\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+)\]"
)
# aiperf's log record: "14:27:38.612 INFO     message (file.py:64)".
LOG_RE = re.compile(r"\d\d:\d\d:\d\d\.\d{3} +([A-Z]+) +")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
WITHHELD_LEVELS = ("ERROR", "CRITICAL")
SERVER_LOG_STUB = (
    "server log withheld: GreenSpec rows are internal and a server log can echo them\n"
)


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def keep(row: dict[str, Any], subset: str, lite: bool, bucket: str | None) -> bool:
    return (
        (subset == "diverse" or row["category"] == subset)
        and (not lite or row["lite"])
        and (bucket is None or row["bucket"] == bucket)
    )


def select(path: Path, subset: str, lite: bool, bucket: str | None) -> list[dict[str, Any]]:
    if subset not in SUBSETS:
        raise ValueError(f"unknown GreenSpec subset {subset!r}; one of {SUBSETS}")
    return [r for r in iter_rows(path) if keep(r, subset, lite, bucket)]


def preset(subset: str, lite: bool, bucket: str | None, recorded: bool) -> str | None:
    """The aiperf preset for a selection, or None when no preset covers it."""
    if bucket is not None:
        if recorded:
            return None
        return "greenspec_" + bucket.replace("__", "_").replace("-", "_").replace("+", "_plus")
    name = "greenspec" if subset == "diverse" else f"greenspec_{subset}"
    return name + ("_lite" if lite else "") + ("_recorded" if recorded else "")


def payload(
    row: dict[str, Any],
    model: str,
    *,
    recorded: bool = False,
    extra: dict[str, Any] | None = None,
    stream: bool = True,
) -> dict[str, Any]:
    """One row -> the request body aiperf's GreenSpecLoader sends for it."""
    body = {k: v for k, v in row.items() if k not in META_KEYS}
    if recorded:
        body.update(row.get("recorded") or {})
    for key, value in (extra or {}).items():
        capped = [k for k in MAX_TOKEN_KEYS if k in body]
        if key in MAX_TOKEN_KEYS and capped and isinstance(value, int):
            for k in capped:
                body[k] = min(body[k], value)
        else:
            body[key] = value
    out: dict[str, Any] = {"model": model, **body}
    if stream:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


class ConsoleFilter:
    """aiperf's console for the job log, minus what can echo server error text.

    The Error Summary table is dropped (its Message column is the raw server
    response) and replaced by its codes and counts; an ERROR or CRITICAL record
    keeps its timestamp and level and loses its text and continuation lines (a
    traceback, wrapped text). An unterminated table stays dropped. Each progress
    bar is rebuilt from its numbers and printed at most once per ``interval``
    seconds, plus once at 100%.
    """

    def __init__(
        self,
        write: Callable[[str], None],
        interval: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._write = write
        self._interval = interval
        self._clock = clock
        self._bars: dict[str, tuple[float, str]] = {}
        self._in_table = False
        self._dropping = False
        self._errors: list[str] = []

    def feed(self, segment: str) -> None:
        text = ANSI_RE.sub("", segment).rstrip()
        if not text:  # tqdm's \r next to a record's \n leaves empty segments
            return
        bar = BAR_RE.match(text.lstrip())
        if bar:
            self._dropping = False  # a bar ends the continuation of an ERROR record
            self._bar(bar)
            # tqdm writes no newline, so a log record can follow on the same line.
            text = text.lstrip()[bar.end():]
            if not text.strip():
                return
        if self._in_table:
            self._table_line(text)
            return
        log = LOG_RE.match(text)
        if log:
            self._dropping = log.group(1) in WITHHELD_LEVELS
            if self._dropping:
                self._write(f"{text[:log.end()]}<withheld: may echo server error text>")
                return
        elif "Error Summary" in text:
            self._in_table, self._dropping = True, False
            return
        elif self._dropping:
            return
        self._write(text)

    def close(self) -> None:
        if self._in_table:
            self._in_table = False
            self._report_errors()

    def _bar(self, m: re.Match[str]) -> None:
        desc, pct = m["desc"].strip(), m["pct"] or m["pct0"]
        now, last = self._clock(), self._bars.get(desc)
        if last and (last[1] == pct == "100" or (pct != "100" and now - last[0] < self._interval)):
            return
        self._bars[desc] = (now, pct)
        count = f" {m['n']}/{m['total']}" if m["n"] else ""
        self._write(
            f"GreenSpec progress: {desc}{count} ({pct}%) elapsed {m['elapsed']} eta {m['eta']}"
        )

    def _table_line(self, text: str) -> None:
        if text.lstrip().startswith(("└", "╰")):
            self._in_table = False
            self._report_errors()
            return
        cells = [c.strip() for c in text.split("│")]
        if len(cells) >= 4 and (cells[1].isdigit() or cells[1] == "N/A"):
            count = cells[-2].replace(",", "")
            if count.isdigit():
                self._errors.append(f"{cells[1]} x{count}")

    def _report_errors(self) -> None:
        listed = ", ".join(self._errors) or "no rows parsed"
        self._write(f"GreenSpec: aiperf error summary withheld (messages can echo requests): {listed}")
        self._errors = []


def filter_console(stream: Any, write: Callable[[str], None], interval: float) -> None:
    """Feed a binary stream to ``ConsoleFilter`` one \\r- or \\n-delimited segment at a time.

    A segment the filter fails on is dropped rather than ending the read: aiperf
    writes into this pipe and would die of a broken pipe.
    """
    console = ConsoleFilter(write, interval)

    def feed(raw: bytes) -> None:
        try:
            console.feed(raw.decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            write("GreenSpec: console line dropped (filter error)")

    pending = b""
    while chunk := stream.read1(65536):
        *segments, pending = re.split(rb"[\r\n]", pending + chunk)
        for segment in segments:
            feed(segment)
    if pending:
        feed(pending)
    console.close()


def _redact_errors(export_path: Path) -> None:
    export = json.loads(export_path.read_text())
    for entry in export.get("error_summary") or []:
        details = entry.get("error_details")
        if isinstance(details, dict) and "message" in details:
            details["message"] = "<withheld>"
    export_path.write_text(json.dumps(export, indent=2))


def guard(
    dataset: Path,
    artifact_dir: Path,
    uploads: list[Path],
    server_logs: list[Path],
    keep_log_dir: Path | None = None,
) -> list[str]:
    """Prune, redact and scan everything a GreenSpec job uploads. Returns the problems."""
    if artifact_dir.is_dir():
        for p in sorted(artifact_dir.rglob("*"), reverse=True):
            rel = p.relative_to(artifact_dir).as_posix()
            if p.is_dir():
                if not any(p.iterdir()):
                    p.rmdir()
            elif rel not in ALLOWLIST:
                p.unlink()
        export = artifact_dir / "profile_export_aiperf.json"
        if export.is_file():
            _redact_errors(export)
    for log in server_logs:
        if log.is_file():
            if keep_log_dir is not None:
                keep_log_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(log), str(keep_log_dir / log.name))
            log.write_text(SERVER_LOG_STUB)
    qids = {r["question_id"] for r in iter_rows(dataset)}
    files = [p for p in artifact_dir.rglob("*") if p.is_file()] if artifact_dir.is_dir() else []
    files += [p for p in uploads if p.is_file()]
    problems = []
    for p in files:
        size = p.stat().st_size
        if size > MAX_BYTES:
            problems.append(f"{p}: {size} bytes > {MAX_BYTES}")
            continue
        text = p.read_text(errors="replace")
        if any(q in text for q in qids):
            problems.append(f"{p}: contains a GreenSpec question_id")
    return problems


def _selection(args: argparse.Namespace) -> tuple[str, bool, str | None]:
    return args.subset, args.lite == "1", args.bucket or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)

    def selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("--file", type=Path, required=True)
        p.add_argument("--subset", default="diverse")
        p.add_argument("--lite", default="0", choices=("0", "1"))
        p.add_argument("--bucket", default="")

    pool = commands.add_parser("pool", help="count the rows a selection keeps")
    selection(pool)
    name = commands.add_parser("preset", help="print the aiperf preset, empty when none")
    name.add_argument("--subset", default="diverse")
    name.add_argument("--lite", default="0", choices=("0", "1"))
    name.add_argument("--bucket", default="")
    name.add_argument("--mode", default="target-default", choices=("target-default", "recorded"))
    prep = commands.add_parser("prepare", help="write a selection as a raw_payload file")
    selection(prep)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--mode", default="target-default", choices=("target-default", "recorded"))
    prep.add_argument("--extra", default="{}", help="JSON object, as --extra-inputs")
    prep.add_argument("--no-stream", action="store_true")
    cons = commands.add_parser("console", help="filter aiperf's console (stdin) for the job log")
    cons.add_argument("--interval", type=float, default=60.0, help="seconds between bar lines")
    grd = commands.add_parser("guard", help="prune, redact and scan before upload")
    grd.add_argument("--file", type=Path, required=True)
    grd.add_argument("--artifact-dir", type=Path, required=True)
    grd.add_argument("--upload", type=Path, action="append", default=[])
    grd.add_argument("--server-log", type=Path, action="append", default=[])
    grd.add_argument("--keep-log-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.operation == "pool":
        print(len(select(args.file, *_selection(args))))
    elif args.operation == "preset":
        lite, bucket = args.lite == "1", args.bucket or None
        print(preset(args.subset, lite, bucket, args.mode == "recorded") or "")
    elif args.operation == "prepare":
        rows = select(args.file, *_selection(args))
        extra = json.loads(args.extra)
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                body = payload(
                    row,
                    args.model,
                    recorded=args.mode == "recorded",
                    extra=extra,
                    stream=not args.no_stream,
                )
                f.write(json.dumps(body, ensure_ascii=False) + "\n")
        print(len(rows))
    elif args.operation == "console":
        filter_console(sys.stdin.buffer, lambda line: print(line, flush=True), args.interval)
    else:
        problems = guard(
            args.file, args.artifact_dir, args.upload, args.server_log, args.keep_log_dir
        )
        for problem in problems:
            print(f"GreenSpec guard: {problem}", file=sys.stderr)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
