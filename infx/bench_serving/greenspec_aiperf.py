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
``progress`` print the numbers of aiperf's progress bars from its console log,
             which the job log never sees (the console can echo server error
             text), so a long cell can be followed.
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
from collections.abc import Iterator
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
# aiperf's tqdm bar: "{desc}: {n}/{total} |{bar}| {pct}% [{elapsed}<{remaining}]".
PROGRESS_RE = re.compile(
    r"([A-Za-z][A-Za-z _-]{0,30}): +([\d,]+)/([\d,]+) \|[^|]*\| *(\d+)% "
    r"\[([\d:]+)<([\d:?]+)\]"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PROGRESS_TAIL_BYTES = 64 * 1024
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
        # Bucket presets cover the full bucket only: no lite or recorded variants.
        if lite or recorded:
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


def progress(console_log: Path) -> list[str]:
    """The latest state of each aiperf progress bar, rebuilt from its numbers only."""
    if not console_log.is_file():
        return []
    with open(console_log, "rb") as f:
        f.seek(max(0, console_log.stat().st_size - PROGRESS_TAIL_BYTES))
        text = f.read().decode("utf-8", errors="replace")
    latest: dict[str, str] = {}
    for line in re.split(r"[\r\n]", ANSI_RE.sub("", text)):
        m = PROGRESS_RE.search(line)
        if m:
            desc, n, total, pct, elapsed, remaining = m.groups()
            latest[desc.strip()] = (
                f"{desc.strip()} {n}/{total} ({pct}%) elapsed {elapsed} eta {remaining}"
            )
    return list(latest.values())


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
    prog = commands.add_parser("progress", help="print aiperf progress-bar numbers")
    prog.add_argument("--console-log", type=Path, required=True)
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
    elif args.operation == "progress":
        for state in progress(args.console_log):
            print(f"GreenSpec progress: {state}")
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
