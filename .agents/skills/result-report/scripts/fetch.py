"""Find benchmark configs in a GitHub Actions run or a local folder, and read their files.

A config is one benchmark job: the small aggregate JSON (artifact ``bmk_agentic_<RF>``)
plus the raw results tree (artifact ``agentic_<RF>``). Remote artifact zips are read
member by member over HTTP Range requests, so only the needed bytes are downloaded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO = "vngcloud/InferenceX"
AGG_PREFIX = "bmk_agentic_"
RAW_PREFIX = "agentic_"
RUN_URL_RE = re.compile(r"github\.com/([^/\s]+/[^/\s]+)/actions/runs/(\d+)")
FETCHED = {"bytes": 0}  # downloaded bytes, reported by the CLI


@dataclass(frozen=True)
class Source:
    kind: str  # "run" or "dir"
    repo: str | None = None
    run_id: str | None = None
    path: Path | None = None

    @property
    def link(self) -> str:
        if self.kind == "run":
            return f"https://github.com/{self.repo}/actions/runs/{self.run_id}"
        return str(self.path)


def parse_source(arg: str, default_repo: str = DEFAULT_REPO) -> Source:
    match = RUN_URL_RE.search(arg)
    if match:
        return Source("run", repo=match.group(1), run_id=match.group(2))
    if arg.isdigit():
        return Source("run", repo=default_repo, run_id=arg)
    path = Path(arg).expanduser()
    if path.is_dir():
        return Source("dir", path=path.resolve())
    raise ValueError(f"not a run URL, run id or directory: {arg}")


class DirReader:
    """Read files under one local results root."""

    def __init__(self, root: Path):
        self.root = root

    def names(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())

    def read(self, rel: str) -> bytes | None:
        path = self.root / rel
        return path.read_bytes() if path.is_file() else None


@dataclass
class Config:
    rf: str  # pairing key: artifact suffix <RF>, or a relative path for hand-named folders
    source: Source
    agg: dict | None  # aggregate JSON
    files: object | None  # DirReader / ZipReader over the results root


def _is_aggregate(doc: object) -> bool:
    return isinstance(doc, dict) and "infmax_model_prefix" in doc and ("request_metrics" in doc or "state" in doc)


def _key(rel: Path, prefix: str) -> str:
    """GH artifact folders pair on their <RF> suffix; anything else keys on its relative path."""
    for part in reversed(rel.parts):
        if part.startswith(prefix):
            return part[len(prefix):]
    return rel.as_posix()


def local_configs(src: Source) -> list[Config]:
    root = src.path
    aggs: dict[str, dict] = {}
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root)
        if any(p.startswith("search_iter_") or p in ("aiperf_artifacts", "aggregate") for p in rel.parts):
            continue
        if path.stat().st_size > 1_000_000:
            continue
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if _is_aggregate(doc):
            aggs[_key(rel, AGG_PREFIX)] = doc
    roots: dict[str, Path] = {}
    for name in ("profile_export_aiperf.json", "search_history.json"):
        for path in sorted(root.rglob(name)):
            rel = path.relative_to(root)
            if any(p.startswith("search_iter_") or p == "aggregate" for p in rel.parts):
                continue
            base = path.parent.parent if path.parent.name == "aiperf_artifacts" else path.parent
            roots.setdefault(_key(base.relative_to(root), RAW_PREFIX), base)
    lone_aggs = [k for k in aggs if k not in roots]
    lone_roots = [k for k in roots if k not in aggs]
    if len(lone_aggs) == 1 and len(lone_roots) == 1:  # e.g. hand-named ag/ + bmk/ copies
        roots[lone_aggs[0]] = roots.pop(lone_roots[0])
    return [
        Config(rf=k, source=src, agg=aggs.get(k), files=DirReader(roots[k]) if k in roots else None)
        for k in sorted(set(aggs) | set(roots))
    ]


def discover(src: Source, cache_dir: Path, max_full_zip_mb: float) -> tuple[list[Config], list[str]]:
    configs = local_configs(src)
    return configs, ([] if configs else [f"{src.link}: no aggregate JSON or aiperf export found"])
