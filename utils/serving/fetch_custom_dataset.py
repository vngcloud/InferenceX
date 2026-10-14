"""Fetch one lock-pinned file dataset into the HF cache; print its local path.

    python utils/serving/fetch_custom_dataset.py <name>          # path on stdout
    python utils/serving/fetch_custom_dataset.py --check <name>  # 0 locked, 1 not, 2 bad lock

<name> is a key of scenarios/datasets/lock.yaml (repo, revision, file, sha256).
Token: GREENLAB_HF_TOKEN when set (public repos too, so private ones need no code
change); unset means truly anonymous (not the runner's HF_TOKEN). Read only here, never echoed. Status and errors go to
stderr; failures are one `ERROR ...` line and exit 1. Writes
$RESULT_DIR/dataset.txt (name repo@revision file sha256 path) when RESULT_DIR is set.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import NoReturn

import yaml

LOCK = Path(__file__).resolve().parents[2] / "scenarios" / "datasets" / "lock.yaml"


def fail(message: str, code: int = 1) -> NoReturn:
    print(f"ERROR {message}", file=sys.stderr)
    sys.exit(code)


def load_lock() -> dict:
    return (yaml.safe_load(LOCK.read_text()) or {}) if LOCK.is_file() else {}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> str:
    entry = load_lock().get(name)
    if not entry:
        fail(f"dataset {name}: not in scenarios/datasets/lock.yaml")
    token = os.environ.get("GREENLAB_HF_TOKEN") or None
    state = "set" if token else "unset"
    print(f"GREENLAB_HF_TOKEN: {state}", file=sys.stderr)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    where = f"{entry['repo']}@{entry['revision']}"
    unreadable = (f"dataset {name}: {where} not readable; GREENLAB_HF_TOKEN is {state} "
                  f"on runner {os.environ.get('RUNNER_NAME', '?')}")
    try:
        path = hf_hub_download(entry["repo"], entry["file"], revision=entry["revision"],
                               repo_type="dataset", token=token or False)
    except RepositoryNotFoundError:  # includes GatedRepoError
        fail(unreadable)
    except HfHubHTTPError as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) in (401, 403):
            fail(unreadable)
        raise
    actual = sha256(path)
    if actual != entry["sha256"]:
        fail(f"dataset {name}: sha256 mismatch for {where}/{entry['file']}: expected {entry['sha256']}, got {actual}")
    if os.environ.get("RESULT_DIR"):
        Path(os.environ["RESULT_DIR"], "dataset.txt").write_text(
            f"{name} {where} file={entry['file']} sha256={actual} path={path}\n")
    return path


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--check":
        try:
            lock = load_lock()
        except (OSError, yaml.YAMLError) as exc:
            fail(f"scenarios/datasets/lock.yaml unreadable: {exc}".replace("\n", " "), 2)
        return 0 if isinstance(lock, dict) and argv[1] in lock else 1
    if len(argv) != 1:
        fail("usage: fetch_custom_dataset.py [--check] <name>")
    print(fetch(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
