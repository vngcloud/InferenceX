#!/usr/bin/env python3
"""Shared helper: scrape a stack's gpu_metrics_url (DCGM Prometheus feed,
from /discover) for its GPU model, so live-check artifacts can bake a
gpu_model snapshot into their own JSON at test time -- see
design/throughput-test.md and the gpu-metrics discussion with
InferenceX-app for why (ingest-time queries against live infra state are
unreliable for backfill / lagged ingest).
"""
from __future__ import annotations

import re

import requests

REQUEST_TIMEOUT_S = 10

# DCGM's Prometheus exposition format labels every metric line with
# modelName="..." per GPU, e.g.:
#   DCGM_FI_DEV_SM_CLOCK{gpu="0",...,modelName="NVIDIA GeForce RTX 5090",...} 2400
_MODEL_NAME_RE = re.compile(r'modelName="([^"]+)"')


def fetch_gpu_model(gpu_metrics_url: str | None) -> str | None:
    """Return the GPU model reported by gpu_metrics_url, or None if the url
    is falsy or no modelName label is found (e.g. an empty/unreachable feed).

    Raises ValueError if more than one distinct modelName is present -- a
    pod with heterogeneous GPUs is a real anomaly worth surfacing loudly,
    not something to silently resolve by picking one.
    """
    if not gpu_metrics_url:
        return None

    resp = requests.get(gpu_metrics_url, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()

    models = set(_MODEL_NAME_RE.findall(resp.text))
    if not models:
        return None
    if len(models) > 1:
        raise ValueError(
            f"gpu_metrics_url reported multiple distinct GPU models in one "
            f"pod: {sorted(models)} -- can't pick one, this needs a schema "
            f"decision, not a guess."
        )
    return next(iter(models))
