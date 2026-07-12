"""Client for inference-cicd's live /discover self-report endpoint.

See design/smoke-test-matrix.md -- /discover is the source of truth for
what's deployed and where. This module never hardcodes stack metadata;
it only fetches and returns whatever the live endpoint reports.
"""
from __future__ import annotations

import requests

DEFAULT_DISCOVER_URL = "http://116.118.91.176.nip.io/discover"
REQUEST_TIMEOUT_S = 10


def fetch_discover(discover_url: str = DEFAULT_DISCOVER_URL) -> dict:
    """Fetch and return the full /discover payload."""
    resp = requests.get(discover_url, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def get_stack(discover_payload: dict, name: str) -> dict | None:
    """Return the /discover entry for `name`, or None if not registered."""
    for stack in discover_payload.get("stacks", []):
        if stack.get("name") == name:
            return stack
    return None


def fetch_version(version_url: str) -> dict:
    """Fetch a stack's per-stack version/metadata self-report."""
    resp = requests.get(version_url, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()
