#!/usr/bin/env python3
"""Find which gateway log field holds a user id, and check if its payloads are masked.

Prints hit counts per candidate field and payload sizes only — never message content.

Usage:
  python scripts/probe_gateway_user_field.py --model gemma-4-31b-it \
      --user user-108942-user-109951-lamtt7 --start 2026-09-18T00:00:00Z --end 2026-09-25T00:00:00Z
"""

from __future__ import annotations

import argparse

from spec_data.data.gateway_convert import (
    _AI_GATEWAY_PROJECT_ID,
    _load_credentials,
    _VMonitorLogClient,
)

CANDIDATE_FIELDS = [
    "maas_user_id",
    "maas_user_id.keyword",
    "response.headers.x-consumer-username.keyword",
    "request.headers.x-greennode-agentbase-user-id.keyword",
]


def _id_like_paths(doc: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Field paths whose name looks like a user/consumer id, with their (short) values."""
    out = []
    for k, v in doc.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out += _id_like_paths(v, path + ".")
        elif any(s in k.lower() for s in ("user", "consumer")) and isinstance(v, str | int):
            out.append((path, str(v)[:80]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--project", default=_AI_GATEWAY_PROJECT_ID)
    args = ap.parse_args()

    client = _VMonitorLogClient(*_load_credentials())
    model_match = {
        "type": "match",
        "value": {"field": "ai.proxy.meta.request_model.keyword", "value": args.model},
    }

    found = None
    for field in CANDIDATE_FIELDS:
        user_match = {"type": "match", "value": {"field": field, "value": args.user}}
        query = {"must": [model_match, user_match]}
        try:
            r = client.search_logs(args.project, query, args.start, args.end, size=1)
        except Exception as e:  # noqa: BLE001
            print(f"{field:60s} ERROR {type(e).__name__}")
            continue
        hits = r.get("hits", {})
        total = hits.get("total", {})
        total = total.get("value", total) if isinstance(total, dict) else total
        print(f"{field:60s} hits={len(hits.get('hits', []))} total={total}")
        if hits.get("hits") and found is None:
            found = hits["hits"][0]["_source"]

    if found is None:
        print("\nNo candidate field matched. Check field names with get_log_project_mappings.")
        return

    print("\nUser/consumer-like fields in a matching doc:")
    for path, value in _id_like_paths(found):
        print(f"  {path} = {value}")
    payload = found.get("ai", {}).get("proxy", {}).get("payload", {})
    print("\nPayload sizes (chars):")
    for k in ("request", "response"):
        v = payload.get(k)
        print(f"  {k}: {len(v) if isinstance(v, str) else type(v).__name__}")


if __name__ == "__main__":
    main()
