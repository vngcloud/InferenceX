#!/usr/bin/env bash
set -eo pipefail
set -x

# Dynamo frontend for the serving-YAML router path
# (docs/superpowers/specs/2026-10-13-serving-router-dynamo-design.md §4).
# runners/launch_hardware-hcm.sh starts it as an srun --overlap sidecar next to
# the bench step; aiperf sends every request to $PORT.
#   ROUTER_CMD  shell-quoted frontend argv, without --http-host/--http-port
#   ROUTER_ENV  KEY=VALUE lines exported before the frontend starts
# Discovery (DYN_DISCOVERY_BACKEND, DYN_FILE_KV) comes from the launcher.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars ROUTER_CMD PORT RESULT_DIR DYN_DISCOVERY_BACKEND DYN_FILE_KV

mkdir -p "$RESULT_DIR"
# xtrace off: ROUTER_ENV values may hold tokens; log only the names.
{ set +x; } 2>/dev/null
if [[ -n "${ROUTER_ENV:-}" ]]; then
    while IFS= read -r assignment; do
        if [[ -n "$assignment" ]]; then
            export "$assignment"
            echo "router env: ${assignment%%=*}"
        fi
    done <<< "$ROUTER_ENV"
fi
set -x

eval "set -- $ROUTER_CMD"
ROUTER_ARGV=("$@" --http-host 0.0.0.0 --http-port "$PORT")
write_command "$RESULT_DIR/router_command.txt" "${ROUTER_ARGV[@]}"
exec "${ROUTER_ARGV[@]}"
