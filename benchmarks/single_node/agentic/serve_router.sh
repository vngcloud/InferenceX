#!/usr/bin/env bash
set -eo pipefail
set -x

# Router sidecar for the serving-YAML router path (any router; Dynamo is the "dynamo" profile)
# (docs/superpowers/specs/2026-10-13-serving-router-dynamo-design.md §4).
# runners/launch_hardware-hcm.sh starts it as an srun --overlap sidecar next to
# the bench step; aiperf sends every request to $PORT.
#   ROUTER_CMD  shell-quoted frontend argv, without --http-host/--http-port
#   ROUTER_ENV  KEY=VALUE lines exported before the frontend starts
#   ROUTER_PROFILE  dynamo: frontend argv without --http-host/--http-port, appended here;
#                   generic: the full command, {{router_port}} {{engine_url}} {{engine_port}} expanded here
# Dynamo discovery (DYN_DISCOVERY_BACKEND, DYN_FILE_KV) comes from the launcher; the
# generic profile gets ENGINE_PORT instead.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars ROUTER_CMD PORT RESULT_DIR
if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
    check_env_vars DYN_DISCOVERY_BACKEND DYN_FILE_KV
else
    check_env_vars ENGINE_PORT
fi

mkdir -p "$RESULT_DIR"
# xtrace off: ROUTER_ENV values may hold tokens; log only the names.
{ set +x; } 2>/dev/null
if [[ -n "${ROUTER_ENV:-}" ]]; then
    while IFS= read -r assignment; do
        if [[ -n "$assignment" ]]; then
            if [[ "${ROUTER_PROFILE:-}" != dynamo ]]; then
                assignment="$(expand_router_placeholders "$assignment")"
            fi
            export "$assignment"
            echo "router env: ${assignment%%=*}"
        fi
    done <<< "$ROUTER_ENV"
fi
set -x

if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
    eval "set -- $ROUTER_CMD"
    ROUTER_ARGV=("$@" --http-host 0.0.0.0 --http-port "$PORT")
else
    eval "set -- $(expand_router_placeholders "$ROUTER_CMD")"
    ROUTER_ARGV=("$@")
fi
write_command "$RESULT_DIR/router_command.txt" "${ROUTER_ARGV[@]}"
exec "${ROUTER_ARGV[@]}"
