#!/usr/bin/env bash
# Dispatch one serving YAML over the sweep scenarios (all, or those matching --only).
#   utils/serving/sweep_dispatch.sh serving/<name>.yaml [--only 'sweep-isl8k-*'] [--ref <branch>]
# For each matching scenarios/sweep-*.yaml it writes a temp copy of the serving
# YAML with scenario: replaced, conc: 1 and name: suffixed with the scenario's
# short name, then dispatches it through utils/serving/dispatch.sh.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

usage() {
    echo "usage: utils/serving/sweep_dispatch.sh <serving.yaml> [--only <glob>] [--ref <branch>]" >&2
    exit 2
}
[[ $# -ge 1 ]] || usage
src="$(realpath "$1")"
shift
only='sweep-*'
if [[ $# -ge 1 && "$1" == --only ]]; then
    [[ $# -ge 2 && -n "$2" ]] || usage
    only="$2"; shift 2
fi
[[ -f "$src" ]] || { echo "ERROR no such serving YAML: $src" >&2; exit 2; }

shopt -s nullglob
# shellcheck disable=SC2206  # $only must glob; assignment context does not word-split
matches=("$REPO_ROOT"/scenarios/$only.yaml)
shopt -u nullglob
if [[ ${#matches[@]} -eq 0 ]]; then
    echo "ERROR no scenarios/sweep-*.yaml match '$only'" >&2
    exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
for scenario_file in "${matches[@]}"; do
    scenario="$(basename "$scenario_file" .yaml)"
    short="${scenario#sweep-}"
    tmp="$tmp_dir/$scenario.yaml"
    sed -E "s|^scenario:.*|scenario: $scenario|; s|^conc:.*|conc: 1|; s|^name:[[:space:]]*(.*)|name: \1-$short|" "$src" > "$tmp"
    echo "== $scenario"
    "$REPO_ROOT/utils/serving/dispatch.sh" "$tmp" "$@"
done
