#!/usr/bin/env bash
# Dispatch one serving YAML as an AgentX benchmark on vngcloud/InferenceX.
#   utils/serving/dispatch.sh serving/<name>.yaml [--ref <branch>]
# Validates locally first (same code CI runs), then starts e2e-tests.yml with the
# YAML as the `serving` input and prints the run URL.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GH_REPO="vngcloud/InferenceX"

usage() {
    echo "usage: utils/serving/dispatch.sh <serving.yaml> [--ref <branch>]" >&2
    exit 2
}
if [[ $# -lt 1 ]]; then
    usage
fi
yaml_file="$(realpath "$1")"
shift
# The workflow input is the file content, not HEAD: surface working-tree divergence
# so a run cannot silently differ from the committed config (run 38527925937 case).
if ! git -C "$REPO_ROOT" diff --quiet -- "$yaml_file" \
    || ! git -C "$REPO_ROOT" diff --cached --quiet -- "$yaml_file" \
    || [ -n "$(git -C "$REPO_ROOT" status --porcelain -- "$yaml_file")" ]; then
    echo "WARNING: $yaml_file differs from HEAD; the run dispatches the working-tree content." >&2
fi
ref=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)
            if [[ $# -lt 2 || -z "$2" ]]; then
                usage
            fi
            ref="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

title=$(cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT" \
    uv run -q --no-project --python 3.12 --with pydantic --with pyyaml \
    python -P -m infx.serving.to_matrix --check "$yaml_file") || exit 2

if [[ -z "$ref" ]]; then
    ref="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
    remote_sha="$(git -C "$REPO_ROOT" ls-remote origin "refs/heads/$ref" | cut -f1)"
    if [[ -z "$remote_sha" ]]; then
        echo "ERROR branch $ref is not on origin; push it first (CI runs the pushed code)" >&2
        exit 2
    fi
    if [[ "$remote_sha" != "$(git -C "$REPO_ROOT" rev-parse HEAD)" ]]; then
        echo "ERROR local HEAD differs from origin/$ref; push or pull first (CI runs the pushed code)" >&2
        exit 2
    fi
fi

# Epoch seconds, 60 s early: tolerate a local clock running ahead of GitHub's.
started=$(( $(date +%s) - 60 ))
gh workflow run e2e-tests.yml --repo "$GH_REPO" --ref "$ref" \
    -f serving="$(cat "$yaml_file")" \
    -f test-name="$title"

for _ in $(seq 1 30); do
    sleep 4
    run_id=$(gh run list --repo "$GH_REPO" -w e2e-tests.yml -L 20 \
        --json databaseId,displayTitle,createdAt \
        --jq "[.[] | select(.displayTitle == \"e2e Test - $title\" and (.createdAt | fromdateiso8601) >= $started)][0].databaseId // empty")
    if [[ -n "$run_id" ]]; then
        echo "https://github.com/$GH_REPO/actions/runs/$run_id"
        exit 0
    fi
done
echo "ERROR dispatched, but the run did not show up within 2 minutes; do not redispatch, check: gh run list --repo $GH_REPO -w e2e-tests.yml -L 5" >&2
exit 1
