#!/usr/bin/env bash
set -euo pipefail
set -x

# Non-GPU controller runner for remote-bench recipes: this box only drives
# aiperf against an externally-managed inference endpoint, it never launches
# a local server, so there's no docker/container-image/GPU-mount setup here
# unlike every other runners/launch_*.sh. Runs the recipe directly against
# the checked-out repo in the job's working directory.
#
# benchmark-tmpl.yml sets RESULT_DIR=/workspace/results and defaults
# INFMAX_CONTAINER_WORKSPACE to /workspace, assuming every other launcher's
# docker bind-mount (-v "$GITHUB_WORKSPACE:/workspace"). No docker here, so
# point both at the actual checkout on this host instead.
export INFMAX_CONTAINER_WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
export RESULT_DIR="${INFMAX_CONTAINER_WORKSPACE}/results"

BENCH_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_${FRAMEWORK}-remote-bench.sh"

bash "$BENCH_SCRIPT"
