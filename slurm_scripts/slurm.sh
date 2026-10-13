#!/usr/bin/env bash
# Slurm workflow wrapper: sync code, submit jobs, view logs.
#
# Usage:
#   ./scripts/slurm.sh sync              # rsync local code → slurm client
#   ./scripts/slurm.sh submit [job.sh] [VAR=value ...]
#                                        # sbatch a job script (default: job_train.sh);
#                                        # VAR=value pairs are exported to the job
#   ./scripts/slurm.sh status            # squeue -u <user>
#   ./scripts/slurm.sh log <jobid>       # tail -f the job's stdout
#   ./scripts/slurm.sh ssh               # interactive SSH to slurm client
#   ./scripts/slurm.sh check             # check env: python, conda, specforge, GPUs
#   ./scripts/slurm.sh setup-vllm        # one-time: separate venv with vLLM (data quality)
#
# Config (override via env vars or edit defaults below):
#   SLURM_HOST=59.153.246.135
#   SLURM_PORT=234
#   SLURM_USER=phucnlt2
#   REMOTE_DIR=/shared/phucnlt2/speculative-training-pipeline
#   SLURM_PASS_FILE=~/.slurm_pass  (password file for sshpass)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SLURM_HOST="${SLURM_HOST:-59.153.246.135}"
SLURM_PORT="${SLURM_PORT:-234}"
SLURM_USER="${SLURM_USER:-phucnlt2}"
REMOTE_DIR="${REMOTE_DIR:-/shared/phucnlt2/speculative-training-pipeline}"

# ── SSH setup ─────────────────────────────────────────────────────────────────
# Home dir on Slurm client is read-only → can't install SSH keys.
# Use sshpass with a password file instead.  Setup:
#   sudo apt install -y sshpass
#   read -s -p "Slurm password: " p && echo "$p" > ~/.slurm_pass && chmod 600 ~/.slurm_pass
SLURM_PASS_FILE="${SLURM_PASS_FILE:-$HOME/.slurm_pass}"
SSH_BASE="ssh -o PreferredAuthentications=password -o StrictHostKeyChecking=no -p ${SLURM_PORT}"
SSH_HOST="${SLURM_USER}@${SLURM_HOST}"
if [ -f "$SLURM_PASS_FILE" ]; then
    SSH_CMD="sshpass -f $SLURM_PASS_FILE $SSH_BASE $SSH_HOST"
    RSYNC_SSH="sshpass -f $SLURM_PASS_FILE $SSH_BASE"
else
    SSH_CMD="$SSH_BASE $SSH_HOST"
    RSYNC_SSH="$SSH_BASE"
fi
RSYNC_CMD="rsync -avz --delete"

# Files/dirs to exclude from sync (SpecForge is set up once on remote via git clone)
EXCLUDES=(
    --exclude '.git/'
    --exclude 'third_party/'
    --include '/data/'
    --include '/data/gateway-smoke/'
    --include '/data/gateway-smoke/**'
    --include '/data/speedbench_inspect/'
    --include '/data/speedbench_inspect/throughput_1k.jsonl'
    --include '/data/*/'
    --include '/data/*/bench/'
    --include '/data/*/bench/**'
    # Smoke rows sync; captured features live only on the cluster (excluded → never deleted).
    # No trailing slash: on the cluster this is a symlink to node-local disk.
    --exclude '/data/gemma4-dflash-smoke/hidden_states'
    --include '/data/gemma4-dflash-smoke/'
    --include '/data/gemma4-dflash-smoke/**'
    --exclude '/data/**'
    --exclude '/checkpoints/'
    --exclude '/deploy/'
    --exclude '/tmp/'
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude '*.egg-info'
    --exclude '.venv/'
    --exclude '.mypy_cache/'
    --exclude '.ruff_cache/'
    --exclude '/outputs/'
    --exclude '/cache/'
)

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_sync() {
    echo "→ Syncing code to ${SLURM_USER}@${SLURM_HOST}:${REMOTE_DIR}"
    $RSYNC_CMD "${EXCLUDES[@]}" \
        -e "$RSYNC_SSH" \
        ./ "${SLURM_USER}@${SLURM_HOST}:${REMOTE_DIR}/"
    echo "✓ Sync done"
}

cmd_submit() {
    local job_script="${1:-scripts/job_train.sh}"
    shift || true
    # Local env does not cross ssh: pass job overrides as VAR=value args. sbatch's
    # default --export=ALL hands them to the job.
    local env_prefix=""
    local kv
    for kv in "$@"; do
        case "$kv" in
            [A-Za-z_]*=*) env_prefix+="$(printf '%q ' "$kv")" ;;
            *) echo "✗ expected VAR=value, got: $kv"; exit 1 ;;
        esac
    done
    echo "→ Submitting ${job_script} on slurm client ${env_prefix:+(env: ${env_prefix})}"
    # Sync first so remote has latest job script + code
    cmd_sync
    # sbatch on remote
    $SSH_CMD "cd ${REMOTE_DIR} && env ${env_prefix} sbatch ${job_script}"
}

cmd_status() {
    $SSH_CMD "squeue -u ${SLURM_USER}"
}

cmd_log() {
    local jobid="${1:?Usage: slurm.sh log <jobid>}"
    # Log files are /shared/<job-name>_<jobid>.out per the example scripts
    echo "→ Tailing log for job ${jobid} (Ctrl+C to stop)"
    $SSH_CMD "tail -f /shared/phucnlt2/job-logs/*_${jobid}.out" 2>/dev/null || \
        $SSH_CMD "ls /shared/phucnlt2/job-logs/*_${jobid}.* 2>/dev/null || echo 'No log files found for job ${jobid}'"
}

cmd_ssh() {
    echo "→ SSH to ${SLURM_USER}@${SLURM_HOST}:${SLURM_PORT}"
    $SSH_CMD
}

cmd_check() {
    echo "→ Checking environment on slurm client"
    $SSH_CMD bash <<'REMOTE'
    echo "=== Python ==="
    which python3 && python3 --version
    echo "=== Conda ==="
    which conda 2>/dev/null && conda env list || echo "no conda"
    echo "=== Venv ==="
    VENV_DIR=/shared/phucnlt2/venv
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        echo "✓ venv at $VENV_DIR"
        echo "=== SpecForge ==="
        which specforge 2>/dev/null && echo "specforge found" || echo "specforge NOT found"
        echo "=== spec-data ==="
        which spec-data 2>/dev/null && echo "spec-data found" || echo "spec-data NOT found"
        echo "=== torchrun ==="
        which torchrun 2>/dev/null && echo "torchrun found" || echo "torchrun NOT found"
        deactivate
    else
        echo "✗ venv NOT found at $VENV_DIR — run: ./scripts/slurm.sh setup"
    fi
    echo "=== vLLM venv (data quality) ==="
    if [ -x /shared/phucnlt2/venv-vllm/bin/vllm ]; then
        /shared/phucnlt2/venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
    else
        echo "✗ vLLM venv NOT found — run: ./scripts/slurm.sh setup-vllm"
    fi
    echo "=== GPU nodes ==="
    sinfo -p test -o "%n %G %T %C" 2>/dev/null | head -10
    echo "=== Shared storage ==="
    df -h /shared
    echo "=== Repo ==="
    ls /shared/phucnlt2/speculative-training-pipeline/ 2>/dev/null | head -10
REMOTE
}

cmd_setup() {
    echo "→ One-time setup on slurm client"
    echo "  1) Clone SpecForge submodule"
    echo "  2) Create venv + install spec-data + SpecForge"
    $SSH_CMD bash <<REMOTE
set -euo pipefail
cd ${REMOTE_DIR}

echo "=== Cloning SpecForge ==="
if [ -d third_party/SpecForge/.git ]; then
    echo "✓ SpecForge already a git repo"
elif [ -d third_party/SpecForge ]; then
    echo "→ third_party/SpecForge exists but not a git repo, removing..."
    rm -rf third_party/SpecForge
    git clone https://github.com/Noridom1/SpecForge.git third_party/SpecForge
    echo "✓ SpecForge cloned"
else
    mkdir -p third_party
    git clone https://github.com/Noridom1/SpecForge.git third_party/SpecForge
    echo "✓ SpecForge cloned"
fi

echo "=== Creating venv ==="
VENV_DIR=/shared/phucnlt2/venv
if [ ! -f "\$VENV_DIR/bin/activate" ]; then
    # System python may lack python3-venv/ensurepip, so use --without-pip
    # and bootstrap pip via get-pip.py
    python3 -m venv --without-pip "\$VENV_DIR"
    source "\$VENV_DIR/bin/activate"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
    pip install --upgrade pip wheel
    pip install -e ".[dev]"
    pip install -e third_party/SpecForge
    echo "✓ venv created + packages installed"
else
    echo "✓ venv already exists"
fi

echo "=== Done ==="
REMOTE
}

# Separate venv for the vLLM server used by `spec-data data-quality --method model`.
# vLLM pins its own torch; keeping it apart leaves SpecForge's torch in the main venv intact.
# spec-data (main venv) only talks to it over HTTP via data_quality.entropy_vllm_executable.
cmd_setup_vllm() {
    local vllm_spec="vllm${VLLM_VERSION:+==${VLLM_VERSION}}"
    echo "→ Creating vLLM venv on slurm client (${vllm_spec})"
    $SSH_CMD bash <<REMOTE
set -euo pipefail
VENV_DIR=/shared/phucnlt2/venv-vllm
export HOME=/shared/phucnlt2
export XDG_CACHE_HOME=/shared/phucnlt2/.cache
if [ ! -f "\$VENV_DIR/bin/activate" ]; then
    python3 -m venv --without-pip "\$VENV_DIR"
    source "\$VENV_DIR/bin/activate"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
    pip install --upgrade pip wheel
else
    source "\$VENV_DIR/bin/activate"
fi
pip install "${vllm_spec}"
python -c "import vllm, torch; print('✓ vllm', vllm.__version__, '| torch', torch.__version__)"
echo "✓ vLLM executable: \$VENV_DIR/bin/vllm"
REMOTE
}

# ── Main ──────────────────────────────────────────────────────────────────────

case "${1:-}" in
    sync)   cmd_sync ;;
    submit) shift; cmd_submit "$@" ;;
    status) cmd_status ;;
    log)    shift; cmd_log "$@" ;;
    ssh)    cmd_ssh ;;
    check)  cmd_check ;;
    setup)  cmd_setup ;;
    setup-vllm) cmd_setup_vllm ;;
    *)
        echo "Usage: slurm.sh {sync|submit|status|log|ssh|check|setup|setup-vllm}"
        echo ""
        echo "  sync              rsync local code → slurm client (excludes third_party, data, etc.)"
        echo "  submit [job.sh] [VAR=value ...]"
        echo "                    sync + sbatch a job script (default: scripts/job_train.sh)"
        echo "  status            squeue -u <user>"
        echo "  log <jobid>       tail -f job output log"
        echo "  ssh               interactive SSH to slurm client"
        echo "  check             check Python/conda/specforge/GPU availability"
        echo "  setup             one-time: clone SpecForge + create venv + install packages"
        echo "  setup-vllm        one-time: separate vLLM venv for data quality (VLLM_VERSION=x.y.z to pin)"
        echo ""
        echo "Env vars: SLURM_HOST, SLURM_PORT, SLURM_USER, REMOTE_DIR, SLURM_PASS_FILE"
        exit 1
        ;;
esac
