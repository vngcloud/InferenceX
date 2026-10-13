# shellcheck shell=bash
# Sourced by the Gemma-4 DFlash smoke jobs. Node-local layout for hgx-h200-01.
#
# /shared is NFS over a slow link from the hgx nodes: ~48 ms per metadata op and
# < 20 MB/s reads (torch import 162 s, a 32 GB model load > 30 min). Everything hot
# therefore lives on the node's local SSD:
#   venv      /tmp/phucnlt2/venv-py310  (same freeze as /shared/phucnlt2/venv-py310)
#   model     /mnt/hf_hub_cache         (admin-managed HF cache; read-only, offline)
#   features  /tmp/phucnlt2/gemma4-dflash-smoke/hidden_states  (symlinked into the repo)
#   outputs   /tmp/phucnlt2/outputs     (symlinked into the repo)
#   caches    /tmp/phucnlt2/{triton,torchinductor,flashinfer,deep-gemm,xdg}
# Code, smoke jsonl and job logs stay on /shared (small).

LOCAL_ROOT="${LOCAL_ROOT:-/tmp/phucnlt2}"
VENV_DIR="${VENV_DIR:-$LOCAL_ROOT/venv-py310}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "✗ no venv at $VENV_DIR on $(hostname); rebuild it with the local-venv job"
    exit 1
fi
source "$VENV_DIR/bin/activate"
mkdir -p /shared/phucnlt2/job-logs

export HOME=/shared/phucnlt2
export XDG_CACHE_HOME="$LOCAL_ROOT/xdg-cache"
export HF_HOME=/shared/phucnlt2/hf-cache
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
export HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE="$LOCAL_ROOT"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/torchinductor-cache"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton-cache"
# SGLang's FP8 DeepGEMM warmup sweeps every batch size for each GEMM shape at every
# launch (~25 min even with a warm cache). Compile lazily for the shapes actually seen.
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DG_CACHE_DIR="$LOCAL_ROOT/deep-gemm-cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$LOCAL_ROOT/gemma4-dflash-smoke/hidden_states" "$LOCAL_ROOT/outputs"

# Point a repo path at node-local storage. A real directory left by an older run
# is kept (moved aside), never deleted.
link_local() {
    local repo_path="$1" local_path="$2"
    mkdir -p "$local_path" "$(dirname "$repo_path")"
    if [ -d "$repo_path" ] && [ ! -L "$repo_path" ]; then
        mv "$repo_path" "${repo_path}.shared-$(date +%s)"
    fi
    ln -sfn "$local_path" "$repo_path"
}

if [ -d /shared/phucnlt2/python-headers/usr/include/python3.10 ]; then
    PYINC="/shared/phucnlt2/python-headers/usr/include/python3.10:/shared/phucnlt2/python-headers/usr/include"
    export C_INCLUDE_PATH="${PYINC}:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="${PYINC}:${CPLUS_INCLUDE_PATH:-}"
fi

if [ -z "${CUDA_HOME:-}" ]; then
    CUDA_HOME="$VENV_DIR/lib/python3.10/site-packages/nvidia/cu13"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
