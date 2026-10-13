#!/usr/bin/env bash
# Launch a cloud GPU instance, run training, copy checkpoint back.
#
# Usage:
#   ./scripts/launch_gpu.sh configs/glm-4.7-flash.yaml
#
# Requires: RunPod CLI (or adapt for Vast.ai / GCP)
# Install: pip install runpod

set -euo pipefail

CONFIG="${1:?Usage: launch_gpu.sh <config.yaml>}"
GPU_TYPE="${2:-H100 80GB PCIe}"  # default: single H100
INSTANCE_NAME="spec-train-$(date +%Y%m%d-%H%M%S)"

echo "=== Launching GPU instance: $INSTANCE_NAME ==="
echo "  Config: $CONFIG"
echo "  GPU:    $GPU_TYPE"

# --- Option A: RunPod ---
# pod_id=$(runpodctl pod create --gpu-type "$GPU_TYPE" --name "$INSTANCE_NAME")
# ssh_key=$(runpodctl pod ssh "$pod_id")
# SSH_CMD="ssh $ssh_key"

# --- Option B: Manual SSH to existing GPU server ---
# Set GPU_SERVER env var: export GPU_SERVER=user@gpu-host
SSH_CMD="ssh ${GPU_SERVER:?Set GPU_SERVER=user@gpu-host}"

echo "=== Syncing code to GPU server ==="
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
    . "$GPU_SERVER:~/speculative-training-pipeline/"

echo "=== Running training on GPU server ==="
$SSH_CMD "cd ~/speculative-training-pipeline && \
    git submodule update --init && \
    pip install -e '.[gpu]' && \
    python cli.py train --config $CONFIG"

echo "=== Copying checkpoint back ==="
# Extract checkpoint dir from config
CHECKPOINT_DIR=$(python -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('training', {}).get('output_dir', './checkpoints'))
")

rsync -avz "$GPU_SERVER:~/speculative-training-pipeline/$CHECKPOINT_DIR/" \
    "./$CHECKPOINT_DIR/"

echo "=== Done. Checkpoint at: $CHECKPOINT_DIR ==="
echo "=== Remember to destroy GPU instance when done! ==="
