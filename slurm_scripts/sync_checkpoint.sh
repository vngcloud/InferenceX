#!/usr/bin/env bash
# Sync trained checkpoint from GPU server or HuggingFace Hub.
#
# Usage:
#   ./scripts/sync_checkpoint.sh ssh user@gpu-server ~/checkpoints/glm-4.7-flash ./checkpoints
#   ./scripts/sync_checkpoint.sh hf your-org/eagle3-glm-4.7-flash ./checkpoints

set -euo pipefail

SOURCE="${1:?Usage: sync_checkpoint.sh <source> <dest>}"
DEST="${2:?Usage: sync_checkpoint.sh <source> <dest>}"

mkdir -p "$DEST"

case "$SOURCE" in
    ssh*)
        # Format: ssh user@host:/path/to/checkpoint
        SSH_PART="${SOURCE#ssh }"
        echo "=== Syncing checkpoint via SSH: $SSH_PART → $DEST ==="
        rsync -avz --progress "$SSH_PART:" "$DEST/"
        ;;
    hf*)
        # Format: hf org/model-name
        HF_MODEL="${SOURCE#hf }"
        echo "=== Downloading checkpoint from HuggingFace: $HF_MODEL ==="
        huggingface-cli download "$HF_MODEL" --local-dir "$DEST"
        ;;
    s3*)
        # Format: s3://bucket/path
        echo "=== Syncing checkpoint from S3: $SOURCE → $DEST ==="
        aws s3 sync "$SOURCE" "$DEST"
        ;;
    *)
        echo "=== Copying checkpoint: $SOURCE → $DEST ==="
        cp -r "$SOURCE"/* "$DEST/"
        ;;
esac

echo "=== Checkpoint synced to: $DEST ==="
