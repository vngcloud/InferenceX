#!/usr/bin/env bash
# Setup InferenceX as git submodule.
#
# The InferenceX repo is the team's own repo — no fork needed.
# We can push directly to feature branches and trigger workflow_dispatch.
#
# Usage:
#   ./scripts/setup_inferencex_fork.sh                    # uses vngcloud/InferenceX
#   ./scripts/setup_inferencex_fork.sh my-org/InferenceX  # explicit repo

set -euo pipefail

REPO="${1:-vngcloud/InferenceX}"
REPO_URL="https://github.com/${REPO}.git"

echo "=== Setting up InferenceX submodule ==="
echo "  Repo: $REPO_URL"
echo ""

# Add as submodule (skip if already present)
if [ -d "third_party/InferenceX" ] && [ -f "third_party/InferenceX/.git" ]; then
  echo "  Submodule already exists, updating..."
  git submodule update --init --recursive third_party/InferenceX
else
  git submodule add "$REPO_URL" third_party/InferenceX
fi

# Commit submodule
git add .gitmodules third_party/InferenceX
git commit -m "Add InferenceX as submodule

- Repo: ${REPO_URL}
- Used by infx eval runner for benchmark scripts and CI workflow templates
- Push generated configs to feature branches and trigger workflow_dispatch

Co-Authored-By: Claude Code <noreply@anthropic.com>"

echo ""
echo "=== Done! ==="
echo ""
echo "InferenceX is at: third_party/InferenceX/"
echo ""
echo "Benchmark scripts: third_party/InferenceX/benchmarks/single_node/speedbench/"
echo "Runner scripts:    third_party/InferenceX/runners/"
echo "CI workflows:      third_party/InferenceX/.github/workflows/"
echo ""
echo "The infx eval runner will:"
echo "  1. Generate benchmark configs"
echo "  2. Push them to a feature branch (NOT main)"
echo "  3. Trigger workflow_dispatch on that branch"
echo "  4. Poll for completion and download artifacts"
