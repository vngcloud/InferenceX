#!/usr/bin/env bash
# Setup SpecForge fork as git submodule.
#
# Prerequisites:
#   1. Fork SpecForge on GitHub: https://github.com/sgl-project/SpecForge → Fork
#   2. Set your GitHub username:  export GITHUB_USER=your-username
#
# Usage:
#   ./scripts/setup_specforge_fork.sh                    # uses $GITHUB_USER
#   ./scripts/setup_specforge_fork.sh my-github-username # explicit username

set -euo pipefail

UPSTREAM_URL="https://github.com/sgl-project/SpecForge.git"
GITHUB_USER="${1:-${GITHUB_USER:?Usage: setup_specforge_fork.sh <github-username>}}"
FORK_URL="https://github.com/${GITHUB_USER}/SpecForge.git"

echo "=== Setting up SpecForge fork ==="
echo "  Upstream: $UPSTREAM_URL"
echo "  Fork:     $FORK_URL"
echo ""

# Remove placeholder
rm -f third_party/README.md
rmdir third_party 2>/dev/null || true

# Add fork as submodule
git submodule add "$FORK_URL" third_party/SpecForge

# Add upstream remote for syncing
cd third_party/SpecForge
git remote add upstream "$UPSTREAM_URL"
cd ../..

# Commit submodule
git add .gitmodules third_party/SpecForge
git commit -m "Add SpecForge fork as submodule

- Fork: ${FORK_URL}
- Upstream: ${UPSTREAM_URL}
- Sync: cd third_party/SpecForge && git fetch upstream && git merge upstream/main

Co-Authored-By: Claude Code <noreply@anthropic.com>"

echo ""
echo "=== Done! ==="
echo ""
echo "SpecForge fork is at: third_party/SpecForge/"
echo ""
echo "To modify training algorithm:"
echo "  cd third_party/SpecForge"
echo "  # edit files..."
echo "  git add -A && git commit -m 'custom training algorithm'"
echo "  git push origin main"
echo ""
echo "To sync upstream changes:"
echo "  cd third_party/SpecForge"
echo "  git fetch upstream"
echo "  git merge upstream/main"
echo "  # resolve conflicts, test..."
echo "  git push origin main"
echo "  cd ../.."
echo "  git add third_party/SpecForge"
echo "  git commit -m 'Update SpecForge fork: sync upstream'"
