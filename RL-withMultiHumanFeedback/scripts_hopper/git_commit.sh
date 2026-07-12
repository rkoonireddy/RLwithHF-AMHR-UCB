#!/bin/bash
# Exit immediately if any command fails
set -e

# --- CONFIGURATION ---
REPO_PATH="/home/user/app" 
# ---------------------

# 1. Change directory (CRITICAL: Must run inside the repository root)
cd "$REPO_PATH" || { echo "FATAL: Could not change directory to $REPO_PATH. Aborting Git operation." >&2; exit 1; }

# 2. Add the changes (Stages all new/modified files)
git add .

# 3. Commit the changes
COMMIT_MESSAGE="Pipeline: Hopper Job Commit - $(date +"%Y-%m-%d %H:%M:%S")"
# Allow commit to fail if there are no changes, but proceed to pull/push
git commit -m "$COMMIT_MESSAGE" || echo "WARNING: No new changes to commit. Proceeding to sync."

# 4. Sync with Remote (Pulls and replays local commit on top of remote changes)
# This step is the key to preventing "rejected" errors.
# git pull --rebase $REMOTE $BRANCH
# echo "--> Synced with remote."

# 5. Push the changes
if git push; then
    echo "SUCCESS: Pushed commit to $REMOTE/$BRANCH."
else
    echo "ERROR: Git push failed. Please check your SSH keys or conflict status." >&2
    exit 1
fi