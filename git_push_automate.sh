#!/bin/sh

# --- Configuration ---
# Set the path to your repository root (optional, but good practice for cron)
REPO_PATH="/home/user/app"
# Remote and branch to push to (defaults are common)
REMOTE="origin"
BRANCH="main" 
# --- End Configuration ---

cd $REPO_PATH

# 1. Check if there are any changes to commit
# -s suppresses the output if the working tree is clean.
if [ -z "$(git status --porcelain)" ]; then
  # No changes, exit cleanly
  echo "$(date): No changes found. Skipping commit and push."
  exit 0
fi

# 2. Add all changes to the staging area
git add .

# 3. Create a unique commit message with a timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
COMMIT_MESSAGE="Auto-commit: Work in progress - $TIMESTAMP"

git commit -m "$COMMIT_MESSAGE"

# 4. Pull to merge remote changes before pushing
# This prevents failed pushes due to remote updates, 
# assuming no merge conflicts require manual intervention.
git pull --rebase $REMOTE $BRANCH

# 5. Push the changes to the remote repository
git push $REMOTE $BRANCH

echo "$(date): Successfully committed and pushed changes with message: \"$COMMIT_MESSAGE\""