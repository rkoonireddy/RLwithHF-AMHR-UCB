#!/bin/bash

# --- Configuration ---
WANDB_DIR="./wandb"
PROJECT_DIR=$(pwd)
# ---------------------

echo "🚀 Starting Selective W&B Sync and Cleanup"
echo "----------------------------------------------------------------------"

# 1. LOG IN TO CLOUD (SAFETY CHECK)
echo "1. Checking W&B Cloud Login..."
wandb login --cloud 

# 2. FIND UNSYNCED OFFLINE RUNS
echo -e "\n2. Finding unsynced offline runs in: ${WANDB_DIR}"

# Find directories matching the offline run pattern (offline-run-*)
# This list contains ALL runs that have not yet been marked "synced"
UNSYNCED_RUNS=$(find "${WANDB_DIR}" -maxdepth 1 -type d -name "offline-run-*" | grep -v "synced")

if [ -z "$UNSYNCED_RUNS" ]; then
    echo "   No unsynced offline runs found."
fi

TOTAL_RUNS=$(echo "$UNSYNCED_RUNS" | wc -l)
CURRENT_RUN_NUM=0
FAILURE_COUNT=0

echo "   Found ${TOTAL_RUNS} runs to attempt syncing."
echo "----------------------------------------------------------------------"

# 3. ATTEMPT TO SYNC EACH RUN INDIVIDUALLY
IFS=$'\n' # Set Internal Field Separator to newline
for run_path in $UNSYNCED_RUNS; do
    CURRENT_RUN_NUM=$((CURRENT_RUN_NUM + 1))
    
    # Extract just the run ID for display
    run_id=$(basename "$run_path")
    
    echo -e "\n[${CURRENT_RUN_NUM}/${TOTAL_RUNS}] Attempting to sync run: ${run_id}"
    
    # Run the sync command. Use "|| true" to prevent the script from exiting on failure (like the 409 error)
    (cd "${PROJECT_DIR}" && wandb sync "${run_path}") || {
        echo -e "\n❌ ERROR: Sync FAILED for run ${run_id}. The run data remains local."
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        sleep 1 
    }
done
unset IFS # Restore IFS

# --- NEW STEP: DELETE OLD RUNS ---
echo -e "\n\n4. Purging Local Runs Older Than Today"
echo "----------------------------------------------------------------------"
# Find directories that are "offline-run-*" and were modified before TODAY
# -type d: Only look for directories
# -maxdepth 1: Only look in the top level of WANDB_DIR
# -daystart: Measure time from the beginning of today (00:00:00)
# -mtime +0: Find directories whose modification time is greater than 0 full days old
OLD_RUNS_TO_DELETE=$(find "${WANDB_DIR}" -maxdepth 1 -type d -name "offline-run-*" -daystart -mtime +0)

if [ -z "$OLD_RUNS_TO_DELETE" ]; then
    echo "   No local W&B run directories older than today found."
else
    echo "🚨 WARNING: Deleting the following old local run directories:"
    echo "$OLD_RUNS_TO_DELETE"
    
    # Delete the found directories (using 'echo rm -rf' as a safe test first!)
    # To ACTUALLY DELETE, remove the 'echo' command below:
    echo "$OLD_RUNS_TO_DELETE" | xargs -r rm -rf

    # The line above is the safe test. To make it live, use:
    # echo "$OLD_RUNS_TO_DELETE" | xargs -r rm -rf
    
    echo "🗑️ Old local runs have been cleaned up."
fi

# 5. FINAL SUMMARY
echo -e "\n----------------------------------------------------------------------"
echo "✅ Selective Sync and Cleanup Complete."
echo "Total Runs Processed (Sync Attempted): ${TOTAL_RUNS}"
echo "Successful Syncs: $((TOTAL_RUNS - FAILURE_COUNT))"
echo "Failed Syncs (These runs were NOT deleted if older): ${FAILURE_COUNT}"
echo "----------------------------------------------------------------------"