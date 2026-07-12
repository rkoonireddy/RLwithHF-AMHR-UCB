#!/bin/bash

# --- Configuration ---
# NOTE: Define the maximum number of simultaneous Python jobs to run.
MAX_PARALLEL_JOBS=2 

ALGO="sac"
ENV="Hopper-v5"
N_SAMPLES=10000

# Your unique set of seeds
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 

SAMPLES_DIR="samples"

# Create the samples directory if it doesn't exist
mkdir -p "$SAMPLES_DIR"

echo "Starting Parallel random sample generation for Demonstrative Feedback."
echo "Total Samples per run: $N_SAMPLES"
echo "Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "Total Seeds to Run: ${#GT_SEEDS[@]}"
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_hopper_parallel/git_commit.sh"


PIDS=() # Array to store PIDs of background jobs
EXIT_STATUS=0 # Variable to track if any job failed

# Loop through each seed and manage parallelism
for SEED in "${GT_SEEDS[@]}"; do
    echo "Queuing sample generation for Seed: $SEED..."
    
    # --- Full Generation Command (Run in background: &) ---
    python multi_type_feedback/generate_samples.py \
        --algorithm "$ALGO" \
        --environment "$ENV" \
        --n-samples "$N_SAMPLES" \
        --random \
        --seed "$SEED" &
    
    # Store the Process ID (PID) of the job that was just launched
    PIDS+=($!)
    
    # Check if we've reached the limit of parallel jobs
    if [ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; then
        echo "Maximum parallel jobs reached ($MAX_PARALLEL_JOBS). Waiting for a job to finish..."
        
        # Wait for ANY currently running background job to finish
        wait -n
        
        # Re-evaluate the PIDS array to find the finished job and check its status
        for i in "${!PIDS[@]}"; do
            PID=${PIDS[$i]}
            
            # Use 'kill -0' to check if the process is still running. If it fails, the process finished.
            if ! kill -0 "$PID" 2>/dev/null; then
                # Process has finished. Wait for it to get its exit code.
                wait "$PID"
                CURRENT_EXIT_CODE=$?
                
                if [ $CURRENT_EXIT_CODE -ne 0 ]; then
                    echo "❌ A background job with PID $PID FAILED (Exit Code: $CURRENT_EXIT_CODE). Continuing execution but tracking failure."
                    EXIT_STATUS=1 # Mark the overall status as failed
                else
                    echo "✅ Job $PID (Seed $SEED) completed successfully."
                fi
                
                echo "--- Job with PID $PID completed successfully. Committing results. ---"
                # Call the dedicated Git script
                bash "$GIT_SCRIPT_PATH"
                echo "------------------------------------------------------------"

                # Remove the finished PID from the array
                unset 'PIDS[i]'
                PIDS=("${PIDS[@]}") # Re-index the array
                break # Exit inner loop, continue to queue the next seed
            fi
        done
        echo "--------------------------------------------------------"
    fi
done

# --- Final Cleanup and Wait ---
echo "--------------------------------------------------------"
echo "All seeds have been queued. Waiting for the final ${#PIDS[@]} job(s) to complete."

# Wait for all remaining background processes to finish and capture their final status
for PID in "${PIDS[@]}"; do
    wait "$PID"
    CURRENT_EXIT_CODE=$?
    
    if [ $CURRENT_EXIT_CODE -ne 0 ]; then
        echo "❌ A final background job with PID $PID FAILED with exit code $CURRENT_EXIT_CODE."
        EXIT_STATUS=1
    else
        echo "✅ Job $PID completed successfully."
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ All random sample generation runs completed successfully."
    echo "You can now proceed with Feedback Generation (Stage 2)."
else
    echo "❌ One or more sample generation runs FAILED. Please check the logs."
    exit 1
fi