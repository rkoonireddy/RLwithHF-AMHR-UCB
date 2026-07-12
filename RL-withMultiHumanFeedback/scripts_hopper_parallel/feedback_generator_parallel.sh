#!/bin/bash

# --- Configuration (Use the same paths and seeds as Stage 1) ---
# NOTE: Define the maximum number of simultaneous jobs to run.
MAX_PARALLEL_JOBS=2

ALGO="sac"
ENV="Hopper-v5"
# Your unique set of seeds
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 

# Directories (ensure these match the output from Stage 1)
GT_AGENTS_DIR="gt_agents/"
FEEDBACK_DIR="feedback"
REWARD_MODEL_DIR="reward_models" # NOTE: Still creating this, though not used in this stage
SAMPLES_DIR="samples" # Required for Demonstrative Feedback generation

# --- Setup ---
mkdir -p "$FEEDBACK_DIR" "$REWARD_MODEL_DIR"

echo "Starting PARALLEL STAGE 2: Feedback Generation (Per Seed)"
echo "Total Seeds to Run: ${#GT_SEEDS[@]}"
echo "Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_hopper_parallel/git_commit.sh"

PIDS=() # Array to store PIDs of background jobs
EXIT_STATUS=0 # Variable to track if any job failed

# ==============================================================================
# STAGE 2: GENERATE ALL SIX FEEDBACK TYPES (PER SEED)
# ==============================================================================
for SEED in "${GT_SEEDS[@]}"; do
    echo "Queuing feedback generation for Seed: $SEED..."
    
    # --- Full Generation Command (Run in background: &) ---
    # This command generates all six feedback types into the $FEEDBACK_DIR folder.
    python multi_type_feedback/generate_feedback.py \
        --algorithm "$ALGO" \
        --environment "$ENV" \
        --seed "$SEED" \
        --n-feedback 10000 \
        --save-folder "$FEEDBACK_DIR" \
        --expert-model-base-path "$GT_AGENTS_DIR" &
    
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
            
            # Check if the process has finished (kill -0 fails if process is done)
            if ! kill -0 "$PID" 2>/dev/null; then
                wait "$PID"
                CURRENT_EXIT_CODE=$?
                
                if [ $CURRENT_EXIT_CODE -ne 0 ]; then
                    echo "❌ Job $PID (Seed $SEED) FAILED with exit code $CURRENT_EXIT_CODE. Stopping all parallel execution."
                    EXIT_STATUS=1 # Mark the overall status as failed
                    # Kill remaining jobs before exiting
                    kill "${PIDS[@]}" 2>/dev/null
                    exit 1
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

# Wait for all remaining background processes to finish
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
    echo "✅ Pipeline STAGE 2 (Feedback Generation) completed successfully for all seeds."
else
    echo "❌ Pipeline STAGE 2 (Feedback Generation) FAILED. Please check the logs."
    exit 1
fi