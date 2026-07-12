#!/bin/bash

# --- Configuration ---
export PYTORCH_ALLOW_UNSAFE_LOAD=1

# NOTE: Set this to the number of available CPUs.
MAX_PARALLEL_JOBS=2 

ALGO="sac"
ENV="Hopper-v5"

# Using the full 10 seeds as specified in the context, but the script only shows 5.
# I will use the 5 seeds shown in the script for the loop logic.
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 

# Directories
REWARD_MODEL_DIR="reward_models"
SINGLE_AGENT_DIR="single_feedback_agents"
mkdir -p "$SINGLE_AGENT_DIR"

# The six feedback types
FEEDBACK_TYPES=(
    "evaluative" "comparative" "demonstrative"
    "corrective" "descriptive" "descriptive_preference"
)

echo "Starting STAGE 4: PARALLEL Single-Type RL Agent Training (6 Types x ${#GT_SEEDS[@]} Seeds)"
echo "Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_hopper_parallel/git_commit.sh"


PIDS=() # Array to store PIDs of background jobs
EXIT_STATUS=0 # Variable to track if any job failed

# ==============================================================================
# STAGE 4: PARALLELIZE TRAINING OF EACH AGENT (F_TYPE x SEED)
# ==============================================================================
for SEED in "${GT_SEEDS[@]}"; do
    
    for F_TYPE in "${FEEDBACK_TYPES[@]}"; do

        # --- Skip logic for seed 1789 (from original script) ---
        if [ "$SEED" -eq 1789 ] && { [ "$F_TYPE" = "evaluative" ] || [ "$F_TYPE" = "comparative" ]; }; then
            echo "-> Skipping $F_TYPE for SEED=1789"
            continue
        fi

        echo "Queuing Agent: $F_TYPE (Seed $SEED)"

        # --- FINAL MINIMAL COMMAND (Run in background: &) ---
        python multi_type_feedback/train_RL_agent.py \
            --algorithm "$ALGO" \
            --environment "$ENV" \
            --seed "$SEED" \
            --save-folder "$SINGLE_AGENT_DIR" \
            --reward-model-folder "$REWARD_MODEL_DIR" \
            --feedback-type "$F_TYPE" &
        
        # Store the Process ID (PID) of the job that was just launched
        PIDS+=($!)

        # Check if we've reached the limit of parallel jobs
        if [ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; then
            echo "--------------------------------------------------------"
            echo "Maximum parallel jobs reached ($MAX_PARALLEL_JOBS). Waiting for a job to finish..."
            
            # Wait for ANY currently running background job to finish
            wait -n
            
            # Cleanup loop to find the finished job and check its status
            for i in "${!PIDS[@]}"; do
                PID=${PIDS[$i]}
                
                # Check if the process has finished
                if ! kill -0 "$PID" 2>/dev/null; then
                    wait "$PID"
                    CURRENT_EXIT_CODE=$?
                    
                    if [ $CURRENT_EXIT_CODE -ne 0 ]; then
                        echo "❌ Agent Training FAILED (F_TYPE: $F_TYPE, Seed: $SEED, PID: $PID) with exit code $CURRENT_EXIT_CODE."
                        EXIT_STATUS=1 # Mark the overall status as failed
                        # Kill all other background processes and exit the script
                        kill "${PIDS[@]}" 2>/dev/null
                        exit 1
                    fi
                    
                    echo "--- Job with PID $PID completed successfully. Committing results. ---"
                    # Call the dedicated Git script
                    bash "$GIT_SCRIPT_PATH"
                    echo "------------------------------------------------------------"
                    
                    # Remove the finished PID from the array
                    unset 'PIDS[i]'
                    PIDS=("${PIDS[@]}") # Re-index the array
                    break # Exit inner loop, continue to queue the next job
                fi
            done
            echo "Continuing with next job..."
        fi
    done
done

# --- Final Cleanup and Wait ---
echo "--------------------------------------------------------"
echo "All agents have been queued. Waiting for the final ${#PIDS[@]} job(s) to complete."

# Wait for all remaining background processes to finish
for PID in "${PIDS[@]}"; do
    wait "$PID"
    CURRENT_EXIT_CODE=$?
    
    if [ $CURRENT_EXIT_CODE -ne 0 ]; then
        echo "❌ A final agent job with PID $PID FAILED with exit code $CURRENT_EXIT_CODE."
        EXIT_STATUS=1
    else
        echo "✅ Job $PID completed successfully."
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Pipeline STAGE 4 (Single-Type Agent Training) completed successfully."
else
    echo "❌ Pipeline STAGE 4 (Single-Type Agent Training) FAILED. Please check the logs."
    exit 1
fi