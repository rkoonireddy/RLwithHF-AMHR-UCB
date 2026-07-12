#!/bin/bash

# --- Configuration ---
# NOTE: Set this to the number of available CPUs (8 in your case).
MAX_PARALLEL_JOBS=2

ALGO="sac"
ENV="Hopper-v5"
# Updated list of 10 unique seeds
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 

# Directories (ensure these match the output from Stage 2)
FEEDBACK_DIR="feedback"
REWARD_MODEL_DIR="reward_models"

# Feedback Types (6 types to be trained sequentially per seed)
FEEDBACK_TYPES=(
    "evaluative" 
    "comparative" 
    "demonstrative" 
    "corrective" 
    "descriptive" 
    "descriptive_preference"
)

# --- Setup ---
mkdir -p "$REWARD_MODEL_DIR"

echo "--------------------------------------------------------"
echo "STAGE 3: Training Reward Models (6 Models/Seed * 10 Seeds = 60 Jobs Total)"
echo "Parallelism: Running up to $MAX_PARALLEL_JOBS seeds simultaneously."
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_hopper_parallel/git_commit.sh"

PIDS=() # Array to store PIDs of background jobs (one PID per seed run)
EXIT_STATUS=0 # Variable to track if any seed run failed

# ==============================================================================
# STAGE 3: PARALLEL BY SEED (Inner loop remains sequential for the 6 models)
# ==============================================================================
for SEED in "${GT_SEEDS[@]}"; do
    echo "Queuing Reward Model Training for Seed: $SEED..."
    
    # Run the inner sequential loop in the background using a subshell
    (
        echo "-> Starting sequential training of 6 models for Seed: $SEED"
        
        # Loop through all six types for the current seed
        for F_TYPE in "${FEEDBACK_TYPES[@]}"; do
            echo "   -> Training: $F_TYPE (Seed $SEED)"
            
            python multi_type_feedback/train_reward_model.py \
                --algorithm "$ALGO" \
                --environment "$ENV" \
                --feedback-type "$F_TYPE" \
                --seed "$SEED" \
                --feedback-folder "$FEEDBACK_DIR" \
                --save-folder "$REWARD_MODEL_DIR"
            
            # Check exit code immediately after training the model
            if [ $? -ne 0 ]; then
                echo "❌ Reward Model Training FAILED for $F_TYPE (Seed $SEED). Terminating this seed's run."
                exit 1 # Exit the subshell, triggering the failure status
            fi
        done
        echo "✅ All 6 reward models trained successfully for Seed $SEED."
    ) & # Launch the entire sequential block for this seed into the background
    
    # Store the Process ID (PID) of the job that was just launched
    PIDS+=($!)
    
    # Check if we've reached the limit of parallel jobs (CPU limit)
    if [ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; then
        echo "Maximum parallel jobs reached ($MAX_PARALLEL_JOBS). Waiting for a seed run to finish..."
        
        # Wait for ANY currently running background job (seed run) to finish
        wait -n
        
        # Cleanup loop (same logic as before, checking finished PIDs)
        for i in "${!PIDS[@]}"; do
            PID=${PIDS[$i]}
            
            # Check if the process has finished
            if ! kill -0 "$PID" 2>/dev/null; then
                wait "$PID"
                CURRENT_EXIT_CODE=$?
                
                if [ $CURRENT_EXIT_CODE -ne 0 ]; then
                    echo "❌ Seed Run (PID $PID) FAILED with exit code $CURRENT_EXIT_CODE. Stopping overall execution."
                    EXIT_STATUS=1 # Mark the overall status as failed
                    # Kill remaining jobs
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
                break # Continue to queue the next seed
            fi
        done
        echo "Continuing with next seed..."
    fi
done

# --- Final Cleanup and Wait ---
echo "--------------------------------------------------------"
echo "All seeds have been queued. Waiting for the final ${#PIDS[@]} seed run(s) to complete."

# Wait for all remaining background processes to finish
for PID in "${PIDS[@]}"; do
    wait "$PID"
    CURRENT_EXIT_CODE=$?
    
    if [ $CURRENT_EXIT_CODE -ne 0 ]; then
        echo "❌ A final seed run with PID $PID FAILED with exit code $CURRENT_EXIT_CODE."
        EXIT_STATUS=1
    else
        echo "✅ Seed Run (PID $PID) completed successfully."
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Pipeline STAGES 3 (Reward Model Training) completed successfully for all 10 seeds."
else
    echo "❌ Pipeline STAGES 3 (Reward Model Training) FAILED. Please check the logs."
    exit 1
fi