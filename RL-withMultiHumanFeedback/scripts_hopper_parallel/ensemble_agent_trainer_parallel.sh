#!/bin/bash

# --- Configuration ---
export PYTORCH_ALLOW_UNSAFE_LOAD=1
# NOTE: Set this to the number of available CPUs.
MAX_PARALLEL_JOBS=2 

ALGO="sac"
ENV="Hopper-v5"
# Using the full 10 seeds
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 


REWARD_MODEL_DIR="reward_models" 
ENSEMBLE_AGENTS_DIR="ensemble_agents" 
ALL_TYPES="evaluative comparative demonstrative corrective descriptive descriptive_preference"
TRAIN_STEPS=1000000 

mkdir -p "$ENSEMBLE_AGENTS_DIR"

echo "Starting STAGE 5 (Ensemble Policy Training) - Running ${#GT_SEEDS[@]} Seeds in Parallel"
echo "Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "Total Training Steps per Agent: $TRAIN_STEPS"
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_hopper_parallel/git_commit.sh"

PIDS=() # Array to store PIDs of background jobs
EXIT_STATUS=0 # Variable to track if any job failed

# ==============================================================================
# STAGE 5: PARALLELIZE TRAINING OF THE ENSEMBLE POLICY BY SEED
# ==============================================================================
for SEED in "${GT_SEEDS[@]}"; do
    echo "Queuing Ensemble Training for Seed: $SEED..."
    
    # --- Full Training Command (Run in background: &) ---
    python multi_type_feedback/train_RL_agent_with_ensemble.py \
        --algorithm "$ALGO" \
        --environment "$ENV" \
        --train-steps "$TRAIN_STEPS" \
        --seed "$SEED" \
        --save-folder "$ENSEMBLE_AGENTS_DIR" \
        --reward-model-folder "$REWARD_MODEL_DIR" \
        --feedback-types $ALL_TYPES &
    
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
                    echo "❌ Ensemble Training FAILED (Seed: $SEED, PID: $PID) with exit code $CURRENT_EXIT_CODE. Stopping overall execution."
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
        echo "Continuing with next seed..."
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
        echo "❌ A final ensemble job with PID $PID FAILED with exit code $CURRENT_EXIT_CODE."
        EXIT_STATUS=1
    else
        echo "✅ Job $PID (Seed $SEED) completed successfully."
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Pipeline STAGE 5 (Ensemble Training) completed successfully for all seeds."
    echo "All final ensemble agents are saved in '$ENSEMBLE_AGENTS_DIR'."
else
    echo "❌ Pipeline STAGE 5 (Ensemble Training) FAILED. Please check the logs."
    exit 1
fi