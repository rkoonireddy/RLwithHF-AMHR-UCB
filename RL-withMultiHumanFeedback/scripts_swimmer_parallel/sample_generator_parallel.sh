#!/bin/bash

# --- Configuration ---
MAX_PARALLEL_JOBS=2 

ALGO="ppo" # Using PPO for consistency with expert training
ENV="Swimmer-v5" 
N_SAMPLES=10000

# Full set of 10 seeds
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 
SAMPLES_DIR="samples" 

# Create the samples directory if it doesn't exist
mkdir -p "$SAMPLES_DIR"

echo "STAGE 1: PARALLEL Random Sample Generation for $ENV"
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_swimmer_parallel/git_commit.sh"


PIDS=() 
EXIT_STATUS=0 

# Loop through each seed and manage parallelism
for SEED in "${GT_SEEDS[@]}"; do
    echo "Queuing sample generation for Seed: $SEED..."
    
    # --- Full Generation Command (Run in background: &) ---
    python multi_type_feedback/generate_samples.py \
        --algorithm "$ALGO" \
        --environment "$ENV" \
        --n-samples "$N_SAMPLES" \
        --random \
        --seed "$SEED" \
        --save-folder "$SAMPLES_DIR" &
    
    PIDS+=($!)
    
    if [ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; then
        echo "Maximum parallel jobs reached ($MAX_PARALLEL_JOBS). Waiting for a job to finish..."
        
        wait -n
        
        for i in "${!PIDS[@]}"; do
            PID=${PIDS[$i]}
            if ! kill -0 "$PID" 2>/dev/null; then
                wait "$PID"
                CURRENT_EXIT_CODE=$?
                
                if [ $CURRENT_EXIT_CODE -ne 0 ]; then
                    echo "❌ Sample Generation FAILED for PID $PID (Exit Code: $CURRENT_EXIT_CODE). Stopping execution."
                    EXIT_STATUS=1
                    kill "${PIDS[@]}" 2>/dev/null
                    exit 1
                fi
                echo "--- Job with PID $PID completed successfully. Committing results. ---"
                # Call the dedicated Git script
                bash "$GIT_SCRIPT_PATH"
                echo "------------------------------------------------------------"

                unset 'PIDS[i]'
                PIDS=("${PIDS[@]}")
                break
            fi
        done
        echo "--------------------------------------------------------"
    fi
done

# --- Final Cleanup and Wait ---
echo "All seeds have been queued. Waiting for the final ${#PIDS[@]} job(s) to complete."

for PID in "${PIDS[@]}"; do
    wait "$PID"
    CURRENT_EXIT_CODE=$?
    
    if [ $CURRENT_EXIT_CODE -ne 0 ]; then
        echo "❌ A final job with PID $PID FAILED with exit code $CURRENT_EXIT_CODE."
        EXIT_STATUS=1
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ STAGE 1 (Sample Generation) completed successfully."
else
    echo "❌ STAGE 1 (Sample Generation) FAILED. Exiting."
    exit 1
fi