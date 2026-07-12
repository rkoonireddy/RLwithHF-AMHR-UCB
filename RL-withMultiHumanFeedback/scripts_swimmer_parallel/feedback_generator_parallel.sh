#!/bin/bash

# --- Configuration ---
MAX_PARALLEL_JOBS=2

ALGO="ppo"
ENV="Swimmer-v5"
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 

# Directories
GT_AGENTS_DIR="gt_agents/"
FEEDBACK_DIR="feedback_swimmer"
REWARD_MODEL_DIR="reward_models_swimmer" 

# --- Setup ---
mkdir -p "$FEEDBACK_DIR" "$REWARD_MODEL_DIR"

echo "Starting PARALLEL STAGE 2: Feedback Generation for $ENV"
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_swimmer_parallel/git_commit.sh"

PIDS=() # Array to store PIDs of background jobs
EXIT_STATUS=0

# ==============================================================================
# STAGE 2: GENERATE ALL SIX FEEDBACK TYPES (PER SEED - IN PARALLEL)
# ==============================================================================
for SEED in "${GT_SEEDS[@]}"; do
    echo "Queuing feedback generation for Seed: $SEED..."
    
    # Run the generation script in the background
    python multi_type_feedback/generate_feedback.py \
        --algorithm "$ALGO" \
        --environment "$ENV" \
        --seed "$SEED" \
        --n-feedback 10000 \
        --save-folder "$FEEDBACK_DIR" \
        --expert-model-base-path "$GT_AGENTS_DIR" &
    
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
                    echo "❌ Job $PID (Seed $SEED) FAILED with exit code $CURRENT_EXIT_CODE. Stopping all execution."
                    EXIT_STATUS=1
                    kill "${PIDS[@]}" 2>/dev/null
                    exit 1
                fi
                
                # ------------------------------------------------------------
                echo "--- Job with PID $PID completed successfully. Committing results. ---"
                # Call the dedicated Git script
                bash "$GIT_SCRIPT_PATH"
                echo "------------------------------------------------------------"

                unset 'PIDS[i]'
                PIDS=("${PIDS[@]}")
                break
            fi
        done
    fi
done

# --- Final Cleanup and Wait ---
echo "--------------------------------------------------------"
echo "Waiting for the final ${#PIDS[@]} job(s) to complete."

for PID in "${PIDS[@]}"; do
    wait "$PID"
    CURRENT_EXIT_CODE=$?
    if [ $CURRENT_EXIT_CODE -ne 0 ]; then
        echo "❌ A final job with PID $PID FAILED with exit code $CURRENT_EXIT_CODE."
        EXIT_STATUS=1
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Pipeline STAGE 2 (Feedback Generation) completed successfully for all seeds."
else
    echo "❌ Pipeline STAGE 2 (Feedback Generation) FAILED. Exiting."
    exit 1
fi
