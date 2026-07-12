#!/bin/bash

# --- Configuration ---
export PYTORCH_ALLOW_UNSAFE_LOAD=1
MAX_PARALLEL_JOBS=2

ALGO="ppo"
ENV="Swimmer-v5" 

# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 

# Directories (UPDATED for Swimmer)
REWARD_MODEL_DIR="reward_models_swimmer" 
ENSEMBLE_AGENTS_DIR="ensemble_agents_swimmer" 
ALL_TYPES="evaluative comparative demonstrative corrective descriptive descriptive_preference"
TRAIN_STEPS=1000000 

mkdir -p "$ENSEMBLE_AGENTS_DIR"

echo "STAGE 5: PARALLEL Ensemble Policy Training for $ENV (10 Seeds)"
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_swimmer_parallel/git_commit.sh"

PIDS=() 
EXIT_STATUS=0 

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
                    echo "❌ Ensemble Training FAILED (Seed: $SEED, PID: $PID) with exit code $CURRENT_EXIT_CODE. Stopping overall execution."
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
        echo "❌ A final ensemble job with PID $PID FAILED with exit code $CURRENT_EXIT_CODE."
        EXIT_STATUS=1
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Pipeline STAGE 5 (Ensemble Training) completed successfully."
else
    echo "❌ Pipeline STAGE 5 (Ensemble Training) FAILED. Exiting."
    exit 1
fi