#!/bin/bash

# --- Configuration ---
export PYTORCH_ALLOW_UNSAFE_LOAD=1
MAX_PARALLEL_JOBS=2

ALGO="ppo"
ENV="Swimmer-v5" 

# Full set of 10 seeds
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 

# Directories (UPDATED for Swimmer)
REWARD_MODEL_DIR="reward_models_swimmer" 
ENSEMBLE_AGENTS_DIR="ensemble_agents_combos_swimmer" 

TRAIN_STEPS=1000000 
mkdir -p "$ENSEMBLE_AGENTS_DIR"

# --- DEFINE FEEDBACK COMBINATIONS ---
# 8 total combinations
COMBOS_2=(
    "evaluative comparative"
    "demonstrative corrective"
    "descriptive descriptive_preference"
    "evaluative descriptive"
)

COMBOS_3=(
    "evaluative comparative demonstrative"
    "corrective descriptive descriptive_preference"
    "evaluative corrective descriptive_preference"
)

COMBOS_4=(
    "evaluative comparative demonstrative corrective"
    "demonstrative corrective descriptive descriptive_preference"
)

ALL_COMBINATIONS=("${COMBOS_2[@]}" "${COMBOS_3[@]}" "${COMBOS_4[@]}")

echo "STAGE 6: PARALLEL Combinatorial Ensemble Training for $ENV"
echo "Running ${#ALL_COMBINATIONS[@]} Combos x ${#GT_SEEDS[@]} Seeds = $(( ${#ALL_COMBINATIONS[@]} * ${#GT_SEEDS[@]} )) Total Jobs."
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_swimmer_parallel/git_commit.sh"


PIDS=() 
EXIT_STATUS=0 

# ==============================================================================
# PARALLELIZE TRAINING OF EACH COMBO-SEED PAIR
# ==============================================================================
for FEEDBACK_TYPES in "${ALL_COMBINATIONS[@]}"; do
    
    COMBINATION_ID=$(echo "$FEEDBACK_TYPES" | tr ' ' '_')
    NUM_TYPES=$(echo "$FEEDBACK_TYPES" | wc -w)
    
    echo "🚀 Queuing $NUM_TYPES-Feedback Ensemble: $COMBINATION_ID"
    
    for SEED in "${GT_SEEDS[@]}"; do
        
        echo "-> Queuing combo '$COMBINATION_ID' with Seed: $SEED"
        
        # --- EXECUTION COMMAND (Run in background: &) ---
        python multi_type_feedback/train_RL_agent_with_ensemble.py \
            --algorithm "$ALGO" \
            --environment "$ENV" \
            --train-steps "$TRAIN_STEPS" \
            --seed "$SEED" \
            --save-folder "$ENSEMBLE_AGENTS_DIR" \
            --reward-model-folder "$REWARD_MODEL_DIR" \
            --feedback-types $FEEDBACK_TYPES &
        
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
                        echo "❌ Job FAILED (Combo: $COMBINATION_ID, Seed: $SEED, PID: $PID) with exit code $CURRENT_EXIT_CODE. Stopping overall execution."
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
done

# --- Final Cleanup and Wait ---
echo "--------------------------------------------------------"
echo "All jobs have been queued. Waiting for the final ${#PIDS[@]} job(s) to complete."

for PID in "${PIDS[@]}"; do
    wait "$PID"
    CURRENT_EXIT_CODE=$?
    
    if [ $CURRENT_EXIT_CODE -ne 0 ]; then
        echo "❌ A final job with PID $PID FAILED with exit code $CURRENT_EXIT_CODE."
        EXIT_STATUS=1
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ STAGE 6 (Combinatorial Ensemble Training) completed successfully."
else
    echo "❌ STAGE 6 (Combinatorial Ensemble Training) FAILED. Exiting."
    exit 1
fi