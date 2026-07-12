#!/bin/bash

# --- Configuration ---
MAX_PARALLEL_JOBS=2 

ALGO="ppo"
ENV="Swimmer-v5"

# Full set of 10 seeds
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 

# Directories (UPDATED for Swimmer)
FEEDBACK_DIR="feedback_swimmer"
REWARD_MODEL_DIR="reward_models_swimmer"

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

echo "STAGE 3: PARALLEL Reward Model Training for $ENV (6 Models/Seed x 10 Seeds = 60 Jobs Total)"
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_swimmer_parallel/git_commit.sh"

PIDS=() 
EXIT_STATUS=0 

# ==============================================================================
# STAGE 3: PARALLEL BY SEED (Inner loop remains sequential for the 6 models)
# ==============================================================================
for SEED in "${GT_SEEDS[@]}"; do
    echo "Queuing Reward Model Training for Seed: $SEED..."
    
    # Run the inner sequential loop in the background using a subshell
    (
        echo "-> Starting sequential training of 6 models for Seed: $SEED"
        
        for F_TYPE in "${FEEDBACK_TYPES[@]}"; do
            echo "   -> Training: $F_TYPE (Seed $SEED)"
            
            python multi_type_feedback/train_reward_model.py \
                --algorithm "$ALGO" \
                --environment "$ENV" \
                --feedback-type "$F_TYPE" \
                --seed "$SEED" \
                --feedback-folder "$FEEDBACK_DIR" \
                --save-folder "$REWARD_MODEL_DIR"
            
            if [ $? -ne 0 ]; then
                echo "❌ Reward Model Training FAILED for $F_TYPE (Seed $SEED). Terminating this seed's run."
                exit 1 
            fi
        done
        echo "✅ All 6 reward models trained successfully for Seed $SEED."
    ) & # Launch the entire sequential block for this seed into the background
    
    PIDS+=($!)
    
    if [ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; then
        echo "Maximum parallel jobs reached ($MAX_PARALLEL_JOBS). Waiting for a seed run to finish..."
        wait -n
        
        for i in "${!PIDS[@]}"; do
            PID=${PIDS[$i]}
            if ! kill -0 "$PID" 2>/dev/null; then
                wait "$PID"
                CURRENT_EXIT_CODE=$?
                
                if [ $CURRENT_EXIT_CODE -ne 0 ]; then
                    echo "❌ Seed Run (PID $PID) FAILED with exit code $CURRENT_EXIT_CODE. Stopping overall execution."
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
echo "Waiting for the final ${#PIDS[@]} seed run(s) to complete."

for PID in "${PIDS[@]}"; do
    wait "$PID"
    CURRENT_EXIT_CODE=$?
    
    if [ $CURRENT_EXIT_CODE -ne 0 ]; then
        echo "❌ A final seed run with PID $PID FAILED with exit code $CURRENT_EXIT_CODE."
        EXIT_STATUS=1
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ STAGE 3 (Reward Model Training) completed successfully."
else
    echo "❌ STAGE 3 (Reward Model Training) FAILED. Exiting."
    exit 1
fi