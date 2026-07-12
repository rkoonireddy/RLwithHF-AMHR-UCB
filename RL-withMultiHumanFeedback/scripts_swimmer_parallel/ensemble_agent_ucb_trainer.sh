#!/bin/bash

# --- Configuration for UCB Ensemble Agent Training ---
MAX_PARALLEL_JOBS=2  # Keep parallel control for seeds

ALGO="ppo"
ENV="Swimmer-v5"

# Seeds to run
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330)

# Directories (Consistent with your Stage 3 script)
REWARD_MODEL_DIR="reward_models_swimmer"
ENSEMBLE_AGENTS_DIR="ensemble_agents_swimmer_ucb" # NEW Directory for UCB agents

# All 6 feedback types must be passed to the ensemble script
ALL_TYPES="evaluative comparative demonstrative corrective descriptive descriptive_preference"

# RL Training steps
TRAIN_STEPS=1000000 

# UCB Hyperparameters
UCB_UPDATE_FREQ=10000
UCB_EXPLORATION_CONSTANT=2.0

# --- Setup ---
mkdir -p "$ENSEMBLE_AGENTS_DIR"

echo "STAGE 4: PARALLEL UCB Ensemble RL Agent Training for $ENV"
echo "--------------------------------------------------------"

# Assuming this script path is still relevant for committing results
GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_swimmer_parallel/git_commit.sh"

PIDS=() 
EXIT_STATUS=0 

# ==============================================================================
# STAGE 4: PARALLEL UCB ENSEMBLE TRAINING
# ==============================================================================
for SEED in "${GT_SEEDS[@]}"; do
    echo "Queuing UCB Ensemble Training for Seed: $SEED..."
    
    # Run the single ensemble job in the background using a subshell
    (
        echo "-> Starting UCB Ensemble training for Seed: $SEED"
        
        python multi_type_feedback/train_RL_agent_with_ensemble_roham.py \
            --algorithm "$ALGO" \
            --environment "$ENV" \
            --train-steps "$TRAIN_STEPS" \
            --seed "$SEED" \
            --save-folder "$ENSEMBLE_AGENTS_DIR" \
            --reward-model-folder "$REWARD_MODEL_DIR" \
            --feedback-types $ALL_TYPES \
            --use-ucb \
            --ucb-update-freq $UCB_UPDATE_FREQ \
            --ucb-exploration-constant $UCB_EXPLORATION_CONSTANT
        
        if [ $? -ne 0 ]; then
            echo "❌ UCB Ensemble Training FAILED for Seed $SEED. Terminating this seed's run."
            exit 1 
        fi
        echo "✅ UCB Ensemble agent trained successfully for Seed $SEED."
    ) & # Launch the single ensemble job for this seed into the background
    
    PIDS+=($!)
    
    if [ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; then
        echo "Maximum parallel jobs reached ($MAX_PARALLEL_JOBS). Waiting for a seed run to finish..."
        wait -n
        
        # Logic to check which job finished and handle success/failure
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
    else
        # Git commit for the successful job that was not committed in the main loop
        echo "--- Job with PID $PID completed successfully. Committing results. ---"
        bash "$GIT_SCRIPT_PATH"
        echo "------------------------------------------------------------"
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ STAGE 4 (UCB Ensemble RL Agent Training) completed successfully."
else
    echo "❌ STAGE 4 (UCB Ensemble RL Agent Training) FAILED. Exiting."
    exit 1
fi