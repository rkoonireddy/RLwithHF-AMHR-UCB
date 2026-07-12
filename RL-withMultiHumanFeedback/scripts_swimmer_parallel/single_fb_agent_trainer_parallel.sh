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
SINGLE_AGENT_DIR="single_feedback_agents_swimmer"
mkdir -p "$SINGLE_AGENT_DIR"

# The six feedback types
FEEDBACK_TYPES=(
    "evaluative" "comparative" "demonstrative"
    "corrective" "descriptive" "descriptive_preference"
)

echo "STAGE 4: PARALLEL Single-Type RL Agent Training (6 Types x 10 Seeds = 60 Agents)"
echo "--------------------------------------------------------"

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_swimmer_parallel/git_commit.sh"


PIDS=() 
EXIT_STATUS=0 

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

        # --- EXECUTION COMMAND (Run in background: &) ---
        python multi_type_feedback/train_RL_agent.py \
            --algorithm "$ALGO" \
            --environment "$ENV" \
            --seed "$SEED" \
            --save-folder "$SINGLE_AGENT_DIR" \
            --reward-model-folder "$REWARD_MODEL_DIR" \
            --feedback-type "$F_TYPE" &
        
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
                        echo "❌ Agent Training FAILED (F_TYPE: $F_TYPE, Seed: $SEED, PID: $PID) with exit code $CURRENT_EXIT_CODE. Stopping overall execution."
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
echo "All agents have been queued. Waiting for the final ${#PIDS[@]} job(s) to complete."

for PID in "${PIDS[@]}"; do
    wait "$PID"
    CURRENT_EXIT_CODE=$?
    
    if [ $CURRENT_EXIT_CODE -ne 0 ]; then
        echo "❌ A final agent job with PID $PID FAILED with exit code $CURRENT_EXIT_CODE."
        EXIT_STATUS=1
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Pipeline STAGE 4 (Single-Type Agent Training) completed successfully."
else
    echo "❌ Pipeline STAGE 4 (Single-Type Agent Training) FAILED. Exiting."
    exit 1
fi