#!/bin/bash

export PYTORCH_ALLOW_UNSAFE_LOAD=1

# --- Configuration ---
ALGO="sac"
ENV="Hopper-v5"
GT_SEEDS=(1789 1687123 12 912391 330)

# Directories
REWARD_MODEL_DIR="reward_models"
SINGLE_AGENT_DIR="single_feedback_agents"
mkdir -p $SINGLE_AGENT_DIR

# The six feedback types
FEEDBACK_TYPES=(
    "evaluative" "comparative" "demonstrative"
    "corrective" "descriptive" "descriptive_preference"
)

echo "Starting STAGE 4: Single-Type RL Agent Training (6 Types x 5 Seeds = 30 Agents)"
echo "--------------------------------------------------------"

for SEED in "${GT_SEEDS[@]}"; do
    echo "-> Starting Single-Type Agent Training for Seed: $SEED"

    for F_TYPE in "${FEEDBACK_TYPES[@]}"; do

        # --- Skip logic for seed 1789 ---
        if [ "$SEED" -eq 1789 ] && { [ "$F_TYPE" = "evaluative" ] || [ "$F_TYPE" = "comparative" ]; }; then
            echo "   -> Skipping $F_TYPE for SEED=1789"
            continue
        fi

        echo "   -> Training Agent using $F_TYPE Reward Model"

        # --- FINAL MINIMAL COMMAND ---
        python multi_type_feedback/train_RL_agent.py \
            --algorithm "$ALGO" \
            --environment "$ENV" \
            --seed "$SEED" \
            --save-folder "$SINGLE_AGENT_DIR" \
            --reward-model-folder "$REWARD_MODEL_DIR" \
            --feedback-type "$F_TYPE"

        if [ $? -ne 0 ]; then
            echo "❌ Single Agent Training FAILED for $F_TYPE (Seed $SEED). Exiting."
            exit 1
        fi
    done

    echo "✅ Completed applicable feedback types for Seed $SEED."
    echo "--------------------------------------------------------"
done
