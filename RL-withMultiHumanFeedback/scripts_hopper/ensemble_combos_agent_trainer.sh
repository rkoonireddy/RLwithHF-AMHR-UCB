#!/bin/bash

# --- Configuration (Keep consistent) ---
ALGO="sac"
ENV="Hopper-v5"
GT_SEEDS=(1789 1687123 12 912391 330)
REWARD_MODEL_DIR="reward_models" 
ENSEMBLE_AGENTS_DIR="ensemble_agents_combos" 

# Set training steps directly in the script for reliability
TRAIN_STEPS=1000000 

echo "Starting STAGE 4: Combinatorial Ensemble Policy Training"
echo "--------------------------------------------------------"

# --- DEFINE FEEDBACK COMBINATIONS ---
# NOTE: The final agent directory name (MODEL_ID) will automatically reflect the feedback types used.
# You can customize these lists, but ensure the names match your reward model filenames.

# Combinations of 2 feedbacks
COMBOS_2=(
    "evaluative comparative"
    "demonstrative corrective"
    "descriptive descriptive_preference"
    "evaluative descriptive" # Example of combining different types
)

# Combinations of 3 feedbacks
COMBOS_3=(
    "evaluative comparative demonstrative"
    "corrective descriptive descriptive_preference"
    "evaluative corrective descriptive_preference"
)

# Combinations of 4 feedbacks
COMBOS_4=(
    "evaluative comparative demonstrative corrective"
    "demonstrative corrective descriptive descriptive_preference"
)

# Combine all combination sets into one list for the outer loop
ALL_COMBINATIONS=("${COMBOS_2[@]}" "${COMBOS_3[@]}" "${COMBOS_4[@]}")
# ------------------------------------

# --- Nested Loops for Training ---

# 1. Loop through each combination of feedback types
for FEEDBACK_TYPES in "${ALL_COMBINATIONS[@]}"; do
    # Create a unique, file-safe ID for the combination
    COMBINATION_ID=$(echo "$FEEDBACK_TYPES" | tr ' ' '_')
    NUM_TYPES=$(echo "$FEEDBACK_TYPES" | wc -w)
    
    echo "========================================================"
    echo "🚀 Starting Training for $NUM_TYPES-Feedback Ensemble: $COMBINATION_ID"
    echo "========================================================"

    # 2. Loop through each Seed
    for SEED in "${GT_SEEDS[@]}"; do
        
        echo "-> Running Combination '$COMBINATION_ID' with Seed: $SEED"
        
        # Execute the ensemble training script
        python multi_type_feedback/train_RL_agent_with_ensemble.py \
            --algorithm "$ALGO" \
            --environment "$ENV" \
            --train-steps "$TRAIN_STEPS" \
            --seed "$SEED" \
            --save-folder "$ENSEMBLE_AGENTS_DIR" \
            --reward-model-folder "$REWARD_MODEL_DIR" \
            --feedback-types $FEEDBACK_TYPES
        
        # Check the exit status of the Python script
        if [ $? -ne 0 ]; then
            echo "❌ Ensemble Training FAILED for Combination '$COMBINATION_ID' and Seed $SEED."
            # Decide whether to exit the entire script or just skip this seed/combo
            exit 1 
        fi
        
        echo "✅ Agent trained for Combo '$COMBINATION_ID' and Seed $SEED."
    done
    echo "--------------------------------------------------------"
done

echo "STAGE 4 Completed. All combinatorial ensemble agents are saved in '$ENSEMBLE_AGENTS_DIR'."