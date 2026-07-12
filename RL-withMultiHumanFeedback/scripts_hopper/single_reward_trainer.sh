#!/bin/bash

# --- Configuration (Use the same paths and seeds as Stage 1) ---
ALGO="sac"
ENV="Hopper-v5"
# Your unique set of seeds
GT_SEEDS=(1789 1687123 12 912391 330)

# Directories (ensure these match the output from Stage 1)
GT_AGENTS_DIR="gt_agents/"
FEEDBACK_DIR="feedback"
REWARD_MODEL_DIR="reward_models"
SAMPLES_DIR="samples" # Required for Demonstrative Feedback generation

# --- Feedback Types (The six types used in the paper) ---
FEEDBACK_TYPES=(
    "evaluative" 
    "comparative" 
    "demonstrative" 
    "corrective" 
    "descriptive" 
    "descriptive_preference"
)

# --- Setup ---
mkdir -p $FEEDBACK_DIR $REWARD_MODEL_DIR

echo "--------------------------------------------------------"
echo "STAGE 3: Training Reward Models (6 Models * 5 Seeds)"
echo "--------------------------------------------------------"

# ==============================================================================
# STAGE 3: TRAIN REWARD MODELS (6 models sequentially per seed)
# ==============================================================================
for SEED in "${GT_SEEDS[@]}"; do
    echo "-> Starting Reward Model Training for Seed: $SEED"
    
    # Loop through all six types for the current seed
    for F_TYPE in "${FEEDBACK_TYPES[@]}"; do
        echo "   -> Training: $F_TYPE"
        
        python multi_type_feedback/train_reward_model.py \
            --algorithm "$ALGO" \
            --environment "$ENV" \
            --feedback-type "$F_TYPE" \
            --seed "$SEED" \
            --feedback-folder "$FEEDBACK_DIR" \
            --save-folder "$REWARD_MODEL_DIR"
        
        if [ $? -ne 0 ]; then
            echo "❌ Reward Model Training FAILED for $F_TYPE (Seed $SEED). Exiting."
            exit 1
        fi
    done
    echo "✅ All 6 reward models trained for Seed $SEED."
done

echo "--------------------------------------------------------"
echo "Pipeline STAGES 3 Completed. Reward models ready for ensemble training."