#!/bin/bash

# --- Configuration (Use the same paths and seeds as Stage 1) ---
ALGO="sac"
ENV="Hopper-v5"
# Your unique set of seeds
GT_SEEDS=(1789 1687123 12 912391 330)
# GT_SEEDS=(1789 1687123)

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

echo "Starting STAGE 2: Feedback Generation (Per Seed)"
echo "--------------------------------------------------------"

# ==============================================================================
# STAGE 2: GENERATE ALL SIX FEEDBACK TYPES (PER SEED)
# (Assumes random samples are already generated and present in $SAMPLES_DIR)
# ==============================================================================
for SEED in "${GT_SEEDS[@]}"; do
    echo "-> Generating ALL 6 Feedback Datasets for Seed: $SEED"
    
    # This single command generates all six feedback types into the $FEEDBACK_DIR folder.
    python multi_type_feedback/generate_feedback.py \
        --algorithm "$ALGO" \
        --environment "$ENV" \
        --seed "$SEED" \
        --n-feedback 10000 \
        --save-folder "$FEEDBACK_DIR" \
        --expert-model-base-path "$GT_AGENTS_DIR" \
    
    if [ $? -ne 0 ]; then
        echo "❌ Feedback Generation FAILED for Seed $SEED. Exiting."
        exit 1
    fi
done

echo "--------------------------------------------------------"
echo "Pipeline STAGES 2. Feedback generated from experts."