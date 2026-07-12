#!/bin/bash

# --- Configuration (Keep consistent) ---
ALGO="sac"
ENV="Hopper-v5"
GT_SEEDS=(1789 1687123 12 912391 330)
# GT_SEEDS=(12)
REWARD_MODEL_DIR="reward_models" 
ENSEMBLE_AGENTS_DIR="ensemble_agents" 
ALL_TYPES="evaluative comparative demonstrative corrective descriptive descriptive_preference"

# Set training steps directly in the script for reliability
TRAIN_STEPS=1000000 

echo "Starting STAGE 4: Ensemble Policy Training (Final Attempt)"
echo "--------------------------------------------------------"

for SEED in "${GT_SEEDS[@]}"; do
    echo "-> Starting Ensemble Training for Seed: $SEED"
    
    # We remove all complex hyperparams and rely entirely on the script's hardcoded defaults,
    # as the ensemble script does not recognize the --hyperparams flag anyway.
    # We only pass the recognized arguments: train-steps, algorithm, env, seed, folders, and types.
    
    python multi_type_feedback/train_RL_agent_with_ensemble.py \
        --algorithm "$ALGO" \
        --environment "$ENV" \
        --train-steps "$TRAIN_STEPS" \
        --seed "$SEED" \
        --save-folder "$ENSEMBLE_AGENTS_DIR" \
        --reward-model-folder "$REWARD_MODEL_DIR" \
        --feedback-types $ALL_TYPES
    
    if [ $? -ne 0 ]; then
        echo "❌ Ensemble Training FAILED for Seed $SEED. Exiting."
        exit 1
    fi
    echo "✅ Ensemble agent trained for Seed $SEED."
    echo "--------------------------------------------------------"
done

echo "STAGE 4 Completed. All final ensemble agents are saved in '$ENSEMBLE_AGENTS_DIR'."