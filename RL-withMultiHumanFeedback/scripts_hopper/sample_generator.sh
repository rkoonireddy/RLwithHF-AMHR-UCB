#!/bin/bash

# --- Configuration ---
ALGO="sac"
ENV="Hopper-v5"
N_SAMPLES=10000
# Your unique set of seeds
GT_SEEDS=(1789 1687123 12 912391 330)
# GT_SEEDS=(1789 1687123)
SAMPLES_DIR="samples"

# Create the samples directory if it doesn't exist
mkdir -p $SAMPLES_DIR

echo "Starting sequential random sample generation for Demonstrative Feedback."
echo "Each run will generate $N_SAMPLES random samples."
echo "--------------------------------------------------------"

# Loop through each seed and run the generation script sequentially
for SEED in "${GT_SEEDS[@]}"; do
    echo "-> Starting sample generation for Seed: $SEED..."
    
    python multi_type_feedback/generate_samples.py \
        --algorithm "$ALGO" \
        --environment "$ENV" \
        --n-samples "$N_SAMPLES" \
        --random \
        --seed "$SEED"
    
    if [ $? -eq 0 ]; then
        echo "✅ Sample generation for Seed $SEED completed successfully."
    else
        echo "❌ Sample generation for Seed $SEED FAILED. Exiting script."
        exit 1
    fi
    echo "--------------------------------------------------------"
done

echo "All random sample generation runs completed. You can now proceed with Feedback Generation (Stage 2)."