#!/bin/bash

# --- Configuration ---
ALGO="sac"
ENV="Hopper-v5"
LOG_DIR="gt_agents"
# Total timesteps for expert training (Table 5)
TOTAL_TIMESTEPS=1000000
# Save frequency (such that you get 20 check points - 1e6/20 = 50000)
SAVE_FREQ=50000
# Your unique set of seeds
GT_SEEDS=(1789 1687123 12 912391 330)
# GT_SEEDS=(1789 1687123)
WANDB_PROJECT="hopper-sac-run"
TENSORBOARD_LOGDIR="tensorboard_logs"

# # --- Hyperparameter Overrides (Table 5 + 3-Layer MLP) ---
# # Core training parameters
# HYPERPARAMS_CORE="learning_rate:3e-4 batch_size:256 gamma:0.99 ent_coef:auto"

# # Policy Architecture (3-layer MLP, 256 units, ReLU)
# # We must explicitly define the policy_kwargs to enforce the 3-layer net_arch.
# POLICY_KWARGS_JSON='{"net_arch":{"pi":[256, 256, 256], "qf":[256, 256, 256]}, "activation_fn":"torch.nn.ReLU"}'

# --- Hyperparameter Overrides (Table 5 + 3-Layer MLP) ---
# NOTE: All string literals (like "auto" and False) must be quoted 
# to prevent Python's eval() from treating them as undefined variable names.
# HYPERPARAMS_LIST=(
#     'learning_rate:3e-4' 
#     'batch_size:256' 
#     'gamma:0.99' 
#     'ent_coef:"auto"' 
#     'normalize:False'
#     'policy_kwargs:{"net_arch":{"pi":[256, 256, 256], "qf":[256, 256, 256]}, "activation_fn":"torch.nn.ReLU"}'
# )

HYPERPARAMS_LIST=(
    'learning_rate:3e-4' 
    'batch_size:256' 
    'gamma:0.99' 
    'ent_coef:"auto"' 
    'normalize:False'
    # We must still pass policy_kwargs to ensure the 3-layer MLP override takes precedence
    'policy_kwargs:{"net_arch":{"pi":[256, 256, 256], "qf":[256, 256, 256]}}'
)

# --- Script Setup ---
mkdir -p $LOG_DIR

echo "Starting Sequential Expert Training for $ENV ($ALGO) with 3-Layer MLP."
echo "--------------------------------------------------------"

echo "Starting Sequential Expert Training for $ENV ($ALGO) with 3-Layer MLP."
echo "Total Timesteps: $TOTAL_TIMESTEPS"
echo "--------------------------------------------------------"

# Loop sequentially through each seed
for SEED in "${GT_SEEDS[@]}"; do
    echo "Starting training for Seed: $SEED..."
    
    # --- Full Training Command with Corrected Hyperparams Parsing ---
    python train_baselines/train.py \
        --algo "$ALGO" \
        --env "$ENV" \
        --n-timesteps "$TOTAL_TIMESTEPS" \
        --save-freq "$SAVE_FREQ" \
        --seed "$SEED" \
        --log-folder "$LOG_DIR" \
        --wandb-project "$WANDB_PROJECT" \
        -tb "$TENSORBOARD_LOGDIR" \
        --verbose 1 \
        --hyperparams "${HYPERPARAMS_LIST[@]}"
    
    # Check for success
    if [ $? -eq 0 ]; then
        echo "✅ Training for Seed $SEED completed successfully."
    else
        echo "❌ Training for Seed $SEED FAILED. Stopping execution."
        exit 1
    fi
    echo "--------------------------------------------------------"
done

echo "All expert training runs completed."