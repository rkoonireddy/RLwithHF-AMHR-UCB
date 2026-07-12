#!/bin/bash

# --- Configuration ---
# NOTE: Set this based on the number of available CPU cores or GPUs.
# Adjusting this value controls the level of parallelism.
MAX_PARALLEL_JOBS=2

# Algorithm and Environment
ALGO="ppo" # Changed from SAC to PPO based on parameters (GAE, Clip Range)
ENV="Swimmer-v5"

# Directories and Logging
LOG_DIR="gt_agents"
WANDB_PROJECT="swimmer-ppo-expert-run"
TENSORBOARD_LOGDIR="tensorboard_logs"

# Total timesteps for expert training
TOTAL_TIMESTEPS=1000000
# Save frequency (1e6 / 20 checkpoints = 50,000)
SAVE_FREQ=50000

# Your unique set of 10 seeds (as requested)
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 

# --- Hyperparameter Overrides (Swimmer-v5 PPO/Tuned Parameters) ---
# Policy Architecture is set to 3-layer MLP (256, 256, 256) based on hidden size 256 and 3 layers
# NOTE: All string literals (like "auto" and False) must be quoted 
HYPERPARAMS_LIST=(
    'learning_rate:6e-4'          # From Table
    'batch_size:256'              # From Table
    'gamma:0.9999'                # From Table
    'gae_lambda:0.98'             # From Table
    'n_envs:4'                    # From Table
    'n_steps:1024'                # From Table (Sampled Steps per Env.)
    'n_epochs:10'                 # From Table (Num. Epochs)
    'ent_coef:0.0554757'          # From Table
    'vf_coef:0.38782'             # From Table
    'max_grad_norm:0.6'           # From Table
    'clip_range:0.3'              # From Table
    'normalize:False'             # From Table (Obs. Normalization)
    'normalize_advantage:False'   # Assuming this is implied by not normalizing Obs/Reward
    
    # Network Architecture: 3-layer MLP, 256 units
    # PPO's MlpPolicy default is two hidden layers (64, 64). We extend it to three layers.
    'policy_kwargs:{"net_arch":[256]}' 
)


GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_swimmer_parallel/git_commit.sh"

# --- Script Setup ---
mkdir -p "$LOG_DIR"
PIDS=() # Array to store PIDs of background jobs

echo "Starting Parallel Expert Training for $ENV ($ALGO) with 3-Layer MLP (256x3)."
echo "Total Timesteps: $TOTAL_TIMESTEPS"
echo "Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "Total Seeds to Run: ${#GT_SEEDS[@]}"
echo "--------------------------------------------------------"

# Loop through each seed and manage parallelism
for SEED in "${GT_SEEDS[@]}"; do
    echo "Queuing training for Seed: $SEED..."
    
    # --- Full Training Command ---
    python train_baselines/train.py \
        --algo "$ALGO" \
        --env "$ENV" \
        --n-timesteps "$TOTAL_TIMESTEPS" \
        --save-freq "$SAVE_FREQ" \
        --seed "$SEED" \
        --log-folder "$LOG_DIR" \
        --wandb-project "$WANDB_PROJECT" \
        -tb "$TENSORBOARD_LOGDIR" \
        --verbose 0 \
        --hyperparams "${HYPERPARAMS_LIST[@]}" &
    
    # Store the Process ID (PID) of the job that was just launched in the background
    PIDS+=($!)
    
    # Check if we've reached the limit of parallel jobs
    if [ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; then
        echo "--------------------------------------------------------"
        echo "Maximum parallel jobs reached ($MAX_PARALLEL_JOBS). Waiting for a job to finish..."
        # Wait for ANY currently running background job to finish
        wait -n
        
        # Re-evaluate the PIDS array to check which job finished and manage exit codes
        for i in "${!PIDS[@]}"; do
            PID=${PIDS[$i]}
            # Use 'ps' to check if the process is still running. If it's not, it finished.
            if ! kill -0 "$PID" 2>/dev/null; then
                # Process has finished. Get its exit code.
                wait "$PID"
                EXIT_CODE=$?
                
                if [ $EXIT_CODE -ne 0 ]; then
                    echo "❌ A background job with PID $PID FAILED with exit code $EXIT_CODE. Stopping execution."
                    # Kill all other background processes and exit
                    kill "${PIDS[@]}" 2>/dev/null
                    exit 1
                else
                    # Remove the finished PID from the array
                    echo "--- Job with PID $PID completed successfully. Committing results. ---"
                    # Call the dedicated Git script
                    bash "$GIT_SCRIPT_PATH"
                    echo "------------------------------------------------------------"
                    unset 'PIDS[i]'
                    PIDS=("${PIDS[@]}") # Re-index the array
                    break # Exit inner loop, continue to queue the next seed
                fi
            fi
        done
        echo "Continuing with next seed..."
    fi
done

# --- Final Cleanup and Wait ---i hav e
echo "--------------------------------------------------------"
echo "All seeds have been queued. Waiting for the final ${#PIDS[@]} job(s) to complete."

# Wait for all remaining background processes to finish
EXIT_STATUS=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || EXIT_STATUS=$? # Capture the exit status of each job
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "❌ A final background job with PID $PID FAILED with exit code $EXIT_STATUS."
    fi
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ All expert training runs completed successfully."
else
    echo "❌ One or more expert training runs FAILED. Please check the logs."
    exit 1
fi