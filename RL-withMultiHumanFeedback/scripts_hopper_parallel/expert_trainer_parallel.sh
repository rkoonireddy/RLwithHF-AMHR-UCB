#!/bin/bash

# --- Configuration ---
# NOTE: Set this based on the number of available CPU cores or GPUs.
# Adjusting this value controls the level of parallelism.
MAX_PARALLEL_JOBS=2

# Algorithm and Environment
ALGO="sac"
ENV="Hopper-v5"

# Directories and Logging
LOG_DIR="gt_agents"
WANDB_PROJECT="hopper-sac-run"
TENSORBOARD_LOGDIR="tensorboard_logs"

# Total timesteps for expert training (Table 5)
TOTAL_TIMESTEPS=1000000
# Save frequency (such that you get 20 check points - 1e6/20 = 50000)
SAVE_FREQ=50000

# Your unique set of 10 seeds (as requested)
# GT_SEEDS=(1789 1687123 12 912391 330 42 101 2025 888 1999)
GT_SEEDS=(1789 1687123 12 912391 330) 

# --- Hyperparameter Overrides (Table 5 + 3-Layer MLP) ---
# NOTE: All string literals (like "auto" and False) must be quoted 
HYPERPARAMS_LIST=(
    'learning_rate:3e-4' 
    'batch_size:256' 
    'gamma:0.99' 
    'ent_coef:"auto"' 
    'normalize:False'
    # Policy Architecture (3-layer MLP, 256 units, ReLU we assume this is only one hidden layer as per standard SAC)
    'policy_kwargs:{"net_arch":{"pi":[256], "qf":[256]}}'
)

GIT_SCRIPT_PATH="/home/user/app/RL-withMultiHumanFeedback/scripts_hopper_parallel/git_commit.sh"

# --- Script Setup ---
mkdir -p "$LOG_DIR"
PIDS=() # Array to store PIDs of background jobs

echo "Starting Parallel Expert Training for $ENV ($ALGO) with 3-Layer MLP."
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

                    echo "--- Job with PID $PID completed successfully. Committing results. ---"
                    # Call the dedicated Git script
                    bash "$GIT_SCRIPT_PATH"
                    echo "------------------------------------------------------------"
                    
                    # Remove the finished PID from the array
                    unset 'PIDS[i]'
                    PIDS=("${PIDS[@]}") # Re-index the array
                    break # Exit inner loop, continue to queue the next seed
                fi
            fi
        done
        echo "Continuing with next seed..."
    fi
done

# --- Final Cleanup and Wait ---
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