#!/bin/bash
export WANDB_MODE=offline
# Define arrays
# ENVS=("Humanoid-v5")
ENVS=("Swimmer-v5")
ALGOS=("ppo" "sac")         # <-- Add multiple algorithms here
SEEDS=(111 2222)
TIMESTEPS=1000000
SAVE_FREQ=50000 
LOG_FOLDER="train_baselines/uros_gt_agents"
BATCH_SIZE=4

# Create log directories
mkdir -p logs
mkdir -p "$LOG_FOLDER"

declare -a combinations

# Generate combinations
for algo in "${ALGOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for env in "${ENVS[@]}"; do
            combinations+=("$algo $seed $env")
        done
    done
done

total_combinations=${#combinations[@]}
job_counter=0

# Execute in batches
for ((i=0; i<$total_combinations; i+=$BATCH_SIZE)); do
    batch=("${combinations[@]:$i:$BATCH_SIZE}")
    batch_id=$((i / BATCH_SIZE))
    pids=()
    
    echo "Starting Batch $((batch_id + 1)) (Max $BATCH_SIZE concurrent jobs)"
    
    # Run jobs concurrently
    for combination in "${batch[@]}"; do
        read algo seed env <<< "$combination"
        job_counter=$((job_counter + 1))
        
        LOG_FILE="logs/${env}_${algo}_seed${seed}.log"
        
        # Execute the training script in the background
        python train_baselines/train.py \
          --algo "$algo" \
          --env "$env" \
          --n-timesteps "$TIMESTEPS" \
          --save-freq "$SAVE_FREQ" \
          --seed "$seed" \
          --log-folder "$LOG_FOLDER" \
          --track \
          2>&1 | tee "$LOG_FILE" &

        pids+=($!)
    done
    
    # Wait for all background jobs in the current batch to complete
    wait "${pids[@]}"
done

echo "All $job_counter training runs completed."
