#!/bin/bash
export WANDB_MODE=offline
export WANDB_SILENT=true

# ALGO="sac"
ALGO="ppo"
ENV="Humanoid-v5"
SEEDS=(111)
FEEDBACK_TYPES=(
  evaluative
  comparative
  corrective
  descriptive
  descriptive_preference
)

REWARD_MODEL_FOLDER="uros_reward_models"
SAVE_FOLDER="uros_fb_agents"
LOG_FOLDER="logs_fb_agents"
BATCH_SIZE=5
job_counter=0

mkdir -p "$SAVE_FOLDER"
mkdir -p "$LOG_FOLDER"

# Prepare all combinations
declare -a combinations
for SEED in "${SEEDS[@]}"; do
    for FEEDBACK_TYPE in "${FEEDBACK_TYPES[@]}"; do
        combinations+=("$ALGO $SEED $ENV $FEEDBACK_TYPE")
    done
done

total_combinations=${#combinations[@]}

# Batch execution
for ((i=0; i<$total_combinations; i+=$BATCH_SIZE)); do
    batch=("${combinations[@]:$i:$BATCH_SIZE}")
    pids=()

    echo "Starting batch $((i / BATCH_SIZE + 1)) (Max $BATCH_SIZE concurrent jobs)"

    for combination in "${batch[@]}"; do
        read algo seed env feedback <<< "$combination"
        job_counter=$((job_counter + 1))

        # Folder created by training script
        AGENT_DIR="agents/RL_${algo}_${env}_${seed}_${feedback}_${seed}/${algo}"
        LOGFILE="${LOG_FOLDER}/${env}_${algo}_${feedback}_seed${seed}.log"

        if [[ -d "$AGENT_DIR" ]]; then
            echo "[SKIP] RL agent already exists: $AGENT_DIR"
            continue
        fi

        echo "[RUN] Training RL agent: $feedback (seed $seed $algo)"

        # Run training in background, redirect both stdout and stderr
        python multi_type_feedback/train_RL_agent.py \
            --algorithm "$algo" \
            --environment "$env" \
            --feedback-type "$feedback" \
            --seed "$seed" \
            --reward-model-folder "$REWARD_MODEL_FOLDER" \
            --save-folder "$SAVE_FOLDER" \
            >"$LOGFILE" 2>&1 &

        pids+=($!)
    done

    # Wait for all jobs in this batch
    if (( ${#pids[@]} > 0 )); then
        wait "${pids[@]}"
    fi
done

echo "All $job_counter feedback-based RL agent trainings completed."
