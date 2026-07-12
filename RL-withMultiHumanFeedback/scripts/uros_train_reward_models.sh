#!/bin/bash
export WANDB_MODE=offline
export WANDB_SILENT=true

ALGO="sac"
ENV="Humanoid-v5"
SEED=111
FEEDBACK_FOLDER="uros_feedback"
SAVE_FOLDER="uros_reward_models"

mkdir -p "$SAVE_FOLDER"
mkdir -p logs_reward_models

FEEDBACK_TYPES=(
  evaluative
  comparative
  corrective
  descriptive
  descriptive_preference
)

PIDS=()

for FEEDBACK_TYPE in "${FEEDBACK_TYPES[@]}"; do

    CKPT_PATH="${SAVE_FOLDER}/${ALGO}_${ENV}_${SEED}_${FEEDBACK_TYPE}_${SEED}.ckpt"
    LOGFILE="logs_reward_models/${ENV}_${ALGO}_${FEEDBACK_TYPE}_seed${SEED}.log"

    if [[ -f "$CKPT_PATH" ]]; then
        echo "[SKIP] Reward model already exists: $CKPT_PATH"
        continue
    fi

    echo "[RUN] Training reward model: $FEEDBACK_TYPE"

    python multi_type_feedback/train_reward_model.py \
        --algorithm "$ALGO" \
        --environment "$ENV" \
        --feedback-type "$FEEDBACK_TYPE" \
        --seed "$SEED" \
        --feedback-folder "$FEEDBACK_FOLDER" \
        --save-folder "$SAVE_FOLDER" \
        --no-loading-bar \
        2>&1 | tee "$LOGFILE" &

    PIDS+=($!)
done

# Wait for all parallel jobs to finish
if (( ${#PIDS[@]} > 0 )); then
    echo "Waiting for ${#PIDS[@]} parallel jobs..."
    wait "${PIDS[@]}"
fi

echo "All reward models completed."
