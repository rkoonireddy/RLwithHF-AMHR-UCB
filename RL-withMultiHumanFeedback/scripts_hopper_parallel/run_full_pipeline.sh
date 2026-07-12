#!/bin/bash
set -e  # Stop on any error

echo "========================================================"
echo "   RL-with-MultiHumanFeedback FULL PIPELINE (Hopper)"
echo "========================================================"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "[1/5] Running Expert Training..."
bash "$SCRIPT_DIR/expert_trainer_parallel.sh"
bash "$SCRIPT_DIR/git_commit.sh"


echo "--------------------------------------------------------"
echo "[2/5] Generating Random Samples for Demonstrative Feedback..."
bash "$SCRIPT_DIR/sample_generator_parallel.sh"
bash "$SCRIPT_DIR/git_commit.sh"

echo "--------------------------------------------------------"
echo "[3/5] Generating Human-style Feedback (All 6 Types)..."
bash "$SCRIPT_DIR/feedback_generator_parallel.sh"
bash "$SCRIPT_DIR/git_commit.sh"


echo "--------------------------------------------------------"
echo "[4/5] Training Single Reward Models..."
bash "$SCRIPT_DIR/single_reward_trainer_parallel.sh"
bash "$SCRIPT_DIR/git_commit.sh"


echo "--------------------------------------------------------"
echo "[5/5] Training Single Reward Models..."
bash "$SCRIPT_DIR/single_fb_agent_trainer_parallel.sh"
bash "$SCRIPT_DIR/git_commit.sh"


echo "--------------------------------------------------------"
echo "[5/5] Training Ensemble Reward Models..."
bash "$SCRIPT_DIR/ensemble_agent_trainer_parallel.sh"
bash "$SCRIPT_DIR/git_commit.sh"


echo "--------------------------------------------------------"
echo "[5/5] Training Ensemble Reward Models..."
bash "$SCRIPT_DIR/ensemble_combo_agent_trainer_parallel.sh"
bash "$SCRIPT_DIR/git_commit.sh"


echo "========================================================"
echo "  🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================================"
