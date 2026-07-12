# --- Configuration (Keep consistent) ---
$ALGO = "sac"
$ENV = "Hopper-v5"
$GT_SEEDS = @(1789, 1687123, 12, 912391, 330)
$REWARD_MODEL_DIR = "reward_models" 
$ENSEMBLE_AGENTS_DIR = "ensemble_agents_combos" 

# Set training steps directly in the script for reliability
$TRAIN_STEPS = 1000000 

Write-Host "Starting STAGE 4: Combinatorial Ensemble Policy Training"
Write-Host "--------------------------------------------------------"

# --- DEFINE FEEDBACK COMBINATIONS ---

# Combinations of 2 feedbacks
$COMBOS_2 = @(
    "evaluative comparative"
    "demonstrative corrective"
    "descriptive descriptive_preference"
    "evaluative descriptive"
)

# Combinations of 3 feedbacks
$COMBOS_3 = @(
    "evaluative comparative demonstrative"
    "corrective descriptive descriptive_preference"
    "evaluative corrective descriptive_preference"
)

# Combinations of 4 feedbacks
$COMBOS_4 = @(
    "evaluative comparative demonstrative corrective"
    "demonstrative corrective descriptive descriptive_preference"
)

# Combine all combination sets into one array for the outer loop
$ALL_COMBINATIONS = @($COMBOS_2 + $COMBOS_3 + $COMBOS_4)
# ------------------------------------

# --- Nested Loops for Training ---

# 1. Loop through each combination of feedback types
foreach ($FEEDBACK_TYPES in $ALL_COMBINATIONS) {
    
    # Create a unique, file-safe ID for the combination by replacing spaces with underscores
    $COMBINATION_ID = $FEEDBACK_TYPES.Replace(' ', '_')
    # Count the number of types by splitting the string by space
    $NUM_TYPES = ($FEEDBACK_TYPES -split ' ').Count
    
    Write-Host "========================================================"
    Write-Host "🚀 Starting Training for $NUM_TYPES-Feedback Ensemble: $COMBINATION_ID"
    Write-Host "========================================================"

    # 2. Loop through each Seed
    foreach ($SEED in $GT_SEEDS) {
        
        Write-Host "-> Running Combination '$COMBINATION_ID' with Seed: $SEED"
        
        # Execute the ensemble training script
        # Note: PowerShell uses '&' to execute commands built with variables
        & python multi_type_feedback/train_RL_agent_with_ensemble.py `
            --algorithm $ALGO `
            --environment $ENV `
            --train-steps $TRAIN_STEPS `
            --seed $SEED `
            --save-folder $ENSEMBLE_AGENTS_DIR `
            --reward-model-folder $REWARD_MODEL_DIR `
            --feedback-types $FEEDBACK_TYPES
        
        # Check the exit status of the Python script ($LASTEXITCODE is preferred over $?)
        # A successful execution returns 0
        if ($LASTEXITCODE -ne 0) {
            Write-Error "❌ Ensemble Training FAILED for Combination '$COMBINATION_ID' and Seed $SEED. Last Exit Code: $LASTEXITCODE"
            # Exit the entire script
            exit 1 
        }
        
        Write-Host "✅ Agent trained for Combo '$COMBINATION_ID' and Seed $SEED."
    }
    Write-Host "--------------------------------------------------------"
}

Write-Host "STAGE 4 Completed. All combinatorial ensemble agents are saved in '$ENSEMBLE_AGENTS_DIR'."