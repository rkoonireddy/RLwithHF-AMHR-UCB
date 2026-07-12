# Define your local project folder (the one containing the Dockerfile, train_baselines/, etc.)
LOCAL_PROJECT_PATH=$(pwd)

hf jobs run \
    --flavor a10g-small \
    --image python:3.10-slim-bullseye \
    --name parallel-sac-training-run \
    --mount local=$LOCAL_PROJECT_PATH,remote=/workspace/project \
    --secrets WANDB_API_KEY='2b2e568aa0775e667f4fc74b109fbbcfd1615eae' \
    --attach \
    -- \
    /bin/bash -c " \
        # 1. Create and activate venv
        python3.10 -m venv /workspace/project/venv; \
        source /workspace/project/venv/bin/activate; \
        
        # 2. Install dependencies (editable install)
        pip install --no-cache-dir --upgrade pip; \
        pip install --no-cache-dir -e /workspace/project; \
        
        # 3. Log into WandB (uses the secret WANDB_API_KEY)
        wandb login; \
        
        # 4. Run your parallel training script
        /workspace/project/scripts_hopper/run_parallel_training.sh \
    "