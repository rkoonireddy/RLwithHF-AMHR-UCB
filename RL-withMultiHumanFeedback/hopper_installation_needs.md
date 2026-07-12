# Create a new environment named rl310 with Python 3.10
conda create -n rl310 python=3.10

# Activate the environment
conda activate rl310

# Upgrade pip inside this environment
pip install --upgrade pip

# Install your project in editable mode
pip install -e .

# activate python