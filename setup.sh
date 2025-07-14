#!/bin/bash
# Quick setup script for RL Humanoid Project

echo "Setting up RL Humanoid Project..."

# Create project structure
mkdir -p src/algorithms/concrete src/environments src/utils src/experiments
mkdir -p configs models logs results tests

# Create __init__.py files
touch src/__init__.py src/algorithms/__init__.py src/algorithms/concrete/__init__.py
touch src/environments/__init__.py src/utils/__init__.py src/experiments/__init__.py
touch tests/__init__.py

# Create config files
cat > configs/base_config.yaml << 'EOF'
training:
  num_episodes: 1000
  max_steps_per_episode: 1000
  eval_frequency: 100
  save_frequency: 500

environment:
  name: "HumanoidBulletEnv-v0"
  render_mode: null

logging:
  log_dir: "./logs"
  tensorboard: true
  mlflow: true
  wandb: false

model:
  save_dir: "./models"
  checkpoint_frequency: 1000
EOF

# Create .env file
cat > .env << 'EOF'
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=humanoid_rl
DEVICE=cuda
NUM_WORKERS=4
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
models/
logs/
results/
.pytest_cache/
.DS_Store
EOF

# Install directly to base environment
echo "Installing requirements to base environment..."
pip install --upgrade pip
pip install -r req.txt

echo "Setup complete! Ready to use with base environment."