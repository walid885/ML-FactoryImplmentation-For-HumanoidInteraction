import argparse
import os
import time
import mlflow
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.factory import RLAlgorithmFactory
from algorithms.concrete.ppo import PPOAlgorithm
from utils.config import RLConfig

def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("humanoid_rl")

def train_algorithm(algo_name: str, env_name: str, config_path: str = None):
    """Train specified algorithm"""
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'epochs': 4,
            'batch_size': 256,
            'num_workers': 6
        }
    
    config = RLConfig(**config_dict)
    
    # Setup factory
    factory = RLAlgorithmFactory()
    factory.register_algorithm("ppo", PPOAlgorithm)
    
    # Create algorithm
    agent = factory.create_algorithm(algo_name, config)
    
    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_params(config.to_dict())
        mlflow.log_param("algorithm", algo_name)
        mlflow.log_param("environment", env_name)
        
        start_time = time.time()
        
        # Train
        print(f"Starting training: {algo_name} on {env_name}")
        metrics = agent.train(env_name, num_episodes=100)
        
        training_time = time.time() - start_time
        
        # Log metrics
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("final_avg_reward", metrics['episode_rewards'][-1])
        
        # Log episode rewards
        for i, reward in enumerate(metrics['episode_rewards']):
            mlflow.log_metric("episode_reward", reward, step=i)
        
        # Save model
        model_path = f"models/{algo_name}_{env_name}_{int(time.time())}.pkl"
        os.makedirs("models", exist_ok=True)
        agent.save(model_path)
        mlflow.log_artifact(model_path)
        
        print(f"Training completed in {training_time/60:.1f} minutes")
        print(f"Model saved to {model_path}")
        print(f"Final average reward: {metrics['episode_rewards'][-1]:.2f}")
        
        return agent, metrics

def main():
    parser = argparse.ArgumentParser(description="Train RL Algorithm")
    parser.add_argument("--algo", choices=["ppo"], default="ppo",
                       help="Algorithm to train")
    parser.add_argument("--env", default="CartPole-v1",
                       help="Environment to train on")
    parser.add_argument("--config", default=None,
                       help="Config file path")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes")
    
    args = parser.parse_args()
    
    # Setup
    setup_mlflow()
    
    # Train
    agent, metrics = train_algorithm(args.algo, args.env, args.config)
    
    print("\n=== Training Summary ===")
    print(f"Algorithm: {args.algo}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Best reward: {max(metrics['episode_rewards']):.2f}")

if __name__ == "__main__":
    main()