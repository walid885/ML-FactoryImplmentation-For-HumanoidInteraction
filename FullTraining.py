# optimized_rl_training.py - Time-Optimized RL Training System
import sys
from pathlib import Path
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import gymnasium as gym
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

class OptimizedTrainingMetrics:
    """Optimized metrics collection with early stopping detection"""
    
    def __init__(self, target_reward: float, patience: int = 10):
        self.target_reward = target_reward
        self.patience = patience
        self.reset()
    
    def reset(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.training_times = []
        self.best_score = float('-inf')
        self.no_improvement_count = 0
        self.converged = False
        self.convergence_episode = None
        
    def update(self, reward: float, length: int, loss: Optional[float] = None, 
               training_time: Optional[float] = None) -> bool:
        """Update metrics and return True if should stop training"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if loss is not None:
            self.training_losses.append(loss)
        if training_time is not None:
            self.training_times.append(training_time)
        
        # Check for improvement
        recent_avg = np.mean(self.episode_rewards[-5:])
        if recent_avg > self.best_score:
            self.best_score = recent_avg
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Check convergence
        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-10:]
            if np.mean(recent_rewards) >= self.target_reward:
                if not self.converged:
                    self.convergence_episode = len(self.episode_rewards)
                    self.converged = True
                    return True
        
        # Early stopping due to no improvement
        if self.no_improvement_count >= self.patience:
            return True
            
        return False
    
    def get_final_metrics(self) -> Dict[str, Any]:
        """Calculate final performance metrics"""
        rewards = np.array(self.episode_rewards)
        
        return {
            'final_performance': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
            'stability_score': 1 / (1 + np.std(rewards[-20:])) if len(rewards) >= 20 else 0.5,
            'learning_efficiency': np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) >= 2 else 0,
            'convergence_episode': self.convergence_episode,
            'episodes_trained': len(rewards),
            'converged': self.converged
        }

class OptimizedTrainer:
    """Time-optimized trainer with intelligent resource allocation"""
    
    def __init__(self, max_training_hours: float = 3.0, output_folder: str = "Full_Training_Session"):
        self.max_training_time = max_training_hours * 3600  # Convert to seconds
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Create subfolders
        (self.output_folder / "reports").mkdir(exist_ok=True)
        (self.output_folder / "visualizations").mkdir(exist_ok=True)
        (self.output_folder / "data").mkdir(exist_ok=True)
        
        self.results = []
        self.start_time = None
        
        # Optimized training configurations
        self.training_configs = {
            'ppo': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'eps_clip': 0.2,
                'epochs': 3,  # Reduced for speed
                'batch_size': 64,  # Reduced for speed
                'gae_lambda': 0.95
            },
            'sac': {
                'lr_actor': 3e-4,
                'lr_critic': 3e-4,
                'lr_alpha': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'batch_size': 64,
                'buffer_size': 10000,  # Reduced for speed
                'automatic_entropy_tuning': True
            },
            'td3': {
                'lr_actor': 3e-4,
                'lr_critic': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'batch_size': 64,
                'buffer_size': 10000,  # Reduced for speed
                'policy_noise': 0.2,
                'noise_clip': 0.5,
                'policy_freq': 2
            },
            'ddpg': {
                'lr_actor': 1e-4,
                'lr_critic': 1e-3,
                'gamma': 0.99,
                'tau': 0.005,
                'batch_size': 64,
                'buffer_size': 10000,  # Reduced for speed
                'noise_std': 0.1
            }
        }
        
        # Environment configurations with time limits
        self.environments = {
            'discrete': {
                'CartPole-v1': {'max_episodes': 150, 'target_reward': 195, 'max_time': 300},
                'LunarLander-v2': {'max_episodes': 200, 'target_reward': 200, 'max_time': 600}
            },
            'continuous': {
                'Pendulum-v1': {'max_episodes': 100, 'target_reward': -200, 'max_time': 400},
                'MountainCarContinuous-v0': {'max_episodes': 150, 'target_reward': 90, 'max_time': 500}
            }
        }
    
    def get_remaining_time(self) -> float:
        """Get remaining training time"""
        if self.start_time is None:
            return self.max_training_time
        return self.max_training_time - (time.time() - self.start_time)
    
    def train_algorithm_optimized(self, algo_name: str, algorithm_class, config: Dict,
                                 env_name: str, env_config: Dict) -> Optional[Dict]:
        """Optimized training with early stopping and time constraints"""
        remaining_time = self.get_remaining_time()
        if remaining_time < 60:  # Less than 1 minute remaining
            print(f"⏰ Skipping {algo_name} on {env_name} - insufficient time")
            return None
        
        print(f"🚀 Training {algo_name.upper()} on {env_name} (Time left: {remaining_time/60:.1f}min)")
        
        try:
            # Initialize
            agent = algorithm_class(config)
            metrics = OptimizedTrainingMetrics(env_config['target_reward'])
            
            # Training loop with multiple constraints
            start_time = time.time()
            max_time = min(env_config['max_time'], remaining_time * 0.8)
            
            for episode in range(env_config['max_episodes']):
                # Time check
                if time.time() - start_time > max_time:
                    print(f"⏰ Time limit reached for {algo_name}")
                    break
                
                # Train episode
                episode_start = time.time()
                episode_metrics = agent.train(env_name, num_episodes=1)
                episode_time = time.time() - episode_start
                
                # Update metrics
                reward = episode_metrics['episode_rewards'][-1]
                length = episode_metrics.get('episode_lengths', [100])[-1]
                loss = episode_metrics.get('losses', [None])[-1]
                
                should_stop = metrics.update(reward, length, loss, episode_time)
                
                # Progress update
                if episode % 20 == 0:
                    avg_reward = np.mean(metrics.episode_rewards[-10:])
                    print(f"  Episode {episode}: Avg={avg_reward:.2f}, Best={metrics.best_score:.2f}")
                
                # Early stopping
                if should_stop:
                    reason = "Converged" if metrics.converged else "No improvement"
                    print(f"✓ {algo_name} stopped early: {reason} at episode {episode}")
                    break
            
            # Quick final evaluation (fewer episodes for speed)
            final_scores = []
            for _ in range(5):  # Reduced from 10
                env = gym.make(env_name)
                state, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action = agent.predict(state)
                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                
                final_scores.append(episode_reward)
                env.close()
            
            total_time = time.time() - start_time
            final_metrics = metrics.get_final_metrics()
            
            result = {
                'algorithm': algo_name,
                'environment': env_name,
                'training_time': total_time,
                'final_evaluation': {
                    'mean_score': np.mean(final_scores),
                    'std_score': np.std(final_scores),
                    'max_score': np.max(final_scores),
                    'min_score': np.min(final_scores)
                },
                'metrics': final_metrics,
                'training_history': {
                    'episode_rewards': metrics.episode_rewards,
                    'episode_lengths': metrics.episode_lengths,
                    'training_losses': metrics.training_losses
                }
            }
            
            print(f"✅ {algo_name.upper()}: {np.mean(final_scores):.2f}±{np.std(final_scores):.2f}")
            return result
            
        except Exception as e:
            print(f"❌ {algo_name.upper()} failed: {e}")
            return None
    
    def run_optimized_training(self) -> List[Dict]:
        """Run optimized training with intelligent scheduling"""
        print(f"🎯 Starting Optimized RL Training (Max: {self.max_training_time/3600:.1f}h)")
        self.start_time = time.time()
        
        # Import algorithms
        from algorithms.concrete.ppo import PPOAlgorithm
        from algorithms.concrete.ddpg import DDPGAlgorithm
        from algorithms.concrete.TD3 import TD3Algorithm
        from algorithms.concrete.SAC import SACAlgorithm
        
        algorithms = {
            'ppo': PPOAlgorithm,
            'ddpg': DDPGAlgorithm,
            'td3': TD3Algorithm,
            'sac': SACAlgorithm
        }
        
        # Training schedule with priorities
        training_schedule = [
            # High priority: PPO on discrete
            ('ppo', 'discrete', 'CartPole-v1'),
            ('ppo', 'discrete', 'LunarLander-v2'),
            # High priority: SAC on continuous
            ('sac', 'continuous', 'Pendulum-v1'),
            ('sac', 'continuous', 'MountainCarContinuous-v0'),
            # Medium priority: TD3 on continuous
            ('td3', 'continuous', 'Pendulum-v1'),
            ('td3', 'continuous', 'MountainCarContinuous-v0'),
            # Low priority: DDPG on continuous
            ('ddpg', 'continuous', 'Pendulum-v1'),
            ('ddpg', 'continuous', 'MountainCarContinuous-v0'),
        ]
        
        # Execute training schedule
        for algo_name, env_type, env_name in training_schedule:
            if self.get_remaining_time() < 120:  # Less than 2 minutes
                print(f"⏰ Stopping training - insufficient time remaining")
                break
            
            env_config = self.environments[env_type][env_name]
            result = self.train_algorithm_optimized(
                algo_name, algorithms[algo_name], 
                self.training_configs[algo_name],
                env_name, env_config
            )
            
            if result:
                self.results.append(result)
        
        total_time = time.time() - self.start_time
        print(f"\n🎉 Training completed in {total_time/3600:.2f}h")
        
        return self.results
    
    def save_results(self) -> str:
        """Save training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_folder / "data" / f"training_results_{timestamp}.json"
        
        # Convert to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            for key, value in result['training_history'].items():
                if isinstance(value, np.ndarray):
                    serializable_result['training_history'][key] = value.tolist()
                elif isinstance(value, list):
                    serializable_result['training_history'][key] = [
                        float(v) if v is not None else None for v in value
                    ]
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return str(results_file)

class OptimizedVisualizer:
    """Optimized visualization system"""
    
    def __init__(self, results: List[Dict], output_folder: Path):
        self.results = results
        self.output_folder = output_folder
        self.df = pd.DataFrame(results) if results else pd.DataFrame()
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_performance_dashboard(self) -> str:
        """Create comprehensive performance dashboard"""
        if self.df.empty:
            return ""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Performance Comparison
        self._plot_performance_comparison(axes[0])
        
        # 2. Learning Curves
        self._plot_learning_curves(axes[1])
        
        # 3. Training Time Analysis
        self._plot_training_time(axes[2])
        
        # 4. Convergence Analysis
        self._plot_convergence(axes[3])
        
        # 5. Algorithm Ranking
        self._plot_algorithm_ranking(axes[4])
        
        # 6. Stability vs Performance
        self._plot_stability_performance(axes[5])
        
        plt.suptitle('RL Algorithm Performance Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = self.output_folder / "visualizations" / f"performance_dashboard_{timestamp}.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(dashboard_file)
    
    def _plot_performance_comparison(self, ax):
        """Plot performance comparison"""
        perf_data = []
        for _, row in self.df.iterrows():
            perf_data.append({
                'Algorithm': row['algorithm'].upper(),
                'Environment': row['environment'].split('-')[0],
                'Score': row['final_evaluation']['mean_score'],
                'Std': row['final_evaluation']['std_score']
            })
        
        perf_df = pd.DataFrame(perf_data)
        
        # Group by algorithm and environment
        pivot_df = perf_df.pivot(index='Environment', columns='Algorithm', values='Score')
        pivot_df.plot(kind='bar', ax=ax, alpha=0.8)
        
        ax.set_title('Performance Comparison')
        ax.set_xlabel('Environment')
        ax.set_ylabel('Average Score')
        ax.legend(title='Algorithm')
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_curves(self, ax):
        """Plot learning curves"""
        for _, row in self.df.iterrows():
            rewards = row['training_history']['episode_rewards']
            if rewards:
                smoothed = pd.Series(rewards).rolling(window=10, min_periods=1).mean()
                ax.plot(smoothed, label=f"{row['algorithm'].upper()} ({row['environment'].split('-')[0]})", 
                       alpha=0.8, linewidth=2)
        
        ax.set_title('Learning Curves')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_time(self, ax):
        """Plot training time analysis"""
        times = [row['training_time'] for _, row in self.df.iterrows()]
        algos = [row['algorithm'].upper() for _, row in self.df.iterrows()]
        
        bars = ax.bar(algos, times, alpha=0.7)
        ax.set_title('Training Time Comparison')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Training Time (seconds)')
        
        # Add value labels
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.0f}s', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence(self, ax):
        """Plot convergence analysis"""
        conv_data = []
        for _, row in self.df.iterrows():
            conv_episode = row['metrics']['convergence_episode']
            if conv_episode:
                conv_data.append({
                    'Algorithm': row['algorithm'].upper(),
                    'Environment': row['environment'].split('-')[0],
                    'Convergence': conv_episode
                })
        
        if conv_data:
            conv_df = pd.DataFrame(conv_data)
            conv_df.groupby('Algorithm')['Convergence'].mean().plot(kind='bar', ax=ax, alpha=0.7)
        
        ax.set_title('Convergence Speed')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Episodes to Convergence')
        ax.grid(True, alpha=0.3)
    
    def _plot_algorithm_ranking(self, ax):
        """Plot algorithm ranking"""
        ranking_data = []
        for _, row in self.df.iterrows():
            ranking_data.append({
                'Algorithm': row['algorithm'].upper(),
                'Score': row['final_evaluation']['mean_score']
            })
        
        rank_df = pd.DataFrame(ranking_data)
        avg_scores = rank_df.groupby('Algorithm')['Score'].mean().sort_values(ascending=False)
        
        bars = ax.bar(avg_scores.index, avg_scores.values, alpha=0.7)
        ax.set_title('Algorithm Ranking')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Average Score')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_stability_performance(self, ax):
        """Plot stability vs performance"""
        for _, row in self.df.iterrows():
            stability = row['metrics']['stability_score']
            performance = row['final_evaluation']['mean_score']
            algo = row['algorithm'].upper()
            
            ax.scatter(stability, performance, s=100, label=algo, alpha=0.7)
        
        ax.set_title('Stability vs Performance')
        ax.set_xlabel('Stability Score')
        ax.set_ylabel('Performance Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def generate_optimization_report(self) -> str:
        """Generate optimization report"""
        if not self.results:
            return ""
        
        report = "="*80 + "\n"
        report += "                 OPTIMIZED RL TRAINING REPORT\n"
        report += "="*80 + "\n\n"
        
        # Performance Summary
        report += "📊 PERFORMANCE SUMMARY:\n"
        report += "-" * 50 + "\n"
        
        for result in self.results:
            algo = result['algorithm'].upper()
            env = result['environment']
            score = result['final_evaluation']['mean_score']
            std = result['final_evaluation']['std_score']
            time_taken = result['training_time']
            
            report += f"{algo:6} | {env:20} | {score:8.2f}±{std:5.2f} | {time_taken:6.1f}s\n"
        
        # Find optimal algorithms
        report += "\n🎯 OPTIMIZATION RESULTS:\n"
        report += "-" * 50 + "\n"
        
        # Best performer per environment
        env_best = {}
        for result in self.results:
            env = result['environment']
            score = result['final_evaluation']['mean_score']
            if env not in env_best or score > env_best[env]['score']:
                env_best[env] = {
                    'algorithm': result['algorithm'].upper(),
                    'score': score,
                    'time': result['training_time']
                }
        
        for env, best in env_best.items():
            report += f"{env:25}: {best['algorithm']:6} (Score: {best['score']:7.2f})\n"
        
        # Overall recommendations
        report += "\n🚀 RECOMMENDATIONS:\n"
        report += "-" * 50 + "\n"
        
        # Calculate efficiency scores
        efficiency_scores = {}
        for result in self.results:
            algo = result['algorithm'].upper()
            score = result['final_evaluation']['mean_score']
            time = result['training_time']
            efficiency = score / time if time > 0 else 0
            
            if algo not in efficiency_scores:
                efficiency_scores[algo] = []
            efficiency_scores[algo].append(efficiency)
        
        # Average efficiency
        avg_efficiency = {algo: np.mean(scores) for algo, scores in efficiency_scores.items()}
        best_efficiency = max(avg_efficiency.items(), key=lambda x: x[1])
        
        report += f"• Most Efficient: {best_efficiency[0]} (Score/Time: {best_efficiency[1]:.3f})\n"
        
        # Best overall performance
        best_overall = max(self.results, key=lambda x: x['final_evaluation']['mean_score'])
        report += f"• Best Performance: {best_overall['algorithm'].upper()} on {best_overall['environment']}\n"
        
        # Fastest convergence
        fastest_conv = min([r for r in self.results if r['metrics']['convergence_episode']], 
                          key=lambda x: x['metrics']['convergence_episode'] or float('inf'))
        if fastest_conv['metrics']['convergence_episode']:
            report += f"• Fastest Convergence: {fastest_conv['algorithm'].upper()} ({fastest_conv['metrics']['convergence_episode']} episodes)\n"
        
        report += "\n" + "="*80 + "\n"
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_folder / "reports" / f"optimization_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return str(report_file)


def main():
    """Main execution function"""
    print("🎯 Optimized RL Training System")
    print("=" * 50)
    
    # Initialize trainer with 3-hour limit
    trainer = OptimizedTrainer(max_training_hours=3.0)
    
    # Run optimized training
    results = trainer.run_optimized_training()
    
    if not results:
        print("❌ No training results obtained!")
        return
    
    # Save results
    results_file = trainer.save_results()
    print(f"💾 Results saved to {results_file}")
    
    # Create visualizations
    visualizer = OptimizedVisualizer(results, trainer.output_folder)
    dashboard_file = visualizer.create_performance_dashboard()
    print(f"📊 Dashboard saved to {dashboard_file}")
    
    # Generate report
    report_file = visualizer.generate_optimization_report()
    print(f"📝 Report saved to {report_file}")
    
    # Display final summary
    print("\n🎉 TRAINING COMPLETE!")
    print("=" * 50)
    print("Generated files in Full_Training_Session/:")
    print(f"• Data: {Path(results_file).name}")
    print(f"• Visualization: {Path(dashboard_file).name}")
    print(f"• Report: {Path(report_file).name}")
    
    # Quick performance summary
    print("\n📈 TOP PERFORMERS:")
    print("-" * 30)
    
    sorted_results = sorted(results, key=lambda x: x['final_evaluation']['mean_score'], reverse=True)
    for i, result in enumerate(sorted_results[:3]):
        print(f"{i+1}. {result['algorithm'].upper()} on {result['environment']}: {result['final_evaluation']['mean_score']:.2f}")
    
    return results


if __name__ == "__main__":
    results = main()