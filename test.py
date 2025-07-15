# streamlined_training.py - Focused 30-minute training with key insights
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import gymnasium as gym
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

class StreamlinedTrainer:
    """Streamlined trainer for quick but comprehensive analysis"""
    
    def __init__(self):
        self.results = []
        self.training_time_limit = 25 * 60  # 25 minutes for training
        
    def get_quick_configs(self):
        """Optimized configs for quick training"""
        return {
            'ppo': {
                'learning_rate': 5e-4,
                'gamma': 0.99,
                'eps_clip': 0.2,
                'epochs': 3,
                'batch_size': 64,
                'num_workers': 2
            },
            'ddpg': {
                'lr_actor': 1e-4,
                'lr_critic': 1e-3,
                'gamma': 0.99,
                'tau': 0.01,
                'batch_size': 64,
                'buffer_size': 20000,
                'noise_std': 0.1
            },
            'td3': {
                'lr_actor': 3e-4,
                'lr_critic': 3e-4,
                'gamma': 0.99,
                'tau': 0.01,
                'batch_size': 64,
                'buffer_size': 20000,
                'policy_noise': 0.2,
                'noise_clip': 0.5,
                'policy_freq': 2
            },
            'sac': {
                'lr_actor': 3e-4,
                'lr_critic': 3e-4,
                'lr_alpha': 3e-4,
                'gamma': 0.99,
                'tau': 0.01,
                'batch_size': 64,
                'buffer_size': 20000,
                'automatic_entropy_tuning': True
            }
        }
    
    def quick_train(self, algo_name, algorithm_class, config, env_name, max_episodes=1500):
        """Quick training with focused metrics"""
        print(f"\n🚀 Training {algo_name.upper()} on {env_name}")
        
        try:
            agent = algorithm_class(config)
            rewards = []
            losses = []
            episode_lengths = []
            
            start_time = time.time()
            
            for episode in range(max_episodes):
                # Check time limit
                if time.time() - start_time > self.training_time_limit / 4:  # 1/4 time per algorithm
                    break
                
                # Train episode
                episode_metrics = agent.train(env_name, num_episodes=1)
                reward = episode_metrics['episode_rewards'][-1]
                rewards.append(reward)
                
                if episode_metrics.get('losses'):
                    losses.append(episode_metrics['losses'][-1])
                
                # Progress update
                if episode % 20 == 0:
                    avg_reward = np.mean(rewards[-10:])
                    print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")
                
                # Early stopping if converged
                if len(rewards) >= 20:
                    recent_avg = np.mean(rewards[-10:])
                    older_avg = np.mean(rewards[-20:-10])
                    if recent_avg > older_avg * 1.05:  # 5% improvement threshold
                        continue
                    else:
                        break
            
            # Quick evaluation
            eval_scores = []
            for _ in range(5):
                env = gym.make(env_name)
                state, _ = env.reset()
                episode_reward = 0
                
                for _ in range(200):  # Max steps
                    action = agent.predict(state)
                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    if terminated or truncated:
                        break
                
                eval_scores.append(episode_reward)
                env.close()
            
            training_time = time.time() - start_time
            
            result = {
                'algorithm': algo_name,
                'environment': env_name,
                'episodes_trained': len(rewards),
                'training_time': training_time,
                'training_rewards': rewards,
                'training_losses': losses,
                'evaluation_scores': eval_scores,
                'final_score': np.mean(eval_scores),
                'final_std': np.std(eval_scores),
                'learning_rate': np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 1 else 0,
                'stability': 1 / (1 + np.std(rewards[-10:]) if len(rewards) >= 10 else np.std(rewards))
            }
            
            print(f"✅ {algo_name.upper()} completed: {result['final_score']:.2f} ± {result['final_std']:.2f}")
            return result
            
        except Exception as e:
            print(f"❌ {algo_name.upper()} failed: {e}")
            return None
    
    def run_streamlined_training(self):
        """Run streamlined training for all algorithms"""
        print("🎯 Starting Streamlined RL Training (30 minutes)")
        print("=" * 60)
        
        # Import algorithms
        from algorithms.concrete.ppo import PPOAlgorithm
        from algorithms.concrete.ddpg import DDPGAlgorithm
        from algorithms.concrete.TD3 import TD3Algorithm
        from algorithms.concrete.SAC import SACAlgorithm
        
        configs = self.get_quick_configs()
        
        # Training plan: focus on key environments
        training_plan = [
            ('ppo', PPOAlgorithm, configs['ppo'], 'CartPole-v1', 80),
            ('ddpg', DDPGAlgorithm, configs['ddpg'], 'Pendulum-v1', 60),
            ('td3', TD3Algorithm, configs['td3'], 'Pendulum-v1', 60),
            ('sac', SACAlgorithm, configs['sac'], 'Pendulum-v1', 60)
        ]
        
        results = []
        total_start_time = time.time()
        
        for algo_name, algorithm_class, config, env_name, max_episodes in training_plan:
            result = self.quick_train(algo_name, algorithm_class, config, env_name, max_episodes)
            if result:
                results.append(result)
            
            # Check total time limit
            if time.time() - total_start_time > self.training_time_limit:
                print("⏰ Time limit reached, stopping training")
                break
        
        self.results = results
        return results

class QuickVisualizer:
    """Quick but comprehensive visualizer"""
    
    def __init__(self, results):
        self.results = results
        plt.style.use('seaborn-v0_8')
        
    def create_comparison_dashboard(self):
        """Create focused comparison dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RL Algorithm Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Learning Curves
        ax1 = axes[0, 0]
        self.plot_learning_curves(ax1)
        
        # 2. Final Performance
        ax2 = axes[0, 1]
        self.plot_final_performance(ax2)
        
        # 3. Training Efficiency
        ax3 = axes[0, 2]
        self.plot_training_efficiency(ax3)
        
        # 4. Stability Analysis
        ax4 = axes[1, 0]
        self.plot_stability_analysis(ax4)
        
        # 5. Algorithm Ranking
        ax5 = axes[1, 1]
        self.plot_algorithm_ranking(ax5)
        
        # 6. Recommendation Matrix
        ax6 = axes[1, 2]
        self.plot_recommendation_heatmap(ax6)
        
        plt.tight_layout()
        
        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'rl_comparison_dashboard_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_learning_curves(self, ax):
        """Plot learning curves"""
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, result in enumerate(self.results):
            rewards = result['training_rewards']
            if len(rewards) > 0:
                # Smooth the curve
                smoothed = pd.Series(rewards).rolling(window=5, min_periods=1).mean()
                ax.plot(smoothed, label=f"{result['algorithm'].upper()}", 
                       color=colors[i], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_final_performance(self, ax):
        """Plot final performance comparison"""
        algorithms = [r['algorithm'].upper() for r in self.results]
        scores = [r['final_score'] for r in self.results]
        stds = [r['final_std'] for r in self.results]
        
        bars = ax.bar(algorithms, scores, yerr=stds, alpha=0.7, capsize=5)
        ax.set_ylabel('Final Score')
        ax.set_title('Final Performance Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + bar.get_height()*0.01,
                   f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    def plot_training_efficiency(self, ax):
        """Plot training efficiency"""
        algorithms = [r['algorithm'].upper() for r in self.results]
        times = [r['training_time'] for r in self.results]
        episodes = [r['episodes_trained'] for r in self.results]
        
        # Scatter plot: time vs episodes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (algo, time, ep) in enumerate(zip(algorithms, times, episodes)):
            ax.scatter(time, ep, s=200, label=algo, color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Episodes Trained')
        ax.set_title('Training Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_stability_analysis(self, ax):
        """Plot stability analysis"""
        algorithms = [r['algorithm'].upper() for r in self.results]
        stability_scores = [r['stability'] for r in self.results]
        final_scores = [r['final_score'] for r in self.results]
        
        # Scatter plot: stability vs performance
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (algo, stab, score) in enumerate(zip(algorithms, stability_scores, final_scores)):
            ax.scatter(stab, score, s=200, label=algo, color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Stability Score')
        ax.set_ylabel('Final Performance')
        ax.set_title('Stability vs Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_algorithm_ranking(self, ax):
        """Plot algorithm ranking"""
        # Create comprehensive ranking based on multiple factors
        rankings = []
        for result in self.results:
            # Normalize scores (higher is better)
            perf_score = (result['final_score'] + 2000) / 4000  # Normalize to 0-1
            efficiency_score = min(1, result['episodes_trained'] / 100)  # Episodes efficiency
            stability_score = result['stability']
            speed_score = 1 / (1 + result['training_time'] / 60)  # Time efficiency
            
            # Weighted overall score
            overall_score = (perf_score * 0.4 + efficiency_score * 0.2 + 
                           stability_score * 0.2 + speed_score * 0.2)
            
            rankings.append({
                'Algorithm': result['algorithm'].upper(),
                'Overall Score': overall_score,
                'Performance': perf_score,
                'Efficiency': efficiency_score,
                'Stability': stability_score,
                'Speed': speed_score
            })
        
        # Sort by overall score
        rankings.sort(key=lambda x: x['Overall Score'], reverse=True)
        
        # Plot ranking
        algos = [r['Algorithm'] for r in rankings]
        scores = [r['Overall Score'] for r in rankings]
        
        bars = ax.barh(algos, scores, alpha=0.7)
        ax.set_xlabel('Overall Score')
        ax.set_title('Algorithm Ranking (Weighted)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    def plot_recommendation_heatmap(self, ax):
        """Plot recommendation heatmap"""
        # Create recommendation matrix
        criteria = ['Performance', 'Speed', 'Stability', 'Simplicity']
        algorithms = [r['algorithm'].upper() for r in self.results]
        
        # Calculate scores for each criterion
        matrix = []
        for result in self.results:
            perf_score = (result['final_score'] + 2000) / 4000
            speed_score = 1 / (1 + result['training_time'] / 60)
            stability_score = result['stability']
            
            # Simplicity score (subjective, PPO is simplest)
            simplicity_scores = {'PPO': 1.0, 'DDPG': 0.7, 'TD3': 0.6, 'SAC': 0.5}
            simplicity_score = simplicity_scores.get(result['algorithm'].upper(), 0.5)
            
            matrix.append([perf_score, speed_score, stability_score, simplicity_score])
        
        # Create heatmap
        matrix = np.array(matrix)
        df = pd.DataFrame(matrix, index=algorithms, columns=criteria)
        
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Algorithm Recommendation Matrix')
        ax.set_xlabel('Criteria')
        ax.set_ylabel('Algorithm')
    
    def generate_quick_report(self):
        """Generate quick analysis report"""
        report = "\n" + "="*70 + "\n"
        report += "           QUICK RL ALGORITHM ANALYSIS REPORT\n"
        report += "="*70 + "\n\n"
        
        # Performance summary
        report += "📊 PERFORMANCE SUMMARY:\n"
        report += "-" * 40 + "\n"
        
        for result in self.results:
            algo = result['algorithm'].upper()
            env = result['environment']
            score = result['final_score']
            std = result['final_std']
            episodes = result['episodes_trained']
            time_min = result['training_time'] / 60
            
            report += f"{algo:6} | {env:15} | {score:8.2f}±{std:5.2f} | {episodes:3}ep | {time_min:4.1f}min\n"
        
        # Quick recommendations
        report += "\n🎯 QUICK RECOMMENDATIONS:\n"
        report += "-" * 40 + "\n"
        
        # Find best performers
        best_discrete = max([r for r in self.results if 'CartPole' in r['environment']], 
                           key=lambda x: x['final_score'], default=None)
        
        continuous_results = [r for r in self.results if 'Pendulum' in r['environment']]
        best_continuous = max(continuous_results, key=lambda x: x['final_score'], default=None)
        
        if best_discrete:
            report += f"✅ Best for discrete actions: {best_discrete['algorithm'].upper()}\n"
        
        if best_continuous:
            report += f"✅ Best for continuous actions: {best_continuous['algorithm'].upper()}\n"
        
        # Stability winner
        most_stable = max(self.results, key=lambda x: x['stability'])
        report += f"✅ Most stable: {most_stable['algorithm'].upper()}\n"
        
        # Speed winner
        fastest = min(self.results, key=lambda x: x['training_time'])
        report += f"✅ Fastest training: {fastest['algorithm'].upper()}\n"
        
        # General recommendations
        report += "\n💡 GENERAL GUIDANCE:\n"
        report += "-" * 40 + "\n"
        report += "• Starting out? → PPO (reliable, works well)\n"
        report += "• Continuous control? → SAC (sample efficient)\n"
        report += "• Need stability? → TD3 (addresses overestimation)\n"
        report += "• Simple tasks? → DDPG (straightforward)\n"
        report += "• Complex environments? → SAC or PPO\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report

def main():
    """Main execution function"""
    print("🎯 Streamlined RL Training & Analysis (30 minutes)")
    print("=" * 60)
    
    # Run training
    trainer = StreamlinedTrainer()
    results = trainer.run_streamlined_training()
    
    if not results:
        print("❌ No results obtained!")
        return
    
    print(f"\n✅ Training completed with {len(results)} algorithms")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    visualizer = QuickVisualizer(results)
    
    # Create dashboard
    dashboard = visualizer.create_comparison_dashboard()
    
    # Generate report
    report = visualizer.generate_quick_report()
    print(report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save report
    with open(f"quick_analysis_report_{timestamp}.txt", 'w') as f:
        f.write(report)
    
    print(f"\n🎉 Analysis complete! Files saved with timestamp {timestamp}")
    
    return results, visualizer

if __name__ == "__main__":
    main()