"""
DQN Agent for Gymnasium's LunarLander-v3 Environment using PyTorch.

This script is designed to run multiple, completely independent reinforcement
learning experiments in parallel. Each experiment runs on a separate CPU core,
with its own agent, its own neural network, and its own replay buffer.

This version runs multiple experiments using the *same* set of hyperparameters
to analyze the variance and stability of the training process.

Key Features:
- Independent Parallel Experiments: Uses Python's `multiprocessing` to run
  fully separate training sessions simultaneously.
- Identical Configuration Runs: All parallel experiments use the same
  hyperparameter settings to test training consistency.
- Centralized Logging of Results: The main process gathers evaluation scores
  from all independent experiments into a single CSV file for easy analysis.
- Comparative Plotting: Generates a single plot that visualizes the performance
  curves of all experiments, making it easy to compare results.
- Dynamic Worker Allocation: Automatically sets the number of parallel experiments
  to 80% of the available CPU cores.
"""

import random
import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from typing import Tuple, List, Dict, Any, Deque
import multiprocessing as mp
import time

# Workaround for OpenMP runtime error on some systems
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- Configuration Constants ---
TOTAL_EXPERIMENTS: int = 15           # The total number of identical experiments to run.
EPISODES_PER_EXPERIMENT: int = 500     # Total number of episodes to run for each independent experiment
TARGET_SCORE_AVG: float = 250          # Target average score for solving the environment
MAX_STEPS_PER_EPISODE: int = 1000      # Max steps per episode (for training and evaluation)
EVAL_EPISODES_COUNT: int = 10          # Number of episodes for each evaluation run
EVALUATION_FREQUENCY: int = 5          # Evaluate the agent every 5 episodes, as requested.

# --- Parallelism Configuration ---
try:
    total_cores = os.cpu_count()
    if total_cores:
        # Limit workers to the number of experiments if it's smaller
        NUM_WORKERS: int = min(int(total_cores * 0.8), TOTAL_EXPERIMENTS)
    else:
        NUM_WORKERS: int = 4 # Fallback value
except NotImplementedError:
    NUM_WORKERS: int = 4

# --- File Paths ---
# Each experiment will save its own model. Results are aggregated.
MODEL_SAVE_DIR: str = "models"
RESULTS_CSV_PATH: str = "all_experiments_results.csv"
PLOT_PATH: str = "all_experiments_plot.png"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- DQN Agent Hyperparameters ---
# Define the single set of hyperparameters to be used for all 20 experiments.
hyperparameter_config_to_test: Dict[str, Any] = {
    'first_hid': 256,
    'second_hid': 128,
    'learning_rate': 0.0005
}

# Common settings for all experiments
base_config: Dict[str, Any] = {
    'batch_size': 256,
    'gamma': 0.99,
    'step_to_update': 2,
    'epsilon_decay': 0.995,
    'replay_memory_capacity': 100000
}

class QNetwork(nn.Module):
    """Neural Network for Q-value approximation."""
    def __init__(self, state_size: int, action_size: int, first_hid: int, second_hid: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, first_hid)
        self.fc2 = nn.Linear(first_hid, second_hid)
        self.fc3 = nn.Linear(second_hid, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgentIndependent:
    """A self-contained DQN Agent for a single experiment."""
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any], agent_device: torch.device):
        self.config = config
        self.device = agent_device
        self.action_size = action_size
        self.memory: Deque = deque(maxlen=self.config['replay_memory_capacity'])
        self.epsilon = 1.0
        self.epsilon_min = 0.01

        self.model = QNetwork(state_size, action_size, config['first_hid'], config['second_hid']).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.loss_fn = nn.MSELoss()

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

    def act(self, state_np: np.ndarray) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.from_numpy(state_np.flatten()).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(action_values[0]).item()

    def replay(self):
        if len(self.memory) < self.config['batch_size']:
            return
        minibatch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states_tensor = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions_tensor = torch.tensor(actions, device=self.device).long().view(-1, 1)
        rewards_tensor = torch.tensor(rewards, device=self.device).float().view(-1, 1)
        next_states_tensor = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones_tensor = torch.tensor(dones, device=self.device).float().view(-1, 1)

        current_q_values = self.model(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q_values = self.model(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q_values = rewards_tensor + (self.config['gamma'] * next_q_values * (1 - dones_tensor))
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.config['epsilon_decay']

    def save(self, filepath: str):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(self.model.state_dict(), filepath)
        except Exception as e:
            logger.error(f"Error saving model to {filepath}: {e}")

def run_experiment(
    experiment_config: dict,
    result_queue: mp.Queue,
):
    """
    This function runs a full, independent training experiment.
    It is executed by each worker process.
    """
    exp_id = experiment_config['id']
    logger.info(f"Starting experiment: {exp_id}")

    # Each experiment runs on the CPU.
    device = torch.device("cpu")
    
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgentIndependent(state_size, action_size, experiment_config, device)
    
    total_steps = 0
    for e in range(1, EPISODES_PER_EXPERIMENT + 1):
        state, _ = env.reset()
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_steps += 1
            if total_steps % agent.config['step_to_update'] == 0:
                agent.replay()
            if done:
                break
        agent.update_epsilon()

        # Perform evaluation every 5 episodes
        if e % EVALUATION_FREQUENCY == 0:
            eval_rewards = []
            for _ in range(EVAL_EPISODES_COUNT):
                state, _ = env.reset()
                episode_reward = 0
                for _ in range(MAX_STEPS_PER_EPISODE):
                    # Act greedily during eval by setting epsilon to 0 temp
                    original_epsilon = agent.epsilon
                    agent.epsilon = 0.0
                    action = agent.act(state)
                    agent.epsilon = original_epsilon

                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    if terminated or truncated:
                        break
                eval_rewards.append(episode_reward)
            
            avg_eval_score = np.mean(eval_rewards)
            result_queue.put({'experiment_id': exp_id, 'episode': e, 'avg_eval_score': avg_eval_score})
            # Reduce logging verbosity to avoid spam
            if e % 50 == 0 or e == EPISODES_PER_EXPERIMENT:
                 logger.info(f"Exp {exp_id} | Ep {e} | Avg Eval Score: {avg_eval_score:.2f} | Epsilon: {agent.epsilon:.3f}")

            if avg_eval_score >= TARGET_SCORE_AVG:
                logger.info(f"Experiment {exp_id} solved the environment!")
                break # End this experiment early if solved
    
    # Save the final model for this experiment
    model_path = os.path.join(MODEL_SAVE_DIR, f"model_{exp_id}.pth")
    agent.save(model_path)
    logger.info(f"Experiment {exp_id} finished.")
    env.close()

def plot_all_results(df_results: pd.DataFrame, plot_path: str):
    """Plots and saves the training progress for all experiments."""
    plt.figure(figsize=(15, 8))
    
    # Plot individual experiment runs with transparency
    for exp_id in df_results['experiment_id'].unique():
        exp_df = df_results[df_results['experiment_id'] == exp_id]
        plt.plot(exp_df['episode'], exp_df['avg_eval_score'], 'o-', label=None, alpha=0.2, color='gray')

    # Calculate and plot the mean and standard deviation across all experiments
    agg_df = df_results.groupby('episode')['avg_eval_score'].agg(['mean', 'std']).reset_index()
    plt.plot(agg_df['episode'], agg_df['mean'], 'o-', color='blue', label='Mean Score Across All Experiments')
    plt.fill_between(
        agg_df['episode'],
        agg_df['mean'] - agg_df['std'],
        agg_df['mean'] + agg_df['std'],
        color='blue',
        alpha=0.2,
        label='Standard Deviation'
    )

    plt.title(f'Comparison of {TOTAL_EXPERIMENTS} Identical DQN Experiments', fontsize=18)
    plt.xlabel('Episode Number', fontsize=14)
    plt.ylabel('Average Evaluation Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"Comparative plot saved to {plot_path}")
    plt.close()

def main():
    """Main function to orchestrate parallel experiments."""
    script_start_time = datetime.datetime.now()
    logger.info(f"Starting script at {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Create the list of 20 identical experiment configurations ---
    configs_to_run = []
    for i in range(TOTAL_EXPERIMENTS):
        # Create a unique ID for each experiment run
        exp_id = f"run_{i+1}"
        config = {**base_config, **hyperparameter_config_to_test, 'id': exp_id}
        configs_to_run.append(config)
    
    logger.info(f"Will run {len(configs_to_run)} identical experiments in parallel batches of {NUM_WORKERS}.")

    manager = mp.Manager()
    result_queue = manager.Queue()
    all_results = []

    # --- Run experiments in batches ---
    for i in range(0, len(configs_to_run), NUM_WORKERS):
        batch_configs = configs_to_run[i:i + NUM_WORKERS]
        logger.info(f"--- Starting Batch {i//NUM_WORKERS + 1}/{len(configs_to_run)//NUM_WORKERS + 1} with {len(batch_configs)} experiments ---")
        
        processes = []
        for exp_config in batch_configs:
            p = mp.Process(target=run_experiment, args=(exp_config, result_queue))
            processes.append(p)
            p.start()

        # Wait for the current batch of processes to finish
        for p in processes:
            p.join()
        
        # Collect all results from the queue after the batch is done
        while not result_queue.empty():
            result = result_queue.get()
            all_results.append(result)

    logger.info("All experiment batches have finished.")

    # --- Results Handling ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.sort_values(by=['experiment_id', 'episode'], inplace=True)
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
        logger.info(f"All experiment results saved to {RESULTS_CSV_PATH}")
        plot_all_results(results_df, PLOT_PATH)
    else:
        logger.warning("No results were collected.")

    script_end_time = datetime.datetime.now()
    logger.info(f"Script finished at {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total script execution time: {script_end_time - script_start_time}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
