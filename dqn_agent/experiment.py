# dqn_agent/experiment.py
import os
import time
import datetime
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp

from .agent import DQNAgentIndependent
from .plotting import plot_all_results
from .config import (
    logger, TOTAL_EXPERIMENTS, EPISODES_PER_EXPERIMENT, MAX_STEPS_PER_EPISODE,
    EVAL_EPISODES_COUNT, EVALUATION_FREQUENCY, TARGET_SCORE_AVG, NUM_WORKERS,
    MODEL_SAVE_DIR, RESULTS_CSV_PATH, PLOT_PATH,
    BASE_CONFIG, HYPERPARAMETER_CONFIGS
)

def run_single_experiment(experiment_config: dict, result_queue: mp.Queue):
    """
    This function runs a full, independent training experiment for a single configuration.
    It is designed to be executed by each worker process.
    
    Args:
        experiment_config (dict): The complete configuration for this run.
        result_queue (mp.Queue): The queue to which results are sent.
    """
    exp_id = experiment_config['id']
    logger.info(f"Starting experiment: {exp_id}")

    # Each experiment runs on the CPU to ensure true parallelism without GIL issues
    # or CUDA context conflicts in a simple setup.
    device = torch.device("cpu")
    
    try:
        env = gym.make("LunarLander-v3")
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
    except Exception as e:
        logger.error(f"Failed to create Gym environment for {exp_id}: {e}")
        return

    agent = DQNAgentIndependent(state_size, action_size, experiment_config, device)
    
    total_steps = 0
    for e in range(1, EPISODES_PER_EXPERIMENT + 1):
        try:
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

            # --- Evaluation Phase ---
            if e % EVALUATION_FREQUENCY == 0:
                eval_rewards = []
                for _ in range(EVAL_EPISODES_COUNT):
                    eval_state, _ = env.reset()
                    episode_reward = 0
                    for _ in range(MAX_STEPS_PER_EPISODE):
                        # Act greedily during evaluation
                        original_epsilon = agent.epsilon
                        agent.epsilon = 0.0
                        action = agent.act(eval_state)
                        agent.epsilon = original_epsilon

                        eval_state, reward, terminated, truncated, _ = env.step(action)
                        episode_reward += reward
                        if terminated or truncated:
                            break
                    eval_rewards.append(episode_reward)
                
                avg_eval_score = np.mean(eval_rewards)
                result_queue.put({
                    'experiment_id': exp_id, 
                    'episode': e, 
                    'avg_eval_score': avg_eval_score,
                    'target_score': TARGET_SCORE_AVG
                })
                
                if e % 50 == 0 or e == EPISODES_PER_EXPERIMENT:
                     logger.info(f"Exp {exp_id} | Ep {e} | Avg Eval Score: {avg_eval_score:.2f} | Epsilon: {agent.epsilon:.3f}")

                if avg_eval_score >= TARGET_SCORE_AVG:
                    logger.info(f"Experiment {exp_id} solved the environment at episode {e}! Stopping early.")
                    break # End this experiment early if solved
        except Exception as ex:
            logger.error(f"An error occurred during training for {exp_id} in episode {e}: {ex}")
            break # Stop this experiment on error
            
    # Save the final model for this experiment
    model_path = os.path.join(MODEL_SAVE_DIR, f"model_{exp_id}.pth")
    agent.save(model_path)
    logger.info(f"Experiment {exp_id} finished.")
    env.close()

class ExperimentOrchestrator:
    """Manages the lifecycle of the parallel experiments."""
    def __init__(self):
        self.script_start_time = datetime.datetime.now()
        logger.info(f"Orchestrator initialized at {self.script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def _prepare_configs(self):
        """Prepares the list of configurations for all experiments."""
        configs_to_run = []
        hyper_config = HYPERPARAMETER_CONFIGS['config_1'] # Using the first defined config
        for i in range(TOTAL_EXPERIMENTS):
            exp_id = f"run_{i+1}"
            config = {**BASE_CONFIG, **hyper_config, 'id': exp_id}
            configs_to_run.append(config)
        return configs_to_run

    def run(self):
        """Main method to execute the entire experiment workflow."""
        configs = self._prepare_configs()
        logger.info(f"Prepared {len(configs)} identical experiments to run in parallel batches of {NUM_WORKERS}.")

        manager = mp.Manager()
        result_queue = manager.Queue()
        all_results = []

        # --- Run experiments in batches using a process pool ---
        for i in range(0, len(configs), NUM_WORKERS):
            batch_configs = configs[i:i + NUM_WORKERS]
            batch_num = i // NUM_WORKERS + 1
            total_batches = (len(configs) + NUM_WORKERS - 1) // NUM_WORKERS
            logger.info(f"--- Starting Batch {batch_num}/{total_batches} with {len(batch_configs)} experiments ---")
            
            processes = []
            for exp_config in batch_configs:
                p = mp.Process(target=run_single_experiment, args=(exp_config, result_queue))
                processes.append(p)
                p.start()

            # Wait for the current batch of processes to finish
            for p in processes:
                p.join()
            
            # Collect results from the queue after the batch is done
            while not result_queue.empty():
                all_results.append(result_queue.get())

        logger.info("All experiment batches have finished.")

        # --- Results Handling ---
        self._process_results(all_results)
        
        script_end_time = datetime.datetime.now()
        logger.info(f"Script finished at {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total script execution time: {script_end_time - self.script_start_time}")
        
    def _process_results(self, all_results: list):
        """Sorts, saves, and plots the final results."""
        if not all_results:
            logger.warning("No results were collected. Check worker processes for errors.")
            return

        results_df = pd.DataFrame(all_results)
        results_df.sort_values(by=['experiment_id', 'episode'], inplace=True)
        
        try:
            results_df.to_csv(RESULTS_CSV_PATH, index=False)
            logger.info(f"All experiment results saved to {RESULTS_CSV_PATH}")
        except Exception as e:
            logger.error(f"Failed to save results to CSV: {e}")
            
        plot_all_results(results_df, PLOT_PATH)