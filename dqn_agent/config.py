# dqn_agent/config.py
import os
import logging
from typing import Dict, Any

# --- Workaround ---
# This addresses a common issue with matplotlib and multiprocessing on some systems.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- Experiment Configuration ---
TOTAL_EXPERIMENTS: int = 1           # The total number of identical experiments to run.
EPISODES_PER_EXPERIMENT: int = 500     # Total number of episodes for each independent experiment
TARGET_SCORE_AVG: float = 250          # Target average score for solving the environment
MAX_STEPS_PER_EPISODE: int = 1000      # Max steps per episode (for training and evaluation)
EVAL_EPISODES_COUNT: int = 20          # Number of episodes for each evaluation run
EVALUATION_FREQUENCY: int = 5          # Evaluate the agent every 5 episodes.

# --- Parallelism Configuration ---
try:
    total_cores = os.cpu_count()
    if total_cores:
        # Use 80% of cores, but not more than the total experiments
        NUM_WORKERS: int = min(int(total_cores * 0.8), TOTAL_EXPERIMENTS)
    else:
        NUM_WORKERS: int = 4 # Sensible fallback
except NotImplementedError:
    NUM_WORKERS: int = 4

# --- File & Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_DIR: str = os.path.join(BASE_DIR, "models")
RESULTS_CSV_PATH: str = os.path.join(BASE_DIR, "all_experiments_results.csv")
PLOT_PATH: str = os.path.join(BASE_DIR, "all_experiments_plot.png")

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- DQN Agent Hyperparameters ---
# By defining configs this way, you could easily add more to test in the future.
HYPERPARAMETER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'config_1': {
        'first_hid': 200,
        'second_hid': 200,
        'learning_rate': 0.0005
    }
    # Future example:
    # 'config_2': {
    #     'first_hid': 512,
    #     'second_hid': 256,
    #     'learning_rate': 0.0001
    # }
}

# Settings common to all experiments
BASE_CONFIG: Dict[str, Any] = {
    'batch_size': 256,
    'gamma': 0.99,
    'step_to_update': 2,
    'epsilon_decay': 0.995,
    'replay_memory_capacity': 100000
}