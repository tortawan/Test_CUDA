# run_experiments.py
import multiprocessing as mp
from dqn_agent.experiment import ExperimentOrchestrator
from dqn_agent.config import logger

def main():
    """
    Main entry point to start the parallel DQN experiments.
    """
    try:
        # On systems other than Linux, 'fork' might not be the default or
        # available, 'spawn' is a safer choice for cross-platform compatibility.
        mp.set_start_method('spawn', force=True)
        
        orchestrator = ExperimentOrchestrator()
        orchestrator.run()
        
    except Exception as e:
        logger.error(f"An error occurred in the main execution block: {e}")
        # Optionally, add more robust error handling or cleanup here

if __name__ == "__main__":
    main()