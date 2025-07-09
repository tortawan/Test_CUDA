# 🚀 Parallel Reinforcement Learning Experiment Framework

## Overview

This project provides a robust framework for running multiple, independent reinforcement learning experiments in parallel. It is designed to analyze the stability and variance of a Deep Q-Network (DQN) agent on the **LunarLander-v3** environment from Gymnasium.

Instead of a single script, the project is organized into a modular structure that separates concerns such as configuration, model architecture, agent logic, and experiment execution. This professional design is scalable, maintainable, and demonstrates best practices in software engineering.

## ✨ Key Features

- **Modular & Organized**: The codebase is logically separated into modules for configuration, model definition, agent logic, plotting, and experiment execution.

- **Parallel Experimentation**: Leverages Python's multiprocessing to run numerous independent experiments simultaneously, making it ideal for analyzing training variance or performing hyperparameter sweeps.

- **Centralized Configuration**: All experiment parameters, hyperparameters, and file paths are managed in a single, easy-to-edit `config.py` file.

- **Robust Orchestration**: An `ExperimentOrchestrator` class manages the entire lifecycle of the experiments, from launching parallel processes to collecting and saving results.

- **Comprehensive Analysis**: Automatically aggregates results from all runs into a single CSV file and generates a comparative plot showing the mean performance and standard deviation across all experiments.

## 📂 Project Structure

```
lunar-lander-framework/
│
├── dqn_agent/
│   ├── __init__.py           # Makes the directory a Python package
│   ├── agent.py              # Contains the DQNAgentIndependent class
│   ├── config.py             # All hyperparameters and configuration constants
│   ├── experiment.py         # The ExperimentOrchestrator and worker function
│   ├── model.py              # The QNetwork (PyTorch model) class
│   └── plotting.py           # The function for plotting results
│
├── models/                   # Directory to save trained model weights
│
├── run_experiments.py        # The main entry point to start the program
└── requirements.txt          # Project dependencies
```

## ⚙️ Setup and Installation

1. Clone the repository and create the directory structure as shown above.

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🏃‍♀️ How to Run

To start the training process, simply run the main entry point script from the root directory of the project (`lunar-lander-framework/`):

```bash
python run_experiments.py
```

The script will automatically:

- Read the settings from `dqn_agent/config.py`.
- Launch and manage the parallel experiments.
- Collect results as experiments complete.
- Save the aggregated results to `all_experiments_results.csv`.
- Generate a comparative performance plot at `all_experiments_plot.png`.
- Save the trained model weights for each run in the `models/` directory.

