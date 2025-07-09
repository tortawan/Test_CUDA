# dqn_agent/agent.py
import os
import random
import numpy as np
from collections import deque
from typing import Dict, Any, Deque

import torch
import torch.optim as optim
import torch.nn as nn

from .model import QNetwork
from .config import logger

class DQNAgentIndependent:
    """A self-contained DQN Agent for a single experiment."""
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any], agent_device: torch.device):
        self.config = config
        self.device = agent_device
        self.action_size = action_size
        self.memory: Deque = deque(maxlen=self.config['replay_memory_capacity'])
        self.epsilon = 1.0
        self.epsilon_min = 0.01

        self.model = QNetwork(
            state_size, 
            action_size, 
            config['first_hid'], 
            config['second_hid']
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.loss_fn = nn.MSELoss()

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add a new experience to memory."""
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

    def act(self, state_np: np.ndarray) -> int:
        """
        Returns actions for given state as per current policy.
        
        Args:
            state_np (np.ndarray): current state
            
        Returns:
            int: the chosen action
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        state_tensor = torch.from_numpy(state_np.flatten()).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(action_values[0]).item()

    def replay(self):
        """Experience replay to learn from a minibatch of experiences."""
        if len(self.memory) < self.config['batch_size']:
            return
            
        minibatch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states_tensor = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions_tensor = torch.tensor(actions, device=self.device).long().view(-1, 1)
        rewards_tensor = torch.tensor(rewards, device=self.device).float().view(-1, 1)
        next_states_tensor = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones_tensor = torch.tensor(dones, device=self.device).float().view(-1, 1)

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            next_q_values = self.model(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q_values = rewards_tensor + (self.config['gamma'] * next_q_values * (1 - dones_tensor))
            
        # Compute Q values for current states
        current_q_values = self.model(states_tensor).gather(1, actions_tensor)
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        """Update epsilon for epsilon-greedy policy."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.config['epsilon_decay']

    def save(self, filepath: str):
        """Save the model's state dictionary."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(self.model.state_dict(), filepath)
        except Exception as e:
            logger.error(f"Error saving model to {filepath}: {e}")