# dqn_agent/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Neural Network for Q-value approximation."""
    def __init__(self, state_size: int, action_size: int, first_hid: int, second_hid: int):
        """
        Initializes the Q-Network.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            first_hid (int): Number of nodes in the first hidden layer.
            second_hid (int): Number of nodes in the second hidden layer.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, first_hid)
        self.fc2 = nn.Linear(first_hid, second_hid)
        self.fc3 = nn.Linear(second_hid, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Build a network that maps state -> action values.
        
        Args:
            state (torch.Tensor): The state tensor.

        Returns:
            torch.Tensor: The tensor of Q-values for each action.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)