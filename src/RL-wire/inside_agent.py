# inside_agent.py
# 
# Define models simulating the behavior of the inside agent.
# It accomplishes the following two tasks:
# (1) Taking the initial state of the lock as inputs, output the predictions of symbols;
# (2) Taking the symbol uttered by the agent outside, output the predictions of actions.

import torch
import torch.nn as nn
import torch.nn.functional as F


class InsideAgentForInitState(nn.Module):
    """
    The model simulating the inside agent conveying the information of 
    initial state to the outside agent through a symbol.
    """
    
    def __init__(self, n_digits: int, n_states_per_digit: int, vocab_size: int, latent: int=32):
        super(InsideAgentForInitState, self).__init__()
        self.n_digits = n_digits 
        self.n_states_per_digit = n_states_per_digit
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(n_states_per_digit*n_digits, latent)
        self.fc2 = nn.Linear(latent, latent)
        self.fc3 = nn.Linear(latent, vocab_size)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state (torch.Tensor): a `n_digits`-dimensional 1D tensor 
            representing the initial state.

        Returns:
            torch.Tensor: a `vocab_size`-dimensional 1D tensor 
            representing the predicted log-probability of each symbol.
        """

        state = F.one_hot(state, num_classes=self.n_states_per_digit).reshape(state.shape[0], -1).float()  
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class InsideAgentForAction(nn.Module):
    """
    The model simulating the inside agent receiving the symbol uttered by 
    the outside agent and deciding its action.
    """
    
    def __init__(self, n_digits: int, n_states_per_digit: int, vocab_size: int, latent: int=64):
        super(InsideAgentForAction, self).__init__()
        self.n_digits = n_digits   
        self.n_states_per_digit = n_states_per_digit
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(vocab_size, latent)
        self.fc2 = nn.Linear(latent, latent)
        self.fc3 = nn.Linear(latent, n_states_per_digit * n_digits)
        
    def forward(self, x):
        """
        Args:
            state (torch.Tensor): a one-hot tensor  of size (1, `vocab_size`)
            representing the symbol uttered by the outside agent.

        Returns:
            torch.Tensor: a 3D tensor of size (1, `n_digits`, `n_states_per_digit`)
            where each component represents the predicted log-probability of 
            each modified digit.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # print(f"shape = {x.shape}")
        x = self.fc3(x).reshape(x.shape[0], self.n_digits, -1)
        return x
