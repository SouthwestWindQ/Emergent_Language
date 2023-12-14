import torch
import torch.nn as nn
import torch.nn.functional as F


class OutsideStateModel(nn.Module):
    """
    The model simulating the outside agent decoding the symbols
    uttered by the inside agent into the actions to be executed
    based on the goal states which are his private knowledge.
    """
    
    def __init__(self, output_dim, hidden_dim, state_dim, vocab_size, embed_size=16):
        super(OutsideStateModel, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.vocab_size = vocab_size
        self.embed = nn.Linear(in_features=vocab_size, out_features=embed_size, bias=False)
        self.fc1 = nn.Linear(in_features=embed_size+output_dim*state_dim, out_features=hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim*state_dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, input: torch.Tensor, goal_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): a tensor of size `(batch_size, vocab_size)`
            representing the uttered symbol by inside agent. Each row is an one-hot vector.
            
            goal_states (torch.Tensor): a tensor of size `(batch_size, self.state_dim)`
            representing the raw goal state. Each digit takes the value in the interval `[0, self.output_dim)`.

        Returns:
            torch.Tensor: a tensor of size `(batch_size, self.output_dim, self.state_dim)`
            representing the probability distribution of its predicted action that should be executed by inside agent.
        """
        
        # Firstly, get the corresponding embedding vectors of symbols, and concatenate it with one-hot goal state vectors.
        embed = self.embed(input)
        goal_states = F.one_hot(goal_states, num_classes=self.output_dim).reshape(goal_states.shape[0], -1).float() 
        input = torch.cat((embed, goal_states), dim=1)
        
        # Secondly, feed the concatenated vector into feed forward vector to get logits.
        hidden_state = self.act(self.bn2(self.fc2(self.act(self.bn1(self.fc1(input))))))
        output_dist = self.fc3(hidden_state).reshape(-1, self.output_dim, self.state_dim)
        return output_dist 


class OutsideComModel(nn.Module):
    """
    The model simulating the outside agent conveying the information of 
    actions to be executed to the inside agent through symbols.
    """
    
    def __init__(self, input_dim, input_range, hidden_dim, vocab_size):
        super(OutsideComModel, self).__init__()
        self.input_dim = input_dim
        self.input_range = input_range
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(in_features=input_dim*input_range, out_features=hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=vocab_size, bias=True)
        self.act = nn.ReLU()

    def forward(self, input):
        """
        Args:
            state (torch.Tensor): a tensor of size `(batch_size, self.output_dim, self.state_dim)`
            representing the action to be executed by the inside action, which should be encoded in this method.
            Each component of it is an one-hot vector.

        Returns:
            torch.Tensor: a tensor `(batch_size, vocab_size)`
            representing the predicted log-probability of each symbol.
        """
        
        input = input.reshape(-1, self.input_dim*self.input_range)
        hidden_state = self.act(self.bn2(self.fc2(self.act(self.bn1(self.fc1(input))))))
        output = self.fc3(hidden_state)
        return output
