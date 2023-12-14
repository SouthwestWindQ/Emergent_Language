import torch
import torch.nn as nn
import torch.nn.functional as F


class OutsideStateModel(nn.Module):
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
        # self.out_head = nn.Linear(in_features=8, out_features=state_dim, bias=True)
        # TODO: Is it suitable to choose "1" here as the hidden state dimension?

    def forward(self, input, goal_states):
        embed = self.embed(input)
        goal_states = F.one_hot(goal_states, num_classes=self.output_dim).reshape(goal_states.shape[0], -1).float() 
        input = torch.cat((embed, goal_states), dim=1)
        
        hidden_state = self.act(self.bn2(self.fc2(self.act(self.bn1(self.fc1(input))))))
        output_dist = self.fc3(hidden_state).reshape(-1, self.output_dim, self.state_dim)
        return output_dist # (Batch_size, output_dim, state_dim)


class OutsideComModel(nn.Module):
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
        input = input.reshape(-1, self.input_dim*self.input_range)
        hidden_state = self.act(self.bn2(self.fc2(self.act(self.bn1(self.fc1(input))))))
        output = self.fc3(hidden_state)
        return output
