import torch
import torch.nn as nn


class OutsideStateModel(nn.Module):
    def __init__(self, output_dim, hidden_dim, state_dim, vocab_size, embed_size=64):
        super(OutsideStateModel, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(in_features=embed_size, out_features=hidden_dim, bias=True)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
        self.act = nn.ReLU()
        self.out_head = nn.Linear(in_features=1, out_features=state_dim, bias=True)  
        # TODO: Is it suitable to choose "1" here as the hidden state dimension?

    def forward(self, input):
        # input (Batch_size, input_dim=1)
        embed = self.embed(input).squeeze(1) # embed (Batch_size, embed_size)
        hidden_state = self.act(self.fc1(embed))
        output = self.act(self.fc2(hidden_state))
        
        # equivalent to parameter sharing but may improve efficiency?
        # one more thing: if we choose another number (larger than 1) as the hidden state dim, 
        # this operation will cause error, but so does the initial `for` loop.
        output_dist = self.out_head(output.unsqueeze(-1)) 
        output_dist = torch.softmax(output_dist, dim=-1)
        return output_dist # (Batch_size, output_dim, state_dim)


class OutsideComModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(OutsideComModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=vocab_size, bias=True)
        self.act = nn.ReLU()

    def forward(self, input):
        hidden_state = self.act(self.fc1(input))
        output = self.act(self.fc2(hidden_state))
        output_dist = torch.softmax(output, dim=-1)
        return output_dist
