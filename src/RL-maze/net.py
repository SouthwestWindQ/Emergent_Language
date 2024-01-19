import torch
import torch.nn as nn
import torch.nn.functional as F


class OutsideModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, vocab_size, embed_size=32):
        super(OutsideModel, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.goal_embed_size = embed_size
        self.embed = nn.Embedding(embedding_dim=embed_size, num_embeddings=vocab_size)
        self.goal_embed = nn.Linear(in_features=state_dim, out_features=embed_size)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=embed_size*2, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, input, goal_state):
        vocab_embed = self.embed(input)
        goal_embed = self.goal_embed(goal_state)
        embed = torch.concat((vocab_embed, goal_embed), dim=-1)
        x = self.act(self.fc1(embed))
        x = self.fc2(x)
        return x


class InsideState(nn.Module):
    def __init__(self, state_dim, hidden_dim, vocab_size):
        super(InsideState, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(in_features=state_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        self.act = nn.ReLU()

    def forward(self, state):
        x = self.act(self.fc1(state))
        x = self.fc2(x)
        return x


class InsideAct(nn.Module):
    def __init__(self, output_dim, hidden_dim, vocab_size, embed_size=16):
        super(InsideAct, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(in_features=embed_size, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.act = nn.ReLU()

    def forward(self, input):
        embed = self.embed(input)
        x = self.act(self.fc1(embed))
        x = self.fc2(x)
        return x






