import torch
import random
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from inside_agent import InsideAgentForInitState, InsideAgentForAction
from outside_agent import OutsideStateModel, OutsideComModel


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # a FIFO queue

    def add(self, state, action, reward, next_state, goal_state, done):  
        # insert a data tuple into the buffer
        self.buffer.append((state, action, reward, next_state, goal_state, done))

    def sample(self, batch_size):  
        # sample `batch_size` number of data tuples from the buffer
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, goal_state, done = [np.array(lst) for lst in zip(*transitions)]
        return state, action, reward, next_state, goal_state, done

    def size(self):  
        # return the current size of the buffer
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, state_range, action_range, vocab_size, device):
        super(Qnet, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_range = state_range
        self.action_range = action_range
        self.vocab_size = vocab_size
        self.device = device

        # transfer the state to symbol sending to outside_agent
        self.inside_state_net = InsideAgentForInitState(n_digits=state_dim, n_states_per_digit=state_range,
                                                       vocab_size=vocab_size)

        # transfer symbol from outside_agent to action
        self.inside_action_net = InsideAgentForAction(n_digits=action_dim, n_states_per_digit=action_range,
                                                     vocab_size=vocab_size)

        # transfer the symbol from inside_agent to state
        self.outside_state_net = OutsideStateModel(output_dim=state_dim, state_dim=state_range, hidden_dim=hidden_dim,
                                                  vocab_size=vocab_size)

        # transfer the action to the symbol sending to the inside_state
        self.outside_com_net = OutsideComModel(input_dim=action_dim, input_range=action_range, hidden_dim=hidden_dim, vocab_size=vocab_size)

    def forward(self, state, goal_state):
        state = state.long()
        
        # 1. Inside agent encodes the current state into a symbol.
        # inside_symbol_dist = torch.softmax(self.inside_state_net(state), dim=-1)
        # print(f"inside_symbol_dist = {inside_symbol_dist}")
        # inside_symbol_idx = torch.argmax(inside_symbol_dist, dim=-1, keepdim=True)
        inside_symbol_onehot = gumbel_softmax(self.inside_state_net(state))
        if len(inside_symbol_onehot.shape) == 1:
            inside_symbol_onehot = torch.unsqueeze(inside_symbol_onehot, dim=0)

        # print(f"inside_symbol_idx = {inside_symbol_idx}")
        # 2. Outside agent decodes the symbol uttered by inside agent into the state.
        outside_state_idx = gumbel_softmax(self.outside_state_net(inside_symbol_onehot, goal_state))
        # outside_state_idx = torch.argmax(outside_state_dist, dim=-1).float()
        # print(f"outside_state_dist = {outside_state_dist}")
        # print(f"outside_state_idx.shape = {outside_state_idx.shape}")

        # 3. Outside agent calculates the desired action and encodes it into a symbol.
        # outside_action = (goal_state - outside_state_idx + self.state_range) % self.state_range
        # outside_action = outside_state_idx
        # if state.shape[0] == 1:
        #     print(f"outside_state_dist = {outside_state_dist}")
        #     print(f"goal_state = {goal_state}")
        outside_symbol_idx = gumbel_softmax(self.outside_com_net(outside_state_idx))
        # print(f"outside_symbol_idx = {outside_symbol_idx}")
        # print(f"outside_symbol_idx.shape = {outside_symbol_idx.shape}")
        # print(f"outside_symbol_dist = {outside_symbol_dist}")
        # outside_symbol_idx = torch.argmax(outside_symbol_dist, dim=-1)
        # print(f"outside_symbol_idx = {outside_symbol_idx}")
        # outside_symbol_onehot = F.one_hot(outside_symbol_idx, num_classes=self.vocab_size, ).float().to(self.device)

        # print(f"outside_symbol_onehot = {outside_symbol_onehot}")
        # 4. Inside agent decodes the symbol uttered by inside agent into the action.
        inside_action_dist = self.inside_action_net(outside_symbol_idx)
        # print(f"inside_action_dist = {inside_action_dist}")
        # print(f"inside_action_dist = {inside_action_dist}")
        # print(f"inside_action_dist.shape = {inside_action_dist.shape}")
        return inside_action_dist


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, state_range, action_range, vocab_size, lr, gamma, epsilon,
                 target_update, device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_range = state_range
        self.action_range = action_range
        self.vocab_size = vocab_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.q_net = Qnet(state_dim, hidden_dim, action_dim, state_range, action_range, vocab_size, device).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim, state_range, action_range, vocab_size, device).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.count = 0

    def take_action(self, state, goal_state):
        # use epsilon-greedy method to decide the agent's action
        if self.epsilon > 0.05:
            self.epsilon = self.epsilon * 0.995
        if np.random.random() < self.epsilon:
            action = [np.random.randint(self.action_range) for _ in range(self.action_dim)]
            action = np.array(action)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            goal_state = torch.tensor(goal_state, dtype=torch.float).unsqueeze(0).to(self.device)
            action_dist = self.q_net(state, goal_state)
            action = torch.argmax(action_dist, dim=-1).cpu().numpy()
        return action

    def update(self, data):
        states = torch.tensor(data['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(data['action'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(data['reward'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(data['next_state'], dtype=torch.float).to(self.device)
        goal_states = torch.tensor(data['goal_state'], dtype=torch.float).to(self.device)
        dones = torch.tensor(data['done'], dtype=torch.float).to(self.device)

        # self.q_net.train()
        q_values = self.q_net(states, goal_states) # (Batch_size, action_dim, action_range)
        q_values = q_values.gather(2, actions.unsqueeze(2))
        next_q_values = self.target_q_net(next_states, goal_states) # (Batch_size, action_dim, action_range)
        max_next_q_values = torch.max(next_q_values, dim=-1) # (Batch_size, aciton_dim)
        rewards = rewards.unsqueeze(1).repeat(1, self.state_dim)
        dones = dones.unsqueeze(1).repeat(1, self.state_dim)
        q_targets = rewards + self.gamma * max_next_q_values[0] * (1-dones)
        # print(f"q_targets = {q_targets}")
        dqn_loss = torch.mean(F.mse_loss(q_values.view(-1), q_targets.view(-1)))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        # for name, paras in self.q_net.named_parameters():
        #     print(f"{name} grad = {paras.grad}")
        self.optimizer.step()
        # for name, paras in self.q_net.named_parameters():
        #     print(f"{name} grad = {paras.grad}")

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
