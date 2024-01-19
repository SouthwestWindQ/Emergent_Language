import torch
import random
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from net import InsideAct, InsideState, OutsideModel

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'





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

        self.q_net_incom = InsideState(state_dim=state_dim, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        self.target_q_net_incom = InsideState(state_dim=state_dim, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        self.q_net_out = OutsideModel(state_dim=state_dim, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        self.target_q_net_out = OutsideModel(state_dim=state_dim, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        self.q_net_inact = InsideAct(output_dim=action_dim, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)
        self.target_q_net_inact = InsideAct(output_dim=action_dim, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)

        self.optimizer_incom = torch.optim.Adam(self.q_net_incom.parameters(), lr=lr)
        self.optimizer_out = torch.optim.Adam(self.q_net_out.parameters(), lr=lr)
        self.optimizer_inact = torch.optim.Adam(self.q_net_inact.parameters(), lr=lr)
        self.count = 0

    def take_action(self, state, goal_state):
        # use epsilon-greedy method to decide the agent's action
        if np.random.random() < self.epsilon:
            incom_symbol = np.array([np.random.randint(0, self.vocab_size)], dtype=np.int32)
            out_symbol = np.array([np.random.randint(0, self.vocab_size)], dtype=np.int32)
            action = np.random.randint(self.action_range)
            action = np.array(action)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            goal_state = torch.tensor(goal_state, dtype=torch.float).unsqueeze(0).to(self.device)
            incom_symbol_dist = self.q_net_incom(state)
            incom_symbol = torch.argmax(incom_symbol_dist, dim=-1)

            out_symbol_dist = self.q_net_out(incom_symbol, goal_state)
            out_symbol = torch.argmax(out_symbol_dist, dim=-1)

            action_dist = self.q_net_inact(out_symbol)
            action = torch.argmax(action_dist, dim=-1).cpu().numpy()
            incom_symbol = incom_symbol.cpu().numpy()
            out_symbol = out_symbol.cpu().numpy()
            if action.shape[0] == 1:
                action = action.squeeze(0)
            # print(f"take action = {(incom_symbol, out_symbol, action)}")
        return incom_symbol, out_symbol, action

    def update(self, incom_data, out_data, inact_data):

        states = torch.tensor(incom_data['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(incom_data['action'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(incom_data['reward'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(incom_data['next_state'], dtype=torch.float).to(self.device)
        goal_states = torch.tensor(incom_data['goal_state'], dtype=torch.float).to(self.device)
        dones = torch.tensor(incom_data['done'], dtype=torch.float).to(self.device)

        incom_q_values = self.q_net_incom(states)
        incom_q_values = incom_q_values.gather(1, actions)
        next_incom_q_values = self.target_q_net_incom(next_states)
        max_next_incom_q_values = torch.max(next_incom_q_values, dim=-1)
        incom_q_targets = rewards + self.gamma * max_next_incom_q_values[0] * (1 - dones)
        incom_q_targets = incom_q_targets.unsqueeze(-1)
        incom_loss = torch.mean(F.mse_loss(incom_q_values, incom_q_targets))
        self.optimizer_incom.zero_grad()
        incom_loss.backward()
        self.optimizer_incom.step()



        states = torch.tensor(out_data['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(out_data['action'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(out_data['reward'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(out_data['next_state'], dtype=torch.float).to(self.device)
        goal_states = torch.tensor(out_data['goal_state'], dtype=torch.float).to(self.device)
        dones = torch.tensor(out_data['done'], dtype=torch.float).to(self.device)

        # self.q_net.train()
        states = states.long().squeeze(-1)
        next_states = next_states.long().squeeze(-1)
        q_values = self.q_net_out(states, goal_states) # (Batch_size, action_dim, action_range)
        q_values = q_values.gather(1, actions).squeeze(-1)
        # print(f"process_q_values = {q_values}")
        next_q_values = self.target_q_net_out(next_states, goal_states) # (Batch_size, action_dim, action_range)
        # print(f"next_states = {next_states}")
        # print(f"goal_states = {goal_states}")
        # print(f"next_q_values = {next_q_values}")
        max_next_q_values = torch.max(next_q_values, dim=-1) # (Batch_size, aciton_dim)
        # rewards = rewards.unsqueeze(1).repeat(1, self.state_dim)
        # dones = dones.unsqueeze(1).repeat(1, self.state_dim)
        q_targets = rewards + self.gamma * max_next_q_values[0] * (1-dones)
        # print(f"q_targets = {q_targets}")
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # print(f"dqn_loss = {dqn_loss}")
        self.optimizer_out.zero_grad()
        dqn_loss.backward()
        # for name, paras in self.q_net.named_parameters():
        #     print(f"{name} grad = {paras.grad}")
        self.optimizer_out.step()
        # for name, paras in self.q_net.named_parameters():
        #     print(f"{name} grad = {paras.grad}")

        states = torch.tensor(inact_data['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(inact_data['action'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(inact_data['reward'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(inact_data['next_state'], dtype=torch.float).to(self.device)
        goal_states = torch.tensor(inact_data['goal_state'], dtype=torch.float).to(self.device)
        dones = torch.tensor(inact_data['done'], dtype=torch.float).to(self.device)

        states = states.long().squeeze(-1)
        next_states = next_states.long().squeeze(-1)
        inact_q_values = self.q_net_inact(states)
        inact_q_values = inact_q_values.gather(1, actions.unsqueeze(1))
        next_inact_q_values = self.target_q_net_inact(next_states)
        max_next_inact_q_values = torch.max(next_inact_q_values, dim=-1)
        # rewards = rewards.unsqueeze(1).repeat(1, self.state_dim)
        # dones = dones.unsqueeze(1).repeat(1, self.state_dim)
        inact_q_targets = rewards + self.gamma * max_next_inact_q_values[0] * (1 - dones)
        inact_loss = torch.mean(F.mse_loss(inact_q_values, inact_q_targets.unsqueeze(1)))
        self.optimizer_inact.zero_grad()
        inact_loss.backward()
        self.optimizer_inact.step()


        if self.count % self.target_update == 0:
            self.target_q_net_inact.load_state_dict(self.q_net_inact.state_dict())
            self.target_q_net_out.load_state_dict(self.q_net_out.state_dict())
            self.target_q_net_incom.load_state_dict(self.q_net_incom.state_dict())
        self.count += 1