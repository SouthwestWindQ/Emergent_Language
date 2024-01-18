import os
import random
import pickle
import gymnasium
import itertools
import numpy as np

from gymnasium import spaces
from gymnasium.utils import seeding

from utils import state2str


def get_rule(space_dim, state_dim, rule_path):
    all_state = np.array(list(itertools.product(np.arange(state_dim), repeat=space_dim)), dtype=np.int64)
    all_change = np.random.randint(0, state_dim, size=(len(all_state), space_dim), dtype=np.int64)
    all_goal_state = (all_state + all_change) % state_dim
    rule = {state2str(state): all_goal_state[i] for i, state in enumerate(all_state)}
    
    if not os.path.exists(os.path.split(rule_path)[0]):
        os.mkdir(os.path.split(rule_path)[0])
    with open(rule_path, "wb") as f:
        pickle.dump(rule, f)
    
    return rule


class Wire3Env(gymnasium.Env):
    metadata = {
        'video.frames_per_second': 2
    }

    def __init__(self, args):
        self.args = args
        self.action_space = spaces.MultiDiscrete([args.state_range for _ in range(args.state_dim)])
        self.observation_space = spaces.MultiDiscrete([args.state_range for _ in range(args.state_dim)]) 
        
        self.init_state = [random.randint(0, args.state_range-1) for _ in range(args.state_dim)]
        self.now_state = self.init_state
        self.now_step = 0
        self.right_lines = [0] * self.args.state_dim
        self.viewer = None
        
        self.rule_path = os.path.join(args.rule_path, f'rule_dim{self.args.state_dim}_range{self.args.state_range}')
        if os.path.exists(self.rule_path):
            with open(self.rule_path, "rb") as f:
                self.rule = pickle.load(f)
        else:
            self.rule = get_rule(space_dim=args.state_dim, state_dim=args.state_range, rule_path=self.rule_path)
        self.goal_state = self.rule[state2str(self.init_state)]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.now_state = (self.now_state + action) % self.args.state_range
        done = False
        reward = -1
        
        if np.all(self.now_state == self.goal_state):
            done = True
            reward = 10
            return self.now_state, reward, done
        else:
            for i in range(self.args.state_dim):
                if (self.now_state[i] == self.goal_state[i]) and self.right_lines[i] == 0:
                    self.right_lines[i] = 1
                    reward += 1
        
        if self.now_step >= 200:
            done = True
            reward = 0

        self.now_step += 1

        return self.now_state, reward, done

    def reset(self):
        self.now_state = self.init_state
        return self.now_state

    def remake(self):
        self.now_step = 0
        self.init_state = [random.randint(0, self.args.state_range-1) for _ in range(self.args.state_dim)]
        self.goal_state = self.rule[state2str(self.init_state)]
        self.now_state = self.init_state
        self.right_lines = [int(self.init_state[i] == self.goal_state[i]) for i in range(self.args.state_dim)]
        return self.init_state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
