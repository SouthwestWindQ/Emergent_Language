import os
import random
import pickle
import gymnasium
import numpy as np

from gymnasium import spaces
from gymnasium.utils import seeding

from utils import state2str


def generate_state(now_dim, space_dim, state_dim, temp_state, all_state):
    if now_dim == space_dim:
        all_state.append(temp_state.copy())
        return
    for i in range(state_dim):
        temp_state[now_dim] = i
        generate_state(now_dim+1, space_dim, state_dim, temp_state, all_state)


def get_rule(space_dim, state_dim, rule_path):
    temp_state = np.zeros(shape=space_dim)
    all_state = []
    generate_state(0, space_dim, state_dim, temp_state, all_state)
    all_state = np.array(all_state, dtype=int)
    all_change = []
    for i in range(len(all_state)):
        single_change = [random.randint(0, state_dim-1) for _ in range(space_dim)]
        all_change.append(single_change)
    all_change = np.array(all_change, dtype=int)
    all_goal_state = (all_state + all_change) % state_dim
    rule = {}
    for i in range(len(all_state)):
        rule[state2str(all_state[i])] = all_goal_state[i]
    
    if not os.path.exists(os.path.split(rule_path)[0]):
        os.mkdir(os.path.split(rule_path)[0])
    with open(rule_path, "wb") as f:
        pickle.dump(rule, f)
    return rule


class Wire3Env(gymnasium.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, args):
        self.args = args
        self.action_space = spaces.MultiDiscrete([args.state_range for _ in range(args.state_dim)])
        self.observation_space = spaces.MultiDiscrete([args.state_range for _ in range(args.state_dim)])
        
        if args.state_dim == 3:
            self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        else:
            raise NotImplementedError   # TODO: How to design an environment for `args.state_dim != 3`?
        
        self.init_state = [random.randint(0, args.state_range-1) for _ in range(args.state_dim)]
        self.now_state = self.init_state
        self.now_step = 0
        self.lines = []
        self.right_lines = [0, 0, 0]
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
            for i in range(len(self.now_state)):
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
        self.right_lines = [0,0,0]
        for i in range(len(self.init_state)):
            if self.init_state[i] == self.goal_state[i]:
                self.right_lines[i] = 1
        return self.init_state

    def render(self, mode='human', close=False):
        if self.args.state_dim == 3:
            screen_width = 600
            screen_height = 400
            from gymnasium.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.Viewer(screen_width, screen_height)

                self.lines.append(rendering.FilledPolygon([(100, 100), (100, 300), (115, 100), (115, 300)]))
                self.lines.append(rendering.FilledPolygon([(300, 100), (300, 300), (315, 100), (315, 300)]))
                self.lines.append(rendering.FilledPolygon([(500, 100), (500, 300), (515, 100), (515, 300)]))

                for i in range(len(self.lines)):
                    color = self.colors[self.now_state[i]]
                    self.lines[i].set_color(color[0], color[1], color[2])
                    self.viewer.add_geom(self.lines[i])

            for i in range(len(self.lines)):
                color = self.colors[self.now_state[i]]
                self.lines[i].set_color(color[0], color[1], color[2])

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        else:
            raise NotImplementedError   # TODO: How to design an environment for `args.state_dim != 3`?

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
