import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import random
import pickle
import os
import time

'''
GridWorld(world_size, wall_prob)
world_size: the size of gridworld, the index will be in [0, world_size - 1]
wall_prob: the prob of wall generation for every possible position and direction


Observation space = (2,) [x,y] for position
Action space = (4,) 0 up, 1 down, 2 left, 3 right
Wall space = (world_size, world_size, 4) wall[i][j][k] refers to the direction k of position (i,j),
the meaning of k is: 0 up, 1 down, 2 left, 3 right

Max_steps = 100

reset: reset init position and goal position, do not change the wall position and the gridworld

remake: reset init position and goal position, and change the wall position and the gridworld

wall_generation: randomly generate the wall of gridworld

valid_check: check if there is a path from init pos to goal pos


reward setting: -1 for every step, -5 if go out of the gridworld or try to cross the wall, +5 for achieve goal

done setting: done for true if achieve goal or timeout.
'''


class GridWorld(gymnasium.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    def __init__(self, world_size = 4, wall_prob=0.1):
        # 0 up 1 down 2 left 3 right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=world_size-1, shape=(2,), dtype=np.int32)
        self.wall_space = spaces.Box(low=0, high=1, shape=(world_size,world_size,4), dtype=np.int32)
        self.world_size = world_size
        self.wall_prob = wall_prob
        self.init_pos = np.array([random.randint(0, world_size), random.randint(0,world_size)])
        self.now_pos = self.init_pos
        self.goal_pos = np.array([random.randint(0, world_size), random.randint(0, world_size)])
        while (self.init_pos[0] == self.goal_pos[0]) and (self.init_pos[1] == self.goal_pos[1]):
            self.goal_pos = [random.randint(0, world_size), random.randint(0, world_size)]

        self.dir = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.int32)
        self.wall = self.wall_generation()
        while self.valid_check() == False:
            self.wall = self.wall_generation()
        self.now_step = 0
        self.max_step = 100

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        new_pos = self.now_pos + self.dir[action]
        done = False
        reward = -1
        self.now_step += 1
        if self.now_step >= self.max_step:
            return self.now_pos, reward, True
        # assert (new_pos >= 0) and (new_pos < self.world_size), "action go out of the gridworld!"
        if (new_pos[0] >= 0) and (new_pos[0] < self.world_size) and (new_pos[1] >= 0) and (new_pos[1] < self.world_size) and (self.wall[self.now_pos[0]][self.now_pos[1]][action] == 0):
            if (new_pos[0] == self.goal_pos[0]) and (new_pos[1] == self.goal_pos[1]):
                done = True
                reward += 5
            self.now_pos = new_pos
            return self.now_pos, reward, done
        else:
            reward -= 5
            return self.now_pos, reward, done

    def reset(self):
        self.init_pos = np.array([random.randint(0, self.world_size), random.randint(0, self.world_size)])
        self.now_pos = self.init_pos
        self.goal_pos = np.array([random.randint(0, self.world_size), random.randint(0, self.world_size)])
        while (self.init_pos[0] == self.goal_pos[0]) and (self.init_pos[1] == self.goal_pos[1]):
            self.goal_pos = [random.randint(0, self.world_size), random.randint(0, self.world_size)]
        self.now_step = 0
        return self.now_pos

    def remake(self):
        self.init_pos = np.array([random.randint(0, self.world_size), random.randint(0, self.world_size)])
        self.now_pos = self.init_pos
        self.goal_pos = np.array([random.randint(0, self.world_size), random.randint(0, self.world_size)])
        while (self.init_pos[0] == self.goal_pos[0]) and (self.init_pos[1] == self.goal_pos[1]):
            self.goal_pos = [random.randint(0, self.world_size), random.randint(0, self.world_size)]
        self.wall = self.wall_generation()
        while self.valid_check() == False:
            self.wall = self.wall_generation()
        self.now_step = 0
        return self.now_pos


    def wall_generation(self):
        wall_sample = np.zeros(shape=(self.world_size, self.world_size, 4), dtype=np.int32)
        for i in range(self.world_size):
            for j in range(self.world_size):
                for k in range(4):
                    if np.random.uniform() < self.wall_prob:
                        wall_sample[i][j][k] = 1
        return wall_sample

    def valid_check(self):
        # print(f"init_pos = {self.init_pos}")
        now_pos = self.init_pos.copy()
        final_pos = self.goal_pos.copy()
        visited_map = np.zeros(shape=[self.world_size, self.world_size], dtype=np.int32)
        nodes = [now_pos.copy()]
        while len(nodes) != 0:
            now_pos = nodes.pop(0)
            for i in range(4):
                next_pos = now_pos + self.dir[i]
                if self.wall[now_pos[0]][now_pos[1]][i] == 1:
                    continue
                if (next_pos[0] < 0) or (next_pos[0] >= self.world_size) or (next_pos[1] < 0) or (next_pos[1] >= self.world_size):
                    continue
                if visited_map[next_pos[0]][next_pos[1]] == 1:
                    continue

                if next_pos[0] == final_pos[0] and next_pos[1] == final_pos[1]:
                    return True
                nodes.append(next_pos.copy())

        return False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        return NotImplementedError


if __name__ == "__main__":


    env = GridWorld()


