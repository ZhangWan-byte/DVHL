import gym
from gym import Env
from gym import spaces
import numpy as np
from copy import deepcopy


class DREnv(Env):
    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.start = [0, 0]
        self.goal = [4, 4]
        self.count = 0
        self.current_state = None

        self.action_space = spaces.Discrete(4)

        # self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([4, 4]))
        # self.single_observation_space = 
        
    def step(self, action):
        self.count = self.count + 1
        new_state = deepcopy(self.current_state)
        if action == 0:  # up
            new_state[0] = max(new_state[0] - 1, 0)
        elif action == 1:  # down
            new_state[0] = min(new_state[0] + 1, self.cols - 1)
        elif action == 2:  # left
            new_state[1] = max(new_state[1] - 1, 0)
        elif action == 3:  # right
            new_state[1] = min(new_state[1] + 1, self.rows - 1)
        else:
            raise Exception("Invalid action")
        self.current_state = new_state

        if self.current_state[1] == self.goal[1] and self.current_state[0] == self.goal[0]:
            done = True
            reward = 10.0
        else:
            done = False
            reward = -1
        if self.count > 200:
            done = True

        info = {}
        return self.current_state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.count = 0
        self.current_state = self.start
        return self.current_state