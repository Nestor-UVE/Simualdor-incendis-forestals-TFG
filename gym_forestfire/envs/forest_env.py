"""
Copyright 2020 Sahand Rezaei-Shoshtari. All Rights Reserved.

Forest-fire gym environment.

Author: Sahand Rezaei-Shoshtari
"""

import numpy as np
import gym
import cv2
from gym import spaces
from gym.utils import seeding

from gym_forestfire.envs.forest import Forest


STATE_W = 64
STATE_H = 64
T_HORIZON = 1500


class ForestFireEnv(gym.Env):

    def __init__(self, **env_kwargs):
        self.seed()
        self.reward = 0
        self.state = None
        self.t = 0
        self.some_aimed = False

        self.forest = Forest(**env_kwargs)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=64, shape=(STATE_H, STATE_W), dtype=np.uint8
        )
        self._max_episode_steps = T_HORIZON

    def step(self, action):
        aimed_fire, is_fire, num_trees, border, fire_cells, burned, _ = self.forest.step(action)
        self.t += 1
        step_reward = 0
        done = bool(self.t > T_HORIZON or not is_fire)
        #if any fire is in a border then done
        
        # reward calculation
        # # if the action has been aimed at fire: add 1 to the reward
        if aimed_fire:
            step_reward += 0.1 
            self.some_aimed = True
        if border:
            step_reward -= 0.05


        if burned:
            step_reward -= 0.2

        # every 10 batch_i if fire_cells is less than at 0 batch_i, then reward is -1
        # if self.batch_i % 20 == 0:
        #     self.store_fire_cells = fire_cells
        # if self.batch_i % 20 == 9:
        #     if fire_cells < self.store_fire_cells and self.some_aimed:
        #         # print("A bit less fire")
        #         step_reward += 0.1
        #     self.batch_i = -1
        
        # self.batch_i += 1 

        # if fire exists but the action has done nothing: subtract 1 from the reward
        # if not aimed_fire and is_fire:
        #     step_reward -= 0.1
        # #count number of trees destroyed
        # step_reward -= (num_trees)*0.1        
        # if done:
        #     # if np.mean(self.forest.world) > 0.4 * self.forest.p_init_tree and self.some_aimed:
        #     #     step_reward += 1
        #     # else:
        #     #     step_reward -= 1
        #     if np.mean(self.forest.world) > 0.75 * self.forest.p_init_tree and self.some_aimed:
        #         step_reward += 1
            if np.mean(self.forest.world) > 0.9 * self.forest.p_init_tree and self.some_aimed:
                step_reward += 1
           
            # if np.mean(self.forest.world) > 0.4 and self.some_aimed:
            #     step_reward += 10 * (np.mean(self.forest.world)-0.4)**2
            # else:
            #     step_reward -= 5 * (0.4-np.mean(self.forest.world))**2
            # if self.some_aimed:
            #     step_reward += 50 * (np.mean(self.forest.world)-0.4)
            #     if np.mean(self.forest.world) < 0.4:
            #         step_reward -= 50
        #     # if self.t <= 3 and not aimed_fire:
        #     #     step_reward = -10 
        #     step_reward = step_reward / self.t
            # print(f"Some aimed: {self.some_aimed}")
            # if not self.some_aimed:
            #     step_reward -= -10
            # if step_reward > 0:
            #     step_reward = step_reward / self.t * 10
                

        
        #Fer que contra menys steps mes punts, també com més arbres
        self.reward = step_reward

        state = self.forest.world
        if state.shape != (STATE_H, STATE_W):
            state = self._scale(state, STATE_H, STATE_W)
        self.state = np.array(state) / self.forest.FIRE_CELL

        #count number of trees
        num_trees = np.sum(self.forest.tree)

        return self.state, step_reward, done, num_trees, {}

    def reset(self):
        self.forest.reset()
        self.reward = 0
        self.t = 0
        self.some_aimed = False
        self.batch_i = 0

        return self.step(None)[0]

    def render(self, name, mode="human"):
        self.forest.render(name)

    def close(self):
        cv2.destroyAllWindows()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _scale(self, im, height, width):
        original_height, original_width = im.shape
        return [
            [
                im[int(original_height * r / height)][int(original_width * c / width)]
                for c in range(width)
            ]
            for r in range(height)
        ]
