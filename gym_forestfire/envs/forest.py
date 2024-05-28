import numpy as np
import cv2
import time
from gym_forestfire.envs.presets import *
from gym_forestfire.envs.ros import compute_rate_of_spread

import matplotlib.pyplot as plt

# Simulation constants
CELL_SIZE = 10
TIME_STEP = 10 / 60 # minutes
GRID_SIZE = (64, 64)


# Fuel parameters
shortgrass = ShortGrass
chaparral = HeavyLoggingSlash
grass = TallGrass
particle = FuelParticle()

w_0 = chaparral.w_0
delta = chaparral.delta
M_x = chaparral.M_x
sigma = chaparral.sigma

h = particle.h
S_T = particle.S_T
S_e = particle.S_e
p_p = particle.p_p

M_f = 0.03
U = 3 * 3.28084  # m/s
U_dir = 0
max_burn_time = 384/sigma * 10


class Forest:
    EMPTY_CELL = 0
    TREE_CELL = 1
    FIRE_CELL = 10

    def __init__(self, world_size=(64, 64), p_fire=0.3, init_tree=0.995, extinguisher_ratio=0.05):
        self.p_fire = p_fire
        self.p_init_tree = init_tree
        self.extinguisher_ratio = extinguisher_ratio
        self.world_size = world_size

        # Initialize grid
        full_size = tuple(i + 2 for i in world_size)
        self.full = np.zeros(full_size, dtype=np.uint8)
        nd_slice = (slice(1, -1),) * len(world_size)
        self.world = self.full[nd_slice]
        self.n_dims = len(self.world.shape)
        self.sum_over = tuple(-(i + 1) for i in range(self.n_dims))
        self.step_counter = 0
        self.action_rect = None

    def reset_grid(self):
        def init_no_fuel():
            no_fuel_matrix = np.zeros(GRID_SIZE, dtype=bool)
            # no_fuel_matrix[48:52, :35] = True  # Riu
            # no_fuel_matrix[35:65, 35:40] = True #Pedra
            no_fuel_matrix[np.random.choice([True, False], GRID_SIZE, p=[1-self.p_init_tree, self.p_init_tree])] = True
            return no_fuel_matrix

        grid = np.zeros(GRID_SIZE, dtype=[
            ('is_burning', bool), ('burned', bool), ('no_fuel', bool), ('time_burning', float), ('ros', float)
        ])

        grid['no_fuel'] = init_no_fuel()
        grid['time_burning'] = 0
        grid['burned'] = False

        return grid

    def init_grid(self, grid):
        grid['no_fuel'] = self.world == self.EMPTY_CELL

    def grid_to_world(self, new_grid):
        self.world = np.where(new_grid['is_burning'], self.FIRE_CELL, self.world)
        self.world = np.where(np.logical_and(~new_grid['is_burning'], ~new_grid['burned'], ~new_grid['no_fuel']), self.TREE_CELL, self.world)
        self.world = np.where(new_grid['no_fuel'], self.EMPTY_CELL, self.world)
        self.world = np.where(new_grid['burned'], self.EMPTY_CELL, self.world)

    def update(self, grid):
        self.init_grid(grid)
        # #plot grid
        # plt.imshow(grid['no_fuel'], cmap='hot', interpolation='nearest')
        # plt.show()
        new_grid = grid.copy()
        
        rows, cols = GRID_SIZE
        
        # Precompute boundaries
        max_row, max_col = rows - 1, cols - 1

        for i in range(rows):
            for j in range(cols):
                cell = grid[i, j]
                updated_cell = new_grid[i, j]
                if cell['is_burning']:
                    updated_cell['time_burning'] += TIME_STEP
                    if updated_cell['time_burning'] > max_burn_time:
                        updated_cell['is_burning'] = False
                        updated_cell['burned'] = True
                    if updated_cell['time_burning'] * cell['ros'] >= CELL_SIZE:
                        for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                            if 0 <= ni <= max_row and 0 <= nj <= max_col:
                                neighbor = new_grid[ni, nj]
                                if not neighbor['is_burning'] and not neighbor['burned'] and not neighbor['no_fuel']:
                                    r = compute_rate_of_spread(
                                        i, j, ni, nj, w_0, delta, M_x, sigma,
                                        h, S_T, S_e, p_p, M_f, U, U_dir
                                    )
                                    neighbor['ros'] = r
                                    neighbor['is_burning'] = True
        self.grid_to_world(new_grid)
        return new_grid

    def step(self, action=None):
        aimed_fire = False
        num_trees = 0
        border = False
        num_fire = 0
        self.step_counter += 1

        if action is not None:
            action = (action + 1) / 2  # Normalize action to [0, 1]
            x, y = int(self.world.shape[1] * action[1]), int(self.world.shape[0] * action[0])
            w, h = int(self.world.shape[1] * self.extinguisher_ratio), int(self.world.shape[0] * self.extinguisher_ratio)
            x_1, x_2 = max(0, int(x - w / 2)), min(self.world.shape[1], int(x + w / 2))
            y_1, y_2 = max(0, int(y - h / 2)), min(self.world.shape[0], int(y + h / 2))
            self.action_rect = [(x_1, y_1), (x_2, y_2)]

            if x_1 == 0 or x_2 == self.world.shape[1] or y_1 == 0 or y_2 == self.world.shape[0]:
                border = True

            aimed_fire = np.any(self.fire[y_1:y_2, x_1:x_2])
            num_trees = np.sum(self.world[y_1:y_2, x_1:x_2] == self.TREE_CELL)
            num_fire = np.sum(self.world[y_1:y_2, x_1:x_2] == self.FIRE_CELL)
            self.world[y_1:y_2, x_1:x_2] = self.EMPTY_CELL
        else:
            self.action_rect = None

        self.fire = self.world == self.FIRE_CELL
        self.tree = self.world == self.TREE_CELL
        self.empty = self.world == self.EMPTY_CELL

        is_fire = np.any(self.fire)

        if not is_fire:
            # start_point = self.world.shape[0] // 2, self.world.shape[1] // 2
            start_point = np.random.randint(0, self.world.shape[0]), np.random.randint(0, self.world.shape[1])
            self.world[start_point] = self.FIRE_CELL
            self.grid['is_burning'][start_point] = True
            self.grid['ros'] = 1000

        if is_fire and self.step_counter % 1 == 0:
            self.grid = self.update(self.grid)

        return aimed_fire, is_fire, num_trees, border, num_fire

    def reset(self):
        self.grid = self.reset_grid()
        self.grid_to_world(self.grid)
        self.step_counter = 0

    def render(self):
        im = cv2.cvtColor(self.world, cv2.COLOR_GRAY2BGR)
        im[self.tree, 1] = 255
        im[self.fire, 2] = 170
        if self.action_rect is not None:
            cv2.rectangle(im, self.action_rect[0], self.action_rect[1], (255, 255, 255), 1)
        im = cv2.resize(im, (640, 640))
        cv2.imshow("Forest", im)
        cv2.waitKey(50)


if __name__ == "__main__":
    forest = Forest(world_size=(64, 64))
    forest.reset()
    n_steps = 100
    
    total_time = 0
    n_steps = 250
    average = 10
    for _ in range(average):
        start = time.time()
        for i in range(n_steps):
            forest.step()
            forest.render()
        forest.reset()

        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

    average_time = total_time / average
    print(f'Average elapsed time: {average_time}', 0.306)