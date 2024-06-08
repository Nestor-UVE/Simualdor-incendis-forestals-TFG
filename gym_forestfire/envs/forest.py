import numpy as np
import cv2
import time
from gym_forestfire.envs.presets import *
from gym_forestfire.envs.ros import compute_rate_of_spread
from gym_forestfire.envs.vent2 import Vent2

import matplotlib.pyplot as plt

# Simulation constants
CELL_SIZE = 10   # 3,048m 
TIME_STEP = 3 / 60 # minutes
GRID_SIZE = (64, 64)


# Fuel parameters
boscs_bages = TimberLitterUnderstory
tallgrass = Chaparral
# boscs_bages = Chaparral
particle = FuelParticle()

w_0 = boscs_bages.w_0
delta = boscs_bages.delta
M_x = boscs_bages.M_x
sigma = boscs_bages.sigma

h = particle.h
S_T = particle.S_T
S_e = particle.S_e
p_p = particle.p_p

M_f = 0.03
# U = 10.5 * 3.28084  # m/s   #usar 10.5 m/s
# U_dir = 0 # degrees
max_burn_time = 384/sigma / TIME_STEP 



class Forest:
    EMPTY_CELL = 0
    TREE_CELL = 1
    FIRE_CELL = 10
    BURNED_CELL = 11

    def __init__(self, world_size=(64, 64), p_fire=0.3, init_tree=1, extinguisher_ratio=0.05):
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
        self.seed = np.random.randint(-5, 5,2)
        self.mean_U = 0
        self.video_writer = None
        self.vent = Vent2(self.seed)
        

    def reset_grid(self):
        def init_other_fuel():
            w_0_array = np.full(GRID_SIZE, w_0, dtype=float)
            delta_array = np.full(GRID_SIZE, delta, dtype=float)
            M_x_array = np.full(GRID_SIZE, M_x, dtype=float)
            sigma_array = np.full(GRID_SIZE, sigma, dtype=float)
            # field is square [25,25] [39,39]
            field = (slice(25, 39), slice(25, 39))
            # w_0_array[field] = tallgrass.w_0
            # delta_array[field] = tallgrass.delta
            # M_x_array[field] = tallgrass.M_x
            # sigma_array[field] = tallgrass.sigma
            return w_0_array, delta_array, M_x_array, sigma_array
        

        grid = np.zeros(GRID_SIZE, dtype=[
            ('is_burning', bool), ('burned', bool), ('no_fuel', bool), ('time_burning', float), ('ros', float), ('fuel_w_0', float), ('fuel_delta', float), ('fuel_M_x', float), ('fuel_sigma', float), ('U', float), ('U_dir', float)
        ])
        grid['fuel_w_0'], grid['fuel_delta'], grid['fuel_M_x'], grid['fuel_sigma'] = init_other_fuel()
        grid['time_burning'] = 0
        grid['burned'] = False
        self.seed = np.random.randint(-5, 5, 2)
        grid['U'], grid['U_dir'] = self.vent.reset(self.seed)

        return grid

    def init_grid(self, grid):
        grid['no_fuel'] = self.world == self.EMPTY_CELL
        grid['is_burning'] = self.world == self.EMPTY_CELL
        grid['is_burning'] = self.world == self.FIRE_CELL
        grid['burned'] = self.world == self.BURNED_CELL
        return grid

    def grid_to_world(self, new_grid):
        self.world = np.where(new_grid['is_burning'], self.FIRE_CELL, self.world)
        self.world = np.where(np.logical_and(~new_grid['is_burning'], ~new_grid['burned'], ~new_grid['no_fuel']), self.TREE_CELL, self.world)
        self.world = np.where(new_grid['no_fuel'], self.EMPTY_CELL, self.world)
        self.world = np.where(new_grid['burned'], self.BURNED_CELL, self.world)

        return self.world

    def update(self, grid):
        grid = self.init_grid(grid)
        new_grid = grid.copy()
        rows, cols = GRID_SIZE
        max_row, max_col = rows - 1, cols - 1

        #update values of wind U and U_dir makeing a simple simulation of wind, it will change the direction and speed of the wind
        

        for i in range(rows):
            for j in range(cols):
                cell = grid[i, j]
                #Comprovem només cel·les amb foc
                if cell['is_burning']:
                    #Cal comprovar el fuel, ja que pot ser mitigada per l'agent. 
                    if cell['time_burning'] > max_burn_time and not cell['no_fuel']:
                        new_grid[i, j]['is_burning'] = False
                        new_grid[i, j]['burned'] = True
                    else: 
                        new_grid[i, j]['time_burning'] += TIME_STEP
                    # Si ha passat temps per propagar-se i encara crema
                    if cell['time_burning'] * cell['ros'] >= CELL_SIZE and cell['is_burning']:
                        # Veïns dins dels límits
                        for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                            if 0 <= ni <= max_row and 0 <= nj <= max_col:
                                neighbor = grid[ni, nj]
                                # Si el veí no crema, ni ha cremat, i té fuel
                                if not neighbor['is_burning'] and not neighbor['burned'] and not neighbor['no_fuel']:
                                    r = compute_rate_of_spread(
                                        i, j, ni, nj, cell['fuel_w_0'], cell['fuel_delta'], cell['fuel_M_x'], cell['fuel_sigma'],
                                        h, S_T, S_e, p_p, M_f, cell['U'], cell['U_dir']
                                    )
                                    new_grid[ni,nj]['ros'] = r
                                    new_grid[ni,nj]['is_burning'] = True
        self.grid_to_world(new_grid)
        return new_grid

    def step(self, action=None):
        aimed_fire = False
        num_trees = 0
        border = False
        self.step_counter += 1
        burned = False
        done = False
        if action is not None:
            action = (action + 1) / 2  # Normalize action to [0, 1]
            x, y = int(self.world.shape[1] * action[1]), int(self.world.shape[0] * action[0])
            w, h = int(self.world.shape[1] * self.extinguisher_ratio), int(self.world.shape[0] * self.extinguisher_ratio)
            x_1, x_2 = max(0, int(x - w / 2)), min(self.world.shape[1], int(x + w / 2))
            y_1, y_2 = max(0, int(y - h / 2)), min(self.world.shape[0], int(y + h / 2))
            self.action_rect = [(x_1, y_1), (x_2, y_2)]

            burned = np.any(self.burned[y_1:y_2, x_1:x_2])
            
            if x_1 == 0 or x_2 == self.world.shape[1] or y_1 == 0 or y_2 == self.world.shape[0]:
                border = True

            aimed_fire = np.any(self.fire[y_1:y_2, x_1:x_2])
            num_trees = np.sum(self.world[y_1:y_2, x_1:x_2] == self.TREE_CELL)
            self.world[y_1:y_2, x_1:x_2] = np.where(self.burned[y_1:y_2, x_1:x_2], self.world[y_1:y_2, x_1:x_2], self.EMPTY_CELL)
        else:
            self.action_rect = None

        self.fire = self.world == self.FIRE_CELL
        self.tree = self.world == self.TREE_CELL
        self.empty = self.world == self.EMPTY_CELL
        self.burned = self.world == self.BURNED_CELL
        if (
            np.any(self.fire[0]) 
            or np.any(self.fire[-1]) 
            or np.any(self.fire[:, 0]) 
            or np.any(self.fire[:, -1])
        ):
            
            done = True
        is_fire = np.any(self.fire)

        if not is_fire:

            # start_point = self.world.shape[0] // 2, self.world.shape[1] // 2
            # start_point = np.random.randint(0, self.world.shape[0]), np.random.randint(0, self.world.shape[1])
            #start point is somewhere within [20,20] [44,44]
            start_point = np.random.randint(20, 44), np.random.randint(20, 44)
            self.world[start_point] = self.FIRE_CELL
            self.grid['is_burning'][start_point] = True
            self.grid['ros'] = 1000

        if is_fire:
            if self.step_counter % 30 == 0:
                self.grid['U'], self.grid['U_dir'], self.mean_U = self.vent.step(self.seed)
            self.grid = self.update(self.grid)

        #fire_cells counts the number of burning cells in the grid
        fire_cells = np.sum(self.fire)

        return aimed_fire, is_fire, num_trees, border, fire_cells, burned, done

    def reset(self):
        self.grid = self.reset_grid()
        self.grid_to_world(self.grid)
        self.step_counter = 0
        self.seed = np.random.randint(-5, 5, 2)
        self.grid['U'], self.grid['U_dir'], self.mean_U = self.vent.step(self.seed)
        self.video_writer = None
  



    def render(self, name):
        # Create two separate images for wind values and other values
        im_other = cv2.cvtColor(self.world, cv2.COLOR_GRAY2RGB)
        
        # Set color for fuel_w_0 = 0.5 to yellow
        # Set color for trees to green
        im_other[self.tree] = (80, 180, 60)
        im_other[self.grid['fuel_sigma'] == tallgrass.sigma] = (0, 240, 240)
        im_other[self.fire] = (0, 0, 200)
        im_other[self.burned] = (105, 105, 105)
        im_other[self.empty] = (0, 25, 60)
        if self.action_rect is not None:
            cv2.rectangle(im_other, self.action_rect[0], self.action_rect[1], (255, 255, 255), 1)
        
        # Generate wind vectors
        X, Y = np.meshgrid(np.arange(64), np.arange(64))
        U = self.grid['U']
        U_dir = self.grid['U_dir']
        U_x = U * np.cos(np.radians(U_dir))
        U_y = U * np.sin(np.radians(U_dir))
        
        # Plot wind vectors using quiver plot
        plt.figure(figsize=(10, 10))
        plt.quiver(X, Y, U_x, U_y, angles='xy', scale = 2000)
        plt.text(0.9, 0.96, f"Velocitat mitja del vent: {self.mean_U*0.911:.2f} km/h ", color='black', fontsize=18, ha='right', va='top', transform=plt.gcf().transFigure)

        plt.savefig("quiver_plot.png")
        plt.close()  # Close the plot to prevent displaying it

        # self.vent.plot(self.grid['U'], self.mean_U)
        
        im_wind = cv2.imread("quiver_plot.png")
        im_wind = cv2.cvtColor(im_wind, cv2.COLOR_BGR2RGB)
        im_wind_resized = cv2.resize(im_wind, (640, 640))
        im_other_resized = cv2.resize(im_other, (640, 640))

        combined_image = np.hstack((im_wind_resized, im_other_resized))

        # Initialize video writer if not already initialized
        # if self.video_writer is None:
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     self.video_writer = cv2.VideoWriter(f"anim/animation_{name}.avi", fourcc, 6.5, (combined_image.shape[1], combined_image.shape[0]))

        # # # # Write the current frame to the video
        # self.video_writer.write(combined_image)
        
        # Display the combined image (optional)
        cv2.imshow("Forest", combined_image)
        cv2.waitKey(1000//200)


if __name__ == "__main__":
    forest = Forest(world_size=(64, 64))
    forest.reset()  
    n_steps = 1800
    for number in range(10):
        print(f'Number {number}')
        for i in range(n_steps):
            _, _, _, _, _, _, done = forest.step()
            if done:
                break
            if i % 20 == 0 or i == 0:
                forest.render(number)
            print(f'Step {i}')
        forest.reset()


