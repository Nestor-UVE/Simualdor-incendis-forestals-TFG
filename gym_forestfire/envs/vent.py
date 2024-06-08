import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class Vent:

    def __init__(self, seed, width=64, height=64, scale=35, viscosity=0.1, dt=0.1):
        self.width = width
        self.height = height
        self.scale = scale
        self.viscosity = viscosity
        self.dt = dt
        self.previous_random_u = np.random.uniform(low=-5, high=5)
        self.previous_random_v = np.random.uniform(low=-5, high=5)
        self.Vx_disturbance = self.generate_smoothed_random_field(seed[0])
        self.Vy_disturbance = self.generate_smoothed_random_field(seed[1])  # Different seed for Vy_disturbance

    def generate_smoothed_random_field(self, seed):
        """Generates a smoothed random field simulating wind speed."""
        # np.random.seed(seed)  # Ensures reproducibility
        random_field = np.random.normal(loc=seed, scale=0.5, size=(self.width, self.height))
        smoothed_field = gaussian_filter(random_field, sigma=50)
        return smoothed_field

    def add_turbulence(self, Vx, Vy, magnitude=0.1):
        """Adds random turbulence to the velocity fields."""
        turbulence_u = np.random.normal(loc=0, scale=magnitude, size=(self.width, self.height))
        turbulence_v = np.random.normal(loc=0, scale=magnitude, size=(self.width, self.height))
        smoothed_turbulence_u = gaussian_filter(turbulence_u, sigma=2)
        smoothed_turbulence_v = gaussian_filter(turbulence_v, sigma=2)
        Vx += smoothed_turbulence_u
        Vy += smoothed_turbulence_v

    def generate_combined_wind_field(self, seed):
        """Generates combined wind speed and direction fields using simplified 2D Navier-Stokes."""
        # Initialize velocity fields
        Vx = np.zeros((self.width, self.height))
        Vy = np.zeros((self.width, self.height))

        # Generate random disturbances
        self.Vx_disturbance = self.generate_smoothed_random_field(seed[0])
        self.Vy_disturbance = self.generate_smoothed_random_field(seed[1])  # Different seed for Vy_disturbance

        # Independent random factors for Vx and Vy disturbances
        random_factor_u = np.random.uniform(low=-2, high=2)
        random_factor_v = np.random.uniform(low=-2, high=2)
        alpha = 0.9
        random_factor_u = random_factor_u * (1 - alpha) + self.previous_random_u * alpha
        random_factor_v = random_factor_v * (1 - alpha) + self.previous_random_v * alpha
        self.previous_random_u = random_factor_u
        self.previous_random_v = random_factor_v
        self.Vx_disturbance *= random_factor_u
        self.Vy_disturbance *= random_factor_v

        # Apply disturbances to the velocity fields
        Vx += self.Vx_disturbance
        Vy += self.Vy_disturbance

        # Time integration loop (Euler integration)
        for _ in range(10):  # Number of iterations (adjust as needed)
            # Calculate gradients
            du_dx, du_dy = np.gradient(Vx)
            dv_dx, dv_dy = np.gradient(Vy)

            # Calculate Laplacian
            d2u_dx2 = np.gradient(du_dx)[0]
            d2u_dy2 = np.gradient(du_dy)[1]
            d2v_dx2 = np.gradient(dv_dx)[0]
            d2v_dy2 = np.gradient(dv_dy)[1]

            # Apply viscosity
            Vx += self.dt * (self.viscosity * (d2u_dx2 + d2u_dy2))
            Vy += self.dt * (self.viscosity * (d2v_dx2 + d2v_dy2))

            # Add random turbulence
            self.add_turbulence(Vx, Vy)

        # Calculate wind speed and direction from velocity components
        wind_speed = np.sqrt(Vx**2 + Vy**2)
        wind_direction = np.arctan2(Vy, Vx)

        # Normalize wind speed to a specific range
        wind_speed = (wind_speed - np.min(wind_speed)) / (np.max(wind_speed) - np.min(wind_speed))  # Normalize to [0, 1]
        wind_speed *= 24 * 3.28084  # Scale to desired wind speed range, e.g., [0, 24]

        wind_direction = np.degrees(wind_direction)  # Convert to degrees

        return wind_speed, wind_direction, np.mean(wind_speed)
    

    def interpolate_fields(self, field1, field2, alpha):
        """Interpolates between two fields."""
        return (1 - alpha) * field1 + alpha * field2

    def reset(self, seed):
        """Resets the wind fields."""
        self.Vx, self.U_dir, mean_U = self.generate_combined_wind_field(seed)
        return self.Vx, self.U_dir

    def step(self, seed):
        """Performs one step of wind evolution."""
        new_U, new_U_dir, mean_U = self.generate_combined_wind_field(seed)
        alpha = 0.1  # You can adjust this value to control the smoothness of evolution
        self.Vx = self.interpolate_fields(self.Vx, new_U, alpha)
        self.U_dir = self.interpolate_fields(self.U_dir, new_U_dir, alpha)
        return self.Vx, self.U_dir, mean_U

    def temporal_evolution(self, num_steps):
        """Simulates temporal evolution of wind fields."""
        for step in range(num_steps):
            self.Vx, self.U_dir = self.step()

# Initialize the Vent class and simulate the temporal evolution
# vent = Vent()
# vent.temporal_evolution(5)

# # Visualize the initial and final wind speed and direction fields
# U_initial, U_dir_initial = vent.generate_combined_wind_field()
# U_final, U_dir_final = vent.generate_combined_wind_field()

