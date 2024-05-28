import numpy as np


def compute_rate_of_spread(
    loc_x: int,
    loc_y: int,
    new_loc_x: int,
    new_loc_y: int,
    w_0: int,
    delta: int,
    M_x: int,
    sigma: int,
    h: int,
    S_T: int,
    S_e: int,
    p_p: int,
    M_f: int,
    U: int,
    U_dir: int,
) -> int:
    """
    Compute the basic Rothermel rate of spread. All measurements are assumed to be in
    feet, minutes, and pounds, and BTU.

    Arguments:
        loc_x: The current x location
        loc_y: The current y location
        new_loc_x: The new x location
        new_loc_y: The new y location
        w_0: The oven-dry fuel load of the fuel at the new location
        delta: The fuel bed depth of the fuel at the new location
        M_x: The dead fuel moisture of extinction of the fuel at the new location
        sigma: The Surface-area-to-volume ratio of the fuel at the new location
        h: The fuel particle low heat content
        S_T: The fuel particle total mineral content
        S_e: The fuel particle effective mineral content
        p_p: The fuel particle oven-dry particle density
        M_f: The environment fuel moisture
        U: The environment wind speed
        U_dir: The environment wind direction (degrees clockwise from North)

    Returns:
        R: The computed rate of spread in ft/min
    """
    

    eta_S = 0.174 * S_e**-0.19
    r_M = M_f / M_x
    eta_M = 1 - 2.59 * r_M + 5.11 * r_M**2 - 3.52 * r_M**3
    w_n = w_0 * (1 - S_T)
    p_b = w_0 / delta
    B = p_b / p_p
    B_op = 3.348 * sigma**-0.8189
    gamma_prime_max = sigma**1.5 / (495 + 0.0594 * sigma**1.5)
    A = 133 * sigma**-0.7913
    gamma_prime = gamma_prime_max * (B / B_op) ** A * np.exp(A * (1 - B / B_op))
    I_R = gamma_prime * w_n * h * eta_M * eta_S
    xi = np.exp((0.792 + 0.681 * sigma**0.5) * (B + 0.1)) / (192 + 0.25 * sigma)
    c = 7.47 * np.exp(-0.133 * sigma**0.55)
    b = 0.02526 * sigma**0.54
    e = 0.715 * np.exp(-3.59e-4 * sigma)

    angle_of_travel = np.arctan2(loc_y - new_loc_y, new_loc_x - loc_x)

    wind_angle_radians = np.radians(90 - U_dir)
    U = U * np.cos(wind_angle_radians - angle_of_travel)
    
    phi_w = c * np.sign(U) * (np.abs(U))**b * (B / B_op) ** -e
    
    phi_s = 0  # Placeholder for slope factor if needed in the future

    epsilon = np.exp(-138 / sigma)
    Q_ig = 250 + 1116 * M_f

    R = ((I_R * xi) * (1 + phi_w + phi_s)) / (p_b * epsilon * Q_ig)

    return R

