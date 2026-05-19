import numpy as np

# For synthetic dataset (kept for backward compatibility)
x_s_coef1 = np.array([
    [-0.2,  1.1, -0.9],
]).T * (1 / 3)

x_s_coef2 = np.array([
    [-0.2,  1.0, -0.8],
    [0.0, 0.1, -0.1],
]).T * (2 / 3)

x_s_coef3 = np.array([
    [0.1, 0.5, -0.35],
    [0.0, 0.05, -0.05],
    [-0.3, 0.55, -0.5],
]).T

x_s_coef4 = np.array([
    [0.1, 0.4, -0.35],
    [0.0, 0.05, -0.05],
    [-0.25, 0.4, -0.5],
    [-0.05, 0.25, 0.0],
]).T * (4 / 3)

x_s_coef5 = np.array([
    [0.1, 0.4, -0.35],
    [0.0, 0.05, -0.05],
    [-0.25, 0.4, -0.5],
    [-0.05, 0.25, 0.0],
    [0.0, 0.0, 0.0],
]).T * (5 / 3)

coef_dict = {
    1: x_s_coef1,
    2: x_s_coef2,
    3: x_s_coef3,
    4: x_s_coef4,
    5: x_s_coef5,
}

# For real ACTG175 dataset
# These coefficients are not used directly for real data
# (estimation is done via ML models instead)
x_s_coef_actg175 = np.array([
    [0.5, -0.3, 0.2],      # Effect of CD4 count
    [-0.2, 0.4, -0.1],     # Effect of CD8 count
    [0.1, 0.2, -0.5],      # Effect of treatment discontinuation
    [0.1, 0.0, 0.0],       # Age effect
    [0.05, 0.0, 0.0],      # Weight effect
    [-0.1, 0.1, 0.0],      # Race effect
    [0.0, 0.05, 0.0],      # Gender effect
]).T

# Extended coefficient dictionary including ACTG175
coef_dict_actg175 = {
    7: x_s_coef_actg175,  # 7 baseline covariates for ACTG175
}