import numpy as np

MIN_MEMORY = 2


def alpha_beta_coefficients(n):
    return 2 * (2 * n - 1) / (n * (n + 1)), 6 / (n * (n + 1))


def alpha_beta_gamma_coefficients(n):
    denominator = n * (n + 1) * (n + 2)
    return 3 * (3 * n ** 2 - 3 * n + 2) / denominator, 18 * (2 * n - 1) / denominator, 60 / denominator


def innovation_variance_factor(points):
    return 1 + 1 / points + 3 * (points + 1) / (points * (points - 1))


def alpha_beta_filter(measurements, sigma=None, threshold_factor=3):
    estimates = np.empty(len(measurements))
    estimates[0] = measurements[0]
    level, speed, memory = measurements[0], 0.0, 1
    innovation_sum, variance_sum = 0.0, 0.0
    resets = []
    for index in range(1, len(measurements)):
        memory += 1
        predicted = level + speed
        innovation = measurements[index] - predicted
        if sigma is not None and memory > MIN_MEMORY:
            innovation_sum += innovation
            variance_sum += sigma ** 2 * innovation_variance_factor(memory - 1)
            if abs(innovation_sum) > threshold_factor * np.sqrt(variance_sum):
                memory = MIN_MEMORY
                innovation_sum, variance_sum = 0.0, 0.0
                resets.append(index)
        alpha, beta = alpha_beta_coefficients(memory)
        level = predicted + alpha * innovation
        speed = speed + beta * innovation
        estimates[index] = level
    return estimates, resets


def alpha_beta_gamma_filter(measurements):
    estimates = np.empty(len(measurements))
    estimates[0] = measurements[0]
    level, speed, acceleration = measurements[0], 0.0, 0.0
    for index in range(1, len(measurements)):
        predicted_level = level + speed + acceleration / 2
        predicted_speed = speed + acceleration
        innovation = measurements[index] - predicted_level
        alpha, beta, gamma = alpha_beta_gamma_coefficients(index + 1)
        level = predicted_level + alpha * innovation
        speed = predicted_speed + beta * innovation
        acceleration = acceleration + gamma * innovation
        estimates[index] = level
    return estimates
