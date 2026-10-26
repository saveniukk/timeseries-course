import numpy as np

from analysis import CANDIDATE_LAWS, evaluate_polynomial


def generate_noise(law_name, params, size, target_mean, target_std, rng):
    frozen = CANDIDATE_LAWS[law_name](*params)
    raw = frozen.rvs(size=size, random_state=rng)
    return (raw - frozen.mean()) / frozen.std() * target_std + target_mean


def synthesize_series(x, coefficients, noise):
    return evaluate_polynomial(x, coefficients) + noise
