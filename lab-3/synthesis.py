import numpy as np

from analysis import evaluate_polynomial


def synthesize_series(x, coefficients, noise_std, rng):
    return evaluate_polynomial(x, coefficients) + rng.normal(0, noise_std, len(x))


def add_anomalous_measurements(synthetic, ideal, noise_std, anomaly_fraction, q_av, rng):
    count = int(len(synthetic) * anomaly_fraction)
    positions = rng.choice(len(synthetic), size=count, replace=False)
    series = synthetic.copy()
    series[positions] = ideal[positions] + rng.normal(0, q_av * noise_std, count)
    return series, np.sort(positions)
