import numpy as np

WINDOW_TREND_DEGREE = 1


def build_design_matrix(x, degree):
    return np.vander(x, degree + 1, increasing=True)


def fit_polynomial_lsm(x, y, degree):
    design = build_design_matrix(x, degree)
    return np.linalg.inv(design.T @ design) @ design.T @ y


def evaluate_polynomial(x, coefficients):
    return build_design_matrix(x, len(coefficients) - 1) @ coefficients


def estimate_noise_std(series):
    return np.std(np.diff(series), ddof=1) / np.sqrt(2)


def window_indices(index, size, window_size):
    start = min(max(index - window_size // 2, 0), size - window_size - 1)
    indices = np.arange(start, start + window_size + 1)
    return indices[indices != index]


def predict_from_neighbours(series, index, window_size):
    neighbours = window_indices(index, len(series), window_size).astype(float)
    coefficients = fit_polynomial_lsm(neighbours, series[neighbours.astype(int)], WINDOW_TREND_DEGREE)
    fitted = evaluate_polynomial(neighbours, coefficients)
    sigma = np.std(series[neighbours.astype(int)] - fitted, ddof=WINDOW_TREND_DEGREE + 1)
    return evaluate_polynomial(np.array([float(index)]), coefficients)[0], sigma


def clean_by_sliding_window(series, window_size, threshold_factor):
    cleaned = series.copy()
    detected = []
    for index in range(len(series)):
        prediction, sigma = predict_from_neighbours(series, index, window_size)
        if abs(series[index] - prediction) > threshold_factor * sigma:
            detected.append(index)
            cleaned[index] = prediction
    return cleaned, np.array(detected, dtype=int)


def detection_quality(true_positions, detected_positions):
    true_set, detected_set = set(true_positions), set(detected_positions)
    hits = len(true_set & detected_set)
    return {
        "hits": hits,
        "false_alarms": len(detected_set - true_set),
        "recall": hits / len(true_set),
        "false_rate": len(detected_set - true_set) / len(true_set),
    }
