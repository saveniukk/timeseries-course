import numpy as np

from analysis import evaluate_polynomial, fit_polynomial_lsm

WINDOW_TREND_DEGREE = 1


def window_indices(index, size, window_size):
    start = min(max(index - window_size // 2, 0), size - window_size - 1)
    indices = np.arange(start, start + window_size + 1)
    return indices[indices != index]


def predict_from_neighbours(series, index, window_size):
    neighbours = window_indices(index, len(series), window_size)
    coefficients = fit_polynomial_lsm(neighbours.astype(float), series[neighbours], WINDOW_TREND_DEGREE)
    fitted = evaluate_polynomial(neighbours.astype(float), coefficients)
    sigma = np.std(series[neighbours] - fitted, ddof=WINDOW_TREND_DEGREE + 1)
    prediction = evaluate_polynomial(np.array([float(index)]), coefficients)[0]
    return prediction, sigma


def clean_by_sliding_window(series, window_size, threshold_factor):
    cleaned = series.copy()
    detected = []
    for index in range(len(series)):
        prediction, sigma = predict_from_neighbours(series, index, window_size)
        if abs(series[index] - prediction) > threshold_factor * sigma:
            detected.append(index)
            cleaned[index] = prediction
    return cleaned, np.array(detected, dtype=int)


def detection_quality(true_positions, detected_positions, size):
    true_set, detected_set = set(true_positions), set(detected_positions)
    hits = len(true_set & detected_set)
    return {
        "hits": hits,
        "false_alarms": len(detected_set - true_set),
        "recall": hits / len(true_set),
        "false_rate": len(detected_set - true_set) / len(true_set),
    }
