import numpy as np
from scipy import stats


def build_design_matrix(x, degree):
    return np.vander(x, degree + 1, increasing=True)


def fit_polynomial_lsm(x, y, degree):
    design = build_design_matrix(x, degree)
    return np.linalg.inv(design.T @ design) @ design.T @ y


def evaluate_polynomial(x, coefficients):
    return build_design_matrix(x, len(coefficients) - 1) @ coefficients


def r_squared(y, y_fit):
    return 1 - np.sum((y - y_fit) ** 2) / np.sum((y - y.mean()) ** 2)


def rmse(y, y_fit):
    return np.sqrt(np.mean((y - y_fit) ** 2))


def split_train_test(x, y, test_share):
    split = int(len(y) * (1 - test_share))
    return x[:split], y[:split], x[split:], y[split:]


def compare_degrees(x, y, degrees, test_share):
    x_train, y_train, x_test, y_test = split_train_test(x, y, test_share)
    rows = []
    for degree in degrees:
        train_coefficients = fit_polynomial_lsm(x_train, y_train, degree)
        full_coefficients = fit_polynomial_lsm(x, y, degree)
        rows.append({
            "degree": degree,
            "r2": r_squared(y, evaluate_polynomial(x, full_coefficients)),
            "test_rmse": rmse(y_test, evaluate_polynomial(x_test, train_coefficients)),
        })
    return rows


def select_degree(rows):
    return min(rows, key=lambda row: row["test_rmse"])["degree"]


def describe_sample(values):
    return {"mean": values.mean(), "std": values.std(ddof=1)}


def fit_normal_noise(residuals):
    mean, std = stats.norm.fit(residuals)
    statistic, p_value = stats.kstest(residuals, stats.norm(mean, std).cdf)
    return {"ks": statistic, "p_value": p_value}


def forecast_grid(x, horizon_share):
    step = x[1] - x[0]
    horizon = int(len(x) * horizon_share)
    return x[-1] + step * np.arange(1, horizon + 1)

