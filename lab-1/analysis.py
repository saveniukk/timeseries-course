import numpy as np
from scipy import stats

CANDIDATE_LAWS = {
    "нормальний": stats.norm,
    "рівномірний": stats.uniform,
    "експоненційний": stats.expon,
    "хі-квадрат": stats.chi2,
}


def build_design_matrix(x, degree):
    return np.vander(x, degree + 1, increasing=True)


def fit_polynomial_lsm(x, y, degree):
    design = build_design_matrix(x, degree)
    return np.linalg.inv(design.T @ design) @ design.T @ y


def evaluate_polynomial(x, coefficients):
    return build_design_matrix(x, len(coefficients) - 1) @ coefficients


def r_squared(y, y_fit):
    return 1 - np.sum((y - y_fit) ** 2) / np.sum((y - y.mean()) ** 2)


def adjusted_r_squared(y, y_fit, degree):
    n = len(y)
    return 1 - (1 - r_squared(y, y_fit)) * (n - 1) / (n - degree - 1)


def bayesian_criterion(y, y_fit, degree):
    n = len(y)
    return n * np.log(np.sum((y - y_fit) ** 2) / n) + (degree + 1) * np.log(n)


def compare_degrees(x, y, max_degree):
    rows = []
    for degree in range(1, max_degree + 1):
        y_fit = evaluate_polynomial(x, fit_polynomial_lsm(x, y, degree))
        rows.append({
            "degree": degree,
            "r2": r_squared(y, y_fit),
            "adjusted_r2": adjusted_r_squared(y, y_fit, degree),
            "bic": bayesian_criterion(y, y_fit, degree),
            "sigma": np.std(y - y_fit, ddof=degree + 1),
        })
    return rows


def select_degree_by_bic(rows):
    return min(rows, key=lambda row: row["bic"])["degree"]


def describe_sample(values):
    return {
        "mean": values.mean(),
        "variance": values.var(ddof=1),
        "std": values.std(ddof=1),
    }


def fit_candidate_laws(residuals):
    results = []
    for name, law in CANDIDATE_LAWS.items():
        params = law.fit(residuals)
        statistic, p_value = stats.kstest(residuals, law(*params).cdf)
        results.append({"law": name, "params": params, "ks": statistic, "p_value": p_value})
    return sorted(results, key=lambda item: item["ks"])
