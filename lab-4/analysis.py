import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, pacf


def decompose(values, period):
    result = STL(values, period=period, seasonal=len(values) | 1).fit()
    return {"trend": np.asarray(result.trend), "seasonal": np.asarray(result.seasonal),
            "residual": np.asarray(result.resid)}


def component_strength(residual, component):
    return max(0.0, 1 - np.var(residual, ddof=1) / np.var(residual + component, ddof=1))


def series_properties(values, parts):
    return {
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "skewness": stats.skew(values),
        "kurtosis": stats.kurtosis(values),
        "adf_p": adfuller(values, result_object=True).pvalue,
        "trend_strength": component_strength(parts["residual"], parts["trend"]),
        "seasonal_strength": component_strength(parts["residual"], parts["seasonal"]),
    }


def manual_autocorrelation(values, lag):
    centered = values - values.mean()
    return np.sum(centered[lag:] * centered[:len(values) - lag]) / np.sum(centered ** 2)


def autocorrelations(values, lags):
    return acf(values, nlags=lags), pacf(values, nlags=lags)


def significance_bound(size):
    return 1.96 / np.sqrt(size)


def standardize(matrix):
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=1)


def cluster_objects(scaled, k_range, seed):
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(scaled)
        scores[k] = silhouette_score(scaled, labels)
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit_predict(scaled)
    return scores, best_k, labels


def monthly_segment_features(dates, values):
    rows = []
    months = pd.Series(values).groupby(dates.dt.to_period("M").to_numpy())
    for month, segment in months:
        segment = segment.to_numpy()
        slope = np.polyfit(np.arange(len(segment)), segment, 1)[0]
        rows.append((str(month), segment.mean(), segment.std(ddof=1), slope))
    names = [row[0] for row in rows]
    return names, np.array([row[1:] for row in rows])


def correlation_matrices(frame):
    return frame.corr(method="pearson"), frame.corr(method="spearman")


def mean_off_diagonal(matrix):
    return matrix.to_numpy()[~np.eye(len(matrix), dtype=bool)].mean()


def mean_abs_difference(first, second):
    mask = ~np.eye(len(first), dtype=bool)
    return np.mean(np.abs(first.to_numpy() - second.to_numpy())[mask])


def normality_test(values):
    mean, std = values.mean(), values.std(ddof=1)
    statistic, p_value = stats.kstest(values, stats.norm(mean, std).cdf)
    return statistic, p_value
