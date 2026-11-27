from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis import (autocorrelations, cluster_objects, component_strength, correlation_matrices, decompose,
                      manual_autocorrelation, mean_abs_difference, mean_off_diagonal, monthly_segment_features,
                      normality_test, series_properties, significance_bound, standardize)
from datasets import load_mortgage_sales
from parsing import load_series
from plots import (plot_autocorrelations, plot_clustered_series, plot_correlation_heatmap, plot_decomposition,
                   plot_frame_real_vs_synthetic, plot_month_clusters, plot_usd_real_vs_synthetic)
from synthesis import synthesize_frame, synthesize_series

BASE_DIR = Path(__file__).parent
USD_PATH = BASE_DIR / "data" / "parsed_minfin_usd_2025.csv"
DATASET_PATH = BASE_DIR / "data" / "Data_Set_4.xlsx"
REFRESH_DATA = False
SEED = 4
USD_PERIOD = 5
QUARTER_PERIOD = 4
ACF_LAGS = 20
CLUSTER_RANGE = range(2, 5)
SHOWN_ACF_LAGS = 5
DECOMPOSED_SERIES = "Перші покупці"
COMPARED_SERIES = ["Перші покупці", "Прямі продажі", "Рефінансування"]
SIGNIFICANCE_LEVEL = 0.05
PRINT_PRECISION = 4
PROPERTY_NAMES = [("mean", "Середнє"), ("std", "СКВ"), ("skewness", "Асиметрія"), ("kurtosis", "Ексцес"),
                  ("adf_p", "p-value ADF"), ("trend_strength", "Сила тренду"),
                  ("seasonal_strength", "Сила сезонності")]


def print_header(title):
    print(f"\n----- {title} -----")


def fmt(value):
    return f"{value:.{PRINT_PRECISION}f}"


def stationarity_verdict(p_value):
    return "стаціонарний" if p_value < SIGNIFICANCE_LEVEL else "нестаціонарний"


def load_usd():
    print_header("1. Реальні дані (курс USD, ЛР1–3): декомпозиція та властивості")
    frame = load_series(USD_PATH, REFRESH_DATA)
    values = frame["usd_rate"].to_numpy()
    print(f"Спостережень: {len(values)}, період {frame['date'].min().date()} – {frame['date'].max().date()}")
    return frame["date"], values


def analyze_usd_properties(values):
    parts = decompose(values, USD_PERIOD)
    properties = series_properties(values, parts)
    for key, name in PROPERTY_NAMES:
        print(f"{name:<16}: {fmt(properties[key])}")
    print(f"За ADF ряд {stationarity_verdict(properties['adf_p'])} (α = {SIGNIFICANCE_LEVEL})")
    return parts, properties


def analyze_usd_correlation(values):
    print_header("2. Кореляційний аналіз курсу USD (АКФ, ЧАКФ)")
    autocorrelation, partial = autocorrelations(values, ACF_LAGS)
    bound = significance_bound(len(values))
    print(f"Межа значущості 95 %: ±{fmt(bound)}")
    print("Лаг |   АКФ | ЧАКФ")
    for lag in range(1, SHOWN_ACF_LAGS + 1):
        print(f"{lag:3d} | {fmt(autocorrelation[lag])} | {fmt(partial[lag])}")
    return autocorrelation, partial, bound


def cluster_usd_months(dates, values):
    print_header("3. Кластеризація місячних сегментів курсу USD")
    names, features = monthly_segment_features(dates, values)
    scores, best_k, labels = cluster_objects(standardize(features), CLUSTER_RANGE, SEED)
    for k, score in scores.items():
        print(f"k = {k}: silhouette = {fmt(score)}")
    print(f"Обрано k = {best_k}")
    for cluster in range(best_k):
        print(f"Кластер {cluster + 1}: " + ", ".join(name for name, label in zip(names, labels) if label == cluster))
    return names, features, labels


def build_usd_model(values, parts, properties, rng):
    print_header("4. Синтезована модель курсу USD та порівняння з реальними даними")
    noise_std = parts["residual"].std(ddof=1)
    synthetic = synthesize_series(parts, noise_std, rng)
    synthetic_parts = decompose(synthetic, USD_PERIOD)
    synthetic_properties = series_properties(synthetic, synthetic_parts)
    print("Показник         |  Реальні | Синтетичні")
    for key, name in PROPERTY_NAMES:
        print(f"{name:<16} | {fmt(properties[key]):>8} | {fmt(synthetic_properties[key]):>10}")
    statistic, p_value = stats.ks_2samp(parts["residual"], synthetic_parts["residual"])
    print(f"KS-тест двох вибірок залишків: статистика = {fmt(statistic)}, p-value = {fmt(p_value)}")
    print(f"Нормальність залишків реальних даних: p-value = {fmt(normality_test(parts['residual'])[1])}")
    return synthetic, synthetic_parts


def load_dataset():
    print_header("5. Дані Data_Set_4 (іпотечні продажі, квартальні): декомпозиція та властивості")
    frame = load_mortgage_sales(DATASET_PATH)
    print(f"Рядів: {frame.shape[1]}, кварталів: {frame.shape[0]} ({frame.index[0]} – {frame.index[-1]})")
    print("Ряди: " + ", ".join(f"{number}. {name}" for number, name in enumerate(frame.columns, start=1)))
    return frame


def analyze_dataset_properties(frame):
    parts_by_name = {name: decompose(frame[name].to_numpy(), QUARTER_PERIOD) for name in frame}
    print("Ряд                | Сила тренду | Сила сезонності | p-value ADF | Асиметрія | Ексцес")
    for name in frame:
        properties = series_properties(frame[name].to_numpy(), parts_by_name[name])
        print(f"{name:<18} | {fmt(properties['trend_strength']):>11} | {fmt(properties['seasonal_strength']):>15} | "
              f"{fmt(properties['adf_p']):>11} | {fmt(properties['skewness']):>9} | {fmt(properties['kurtosis']):>6}")
    return parts_by_name


def analyze_dataset_correlation(frame, parts_by_name):
    print_header("6. Кореляційний аналіз рядів Data_Set_4")
    pearson, spearman = correlation_matrices(frame)
    print("Кореляція Пірсона:")
    print(pearson.round(PRINT_PRECISION).set_axis(range(1, len(pearson) + 1), axis=1).to_string())
    print(f"Середня недіагональна кореляція: Пірсона {fmt(mean_off_diagonal(pearson))}, "
          f"Спірмена {fmt(mean_off_diagonal(spearman))}")
    residuals = pd.DataFrame({name: parts["residual"] for name, parts in parts_by_name.items()})
    print(f"Середня кореляція залишків (без тренду і сезонності): {fmt(mean_off_diagonal(residuals.corr()))}")
    return pearson


def cluster_dataset_series(frame):
    print_header("7. Кластеризація рядів Data_Set_4")
    standardized = standardize(frame)
    scores, best_k, labels = cluster_objects(standardized.to_numpy().T, CLUSTER_RANGE, SEED)
    for k, score in scores.items():
        print(f"k = {k}: silhouette = {fmt(score)}")
    print(f"Обрано k = {best_k}")
    for cluster in range(best_k):
        print(f"Кластер {cluster + 1}: " + ", ".join(name for name, label in zip(frame.columns, labels) if label == cluster))
    return standardized, labels


def build_dataset_model(frame, parts_by_name, pearson, rng):
    print_header("8. Синтезована модель Data_Set_4 та порівняння з реальними даними")
    synthetic = synthesize_frame(parts_by_name, rng)
    synthetic.index = frame.index
    print("Ряд                | Сила сез. (реал.) | Сила сез. (синт.) | СКВ (реал.) | СКВ (синт.)")
    for name in frame:
        synthetic_parts = decompose(synthetic[name].to_numpy(), QUARTER_PERIOD)
        print(f"{name:<18} | {fmt(component_strength(parts_by_name[name]['residual'], parts_by_name[name]['seasonal'])):>17} | "
              f"{fmt(component_strength(synthetic_parts['residual'], synthetic_parts['seasonal'])):>17} | "
              f"{fmt(frame[name].std(ddof=1)):>11} | {fmt(synthetic[name].std(ddof=1)):>11}")
    print(f"Середня різниця кореляцій реальних і синтетичних рядів: {fmt(mean_abs_difference(pearson, synthetic.corr()))}")
    synthetic_residuals = pd.DataFrame({name: decompose(synthetic[name].to_numpy(), QUARTER_PERIOD)["residual"]
                                        for name in frame})
    real_residuals = pd.DataFrame({name: parts["residual"] for name, parts in parts_by_name.items()})
    print(f"Середня різниця кореляцій залишків: {fmt(mean_abs_difference(real_residuals.corr(), synthetic_residuals.corr()))}")
    return synthetic


def verify_decomposition(values, parts, frame, parts_by_name, autocorrelation, pearson):
    print_header("Верифікація: декомпозиція та модель")
    usd_error = np.max(np.abs(parts["trend"] + parts["seasonal"] + parts["residual"] - values))
    set_error = max(np.max(np.abs(p["trend"] + p["seasonal"] + p["residual"] - frame[name].to_numpy()))
                    for name, p in parts_by_name.items())
    print(f"Макс. похибка суми складових для USD: {usd_error:.2e}")
    print(f"Макс. похибка суми складових для Data_Set_4: {set_error:.2e}")
    manual = max(abs(manual_autocorrelation(values, lag) - autocorrelation[lag]) for lag in range(1, ACF_LAGS + 1))
    print(f"Макс. різниця власної АКФ і statsmodels: {manual:.2e}")
    numpy_pearson = np.corrcoef(frame.to_numpy(), rowvar=False)
    print(f"Макс. різниця кореляції Пірсона (pandas і numpy): {np.max(np.abs(pearson.to_numpy() - numpy_pearson)):.2e}")


def main():
    rng = np.random.default_rng(SEED)
    dates, values = load_usd()
    parts, properties = analyze_usd_properties(values)
    autocorrelation, partial, bound = analyze_usd_correlation(values)
    month_names, month_features, month_labels = cluster_usd_months(dates, values)
    usd_synthetic, usd_synthetic_parts = build_usd_model(values, parts, properties, rng)
    frame = load_dataset()
    parts_by_name = analyze_dataset_properties(frame)
    pearson = analyze_dataset_correlation(frame, parts_by_name)
    standardized, series_labels = cluster_dataset_series(frame)
    synthetic = build_dataset_model(frame, parts_by_name, pearson, rng)
    verify_decomposition(values, parts, frame, parts_by_name, autocorrelation, pearson)
    plot_decomposition(dates, values, parts, "Декомпозиція курсу USD", "грн")
    plot_autocorrelations(autocorrelation, partial, bound)
    plot_month_clusters(month_names, month_features, month_labels)
    plot_usd_real_vs_synthetic(dates, values, usd_synthetic, parts["residual"], usd_synthetic_parts["residual"])
    plot_decomposition(frame.index.to_timestamp(), frame[DECOMPOSED_SERIES].to_numpy(), parts_by_name[DECOMPOSED_SERIES],
                       f"Декомпозиція: {DECOMPOSED_SERIES}", "кількість продажів")
    plot_correlation_heatmap(pearson)
    plot_clustered_series(frame.index.to_timestamp(), standardized, series_labels)
    plot_frame_real_vs_synthetic(frame.index.to_timestamp(), frame, synthetic, COMPARED_SERIES)


if __name__ == "__main__":
    main()
