from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats

from analysis import (CANDIDATE_LAWS, build_design_matrix, compare_degrees, describe_sample, evaluate_polynomial,
                      fit_candidate_laws, fit_polynomial_lsm, r_squared, select_degree_by_bic)
from parsing import load_series
from plots import (plot_degree_selection, plot_histogram_comparison, plot_real_vs_synthetic, plot_residual_laws,
                   plot_series_with_trend)
from synthesis import generate_noise, synthesize_series

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "parsed_minfin_usd_2025.csv"
REFRESH_DATA = False
MAX_CONDITION_NUMBER = 1e11
SEED = 4
SIGNIFICANCE_LEVEL = 0.05
PRINT_PRECISION = 4


def print_header(title):
    print(f"\n==== {title} ====")


def format_number(value):
    return f"{value:.{PRINT_PRECISION}f}"


def parse_and_save_data():
    print_header("1-2. Парсинг сайту та збереження у файл")
    frame = load_series(DATA_PATH, REFRESH_DATA)
    values = frame["usd_rate"].to_numpy()
    print(f"Період: {frame['date'].min().date()} – {frame['date'].max().date()}, спостережень: {len(frame)}")
    print(f"Курс: мін {format_number(values.min())}, макс {format_number(values.max())}")
    return frame


def estimate_trend(x, y):
    print_header("3. Динаміка тренду реальних даних")
    rows = compare_degrees(x, y, MAX_CONDITION_NUMBER)
    print("Степінь |      R² | Скоригований R² |       BIC | σ (ddof=p+1) | cond(FᵀF)")
    for row in rows:
        print(f"{row['degree']:7d} | {format_number(row['r2'])} | {format_number(row['adjusted_r2']):>15} | "
              f"{format_number(row['bic']):>9} | {format_number(row['sigma']):>12} | {row['condition']:.3e}")
    degree = select_degree_by_bic(rows)
    coefficients = fit_polynomial_lsm(x, y, degree)
    trend = evaluate_polynomial(x, coefficients)
    print(f"Обрано степінь за BIC: {degree}; R² тренду = {format_number(r_squared(y, trend))}")
    print("Коефіцієнти:", ", ".join(format_number(c) for c in coefficients))
    return rows, degree, coefficients, trend


def describe_residuals(residuals):
    print_header("4. Статистичні характеристики залишків")
    summary = describe_sample(residuals)
    print(f"Математичне сподівання M = {format_number(summary['mean'])}")
    print(f"Дисперсія D = {format_number(summary['variance'])}")
    print(f"СКВ σ = {format_number(summary['std'])}")
    laws = fit_candidate_laws(residuals)
    print("Закон            |     KS | p-value")
    for item in laws:
        print(f"{item['law']:<16} | {format_number(item['ks'])} | {format_number(item['p_value'])}")
    print(f"Найкраще відповідає: {laws[0]['law']}")
    return summary, laws


def build_synthetic_model(x, coefficients, summary, best_law, rng):
    noise = generate_noise(best_law["law"], best_law["params"], len(x), summary["mean"], summary["std"], rng)
    return noise, synthesize_series(x, coefficients, noise)


def verify_least_squares(x, y, degree, coefficients):
    print_header("Верифікація: власний МНК проти бібліотечних")
    numpy_coefficients = np.polynomial.polynomial.polyfit(x, y, degree)
    ols = sm.OLS(y, build_design_matrix(x, degree)).fit()
    print(f"Макс. різниця коефіцієнтів з numpy: {np.max(np.abs(coefficients - numpy_coefficients)):.2e}")
    print(f"Макс. різниця коефіцієнтів з statsmodels: {np.max(np.abs(coefficients - ols.params)):.2e}")
    print(f"R² власний = {format_number(r_squared(y, evaluate_polynomial(x, coefficients)))}, "
          f"statsmodels = {format_number(ols.rsquared)}")


def verify_synthetic_model(x, y, degree, coefficients, residuals, summary, synthetic, noise, best_law):
    print_header("Верифікація: реальні дані та синтезована модель")
    synthetic_coefficients = fit_polynomial_lsm(x, synthetic, degree)
    synthetic_residuals = synthetic - evaluate_polynomial(x, synthetic_coefficients)
    synthetic_summary = describe_sample(synthetic_residuals)
    real_r2 = r_squared(y, evaluate_polynomial(x, coefficients))
    synthetic_r2 = r_squared(synthetic, evaluate_polynomial(x, synthetic_coefficients))
    print("Показник            |    Реальні | Синтетичні")
    print(f"M залишків          | {format_number(summary['mean']):>10} | {format_number(synthetic_summary['mean']):>10}")
    print(f"D залишків          | {format_number(summary['variance']):>10} | {format_number(synthetic_summary['variance']):>10}")
    print(f"σ залишків          | {format_number(summary['std']):>10} | {format_number(synthetic_summary['std']):>10}")
    print(f"R² тренду           | {format_number(real_r2):>10} | {format_number(synthetic_r2):>10}")
    print(f"Генератор шуму: M = {format_number(noise.mean())}, σ = {format_number(noise.std(ddof=1))}, "
          f"закон: {best_law['law']}")
    statistic, p_value = stats.ks_2samp(residuals, synthetic_residuals)
    print(f"KS-тест двох вибірок залишків: статистика = {format_number(statistic)}, p-value = {format_number(p_value)}")
    verdict = "не відхиляється" if p_value > SIGNIFICANCE_LEVEL else "відхиляється"
    print(f"Гіпотеза про однаковий розподіл при α = {SIGNIFICANCE_LEVEL} {verdict}")
    print("Коефіцієнти тренду (реальні | синтетичні):")
    for index, (real_c, synthetic_c) in enumerate(zip(coefficients, synthetic_coefficients)):
        print(f"  c{index}: {format_number(real_c)} | {format_number(synthetic_c)}")
    return synthetic_residuals


def main():
    rng = np.random.default_rng(SEED)
    frame = parse_and_save_data()
    dates, y = frame["date"], frame["usd_rate"].to_numpy()
    x = np.linspace(0, 1, len(y))
    rows, degree, coefficients, trend = estimate_trend(x, y)
    residuals = y - trend
    summary, laws = describe_residuals(residuals)
    noise, synthetic = build_synthetic_model(x, coefficients, summary, laws[0], rng)
    verify_least_squares(x, y, degree, coefficients)
    synthetic_residuals = verify_synthetic_model(x, y, degree, coefficients, residuals, summary, synthetic, noise, laws[0])
    plot_series_with_trend(dates, y, trend, degree)
    plot_degree_selection(rows, degree)
    plot_residual_laws(residuals, laws, CANDIDATE_LAWS)
    plot_real_vs_synthetic(dates, y, synthetic)
    plot_histogram_comparison(residuals, synthetic_residuals)


if __name__ == "__main__":
    main()
