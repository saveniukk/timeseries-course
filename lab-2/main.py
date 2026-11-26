from pathlib import Path

import numpy as np
import statsmodels.api as sm

from analysis import (build_design_matrix, compare_degrees, describe_sample, evaluate_polynomial, fit_normal_noise,
                      fit_polynomial_lsm, forecast_grid, r_squared, rmse, select_degree)
from anomalies import clean_by_sliding_window, detection_quality
from parsing import load_series
from plots import (plot_anomaly_cleaning, plot_degree_selection, plot_extrapolation,
                   plot_real_forecast)
from synthesis import add_anomalous_measurements, synthesize_series

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "parsed_minfin_usd_2025.csv"
REFRESH_DATA = False
TRAIN_YEAR = 2025
SEED = 4
CANDIDATE_DEGREES = range(1, 6)
TEST_SHARE = 0.3
FORECAST_SHARE = 0.5
ANOMALY_FRACTION = 0.1
Q_AV = 3
WINDOW_SIZE = 10
THRESHOLD_FACTOR = 3
PRINT_PRECISION = 4


def print_header(title):
    print(f"\n----- {title} -----")


def fmt(value):
    return f"{value:.{PRINT_PRECISION}f}"


def load_input_data():
    print_header("1. Вхідні дані (з ЛР1)")
    frame = load_series(DATA_PATH, REFRESH_DATA, TRAIN_YEAR)
    y = frame["usd_rate"].to_numpy()
    print(f"Спостережень: {len(y)}, період {frame['date'].min().date()} – {frame['date'].max().date()}")
    return y, np.linspace(0, 1, len(y))


def choose_degree(x, y):
    print_header("2. Вибір степеня полінома")
    rows = compare_degrees(x, y, CANDIDATE_DEGREES, TEST_SHARE)
    print("Степінь |      R² | RMSE на відкладеній частині")
    for row in rows:
        print(f"{row['degree']:7d} | {fmt(row['r2'])} | {fmt(row['test_rmse']):>27}")
    degree = select_degree(rows)
    print(f"Обрано степінь {degree} (мінімальний RMSE на відкладеній частині)")
    return rows, degree


def learn_and_forecast(x, y, degree):
    print_header("3. Навчання МНК і прогноз на 0,5 інтервалу")
    coefficients = fit_polynomial_lsm(x, y, degree)
    trend = evaluate_polynomial(x, coefficients)
    x_future = forecast_grid(x, FORECAST_SHARE)
    forecast = evaluate_polynomial(x_future, coefficients)
    print("Коефіцієнти:", ", ".join(fmt(c) for c in coefficients))
    print(f"R² тренду = {fmt(r_squared(y, trend))}")
    print(f"Прогноз: {len(x_future)} точок, від {fmt(forecast[0])} до {fmt(forecast[-1])} грн")
    return coefficients, trend, x_future, forecast


def build_synthetic_model(x, y, trend, coefficients, rng):
    print_header("4. Синтезована модель (тренд + нормальний шум)")
    residuals = y - trend
    summary = describe_sample(residuals)
    noise = fit_normal_noise(residuals)
    print(f"Залишки реальних даних: M = {fmt(summary['mean'])}, σ = {fmt(summary['std'])}")
    print(f"Нормальний закон: KS = {fmt(noise['ks'])}, p-value = {fmt(noise['p_value'])}")
    synthetic = synthesize_series(x, coefficients, summary["std"], rng)
    print(f"Синтетичний ряд: σ шуму = {fmt(summary['std'])}")
    return synthetic, summary["std"]


def study_anomalies(x, synthetic, ideal, noise_std, rng):
    print_header("5. Аномальні виміри та очищення")
    contaminated, true_positions = add_anomalous_measurements(synthetic, ideal, noise_std, ANOMALY_FRACTION, Q_AV, rng)
    cleaned, detected = clean_by_sliding_window(contaminated, WINDOW_SIZE, THRESHOLD_FACTOR)
    quality = detection_quality(true_positions, detected, len(x))
    print(f"Задано АВ: {len(true_positions)} ({ANOMALY_FRACTION:.0%}), Q_AV = {Q_AV}, σ_AV = {fmt(Q_AV * noise_std)}")
    print(f"Виявлено: {len(detected)}, з них вірно: {quality['hits']}, хибно: {quality['false_alarms']}")
    print(f"Ймовірність вірного виявлення = {fmt(quality['recall'])}")
    print(f"Ймовірність хибної ідентифікації = {fmt(quality['false_rate'])}")
    return contaminated, cleaned, true_positions, detected


def compare_before_after(x, ideal, series_by_name, degree, ideal_coefficients):
    print_header("6. Вплив очищення на модель і прогноз")
    x_future = forecast_grid(x, FORECAST_SHARE)
    ideal_forecast = evaluate_polynomial(x_future, ideal_coefficients)
    forecasts = {}
    print("Дані                  |      R² | σ залишків | RMSE тренду | RMSE прогнозу")
    for name, series in series_by_name.items():
        coefficients = fit_polynomial_lsm(x, series, degree)
        trend = evaluate_polynomial(x, coefficients)
        forecasts[name] = evaluate_polynomial(x_future, coefficients)
        print(f"{name:<21} | {fmt(r_squared(series, trend))} | {fmt(np.std(series - trend, ddof=degree + 1)):>10} | "
              f"{fmt(rmse(ideal, trend)):>11} | {fmt(rmse(ideal_forecast, forecasts[name])):>13}")
    return x_future, ideal_forecast, forecasts


def clean_real_data(x, y, degree):
    print_header("7. Очищення реальних даних")
    cleaned, detected = clean_by_sliding_window(y, WINDOW_SIZE, THRESHOLD_FACTOR)
    print(f"Позначено вимірів: {len(detected)} з {len(y)}")
    print("Дані           |      R² | σ залишків")
    for name, series in (("Реальні", y), ("Після очищення", cleaned)):
        trend = evaluate_polynomial(x, fit_polynomial_lsm(x, series, degree))
        print(f"{name:<14} | {fmt(r_squared(series, trend))} | {fmt(np.std(series - trend, ddof=degree + 1)):>10}")


def verify_least_squares(x, y, degree, coefficients):
    print_header("Верифікація: власний МНК проти бібліотек")
    numpy_coefficients = np.polynomial.polynomial.polyfit(x, y, degree)
    ols = sm.OLS(y, build_design_matrix(x, degree)).fit()
    print(f"Макс. різниця коефіцієнтів з numpy: {np.max(np.abs(coefficients - numpy_coefficients)):.2e}")
    print(f"Макс. різниця коефіцієнтів з statsmodels: {np.max(np.abs(coefficients - ols.params)):.2e}")
    print(f"R² власний = {fmt(r_squared(y, evaluate_polynomial(x, coefficients)))}, statsmodels = {fmt(ols.rsquared)}")


def main():
    rng = np.random.default_rng(SEED)
    y, x = load_input_data()
    rows, degree = choose_degree(x, y)
    coefficients, trend, x_future, forecast = learn_and_forecast(x, y, degree)
    synthetic, noise_std = build_synthetic_model(x, y, trend, coefficients, rng)
    contaminated, cleaned, true_positions, detected = study_anomalies(x, synthetic, trend, noise_std, rng)
    x_future, ideal_forecast, forecasts = compare_before_after(
        x, trend, {"Без АВ": synthetic, "З АВ": contaminated, "Після очищення": cleaned}, degree, coefficients)
    clean_real_data(x, y, degree)
    verify_least_squares(x, y, degree, coefficients)
    plot_real_forecast(x, y, trend, x_future, forecast, degree)
    plot_degree_selection(rows, degree)
    plot_anomaly_cleaning(x, trend, contaminated, cleaned, true_positions, detected)
    plot_extrapolation(x, x_future, trend, ideal_forecast, forecasts["З АВ"], forecasts["Після очищення"], degree)


if __name__ == "__main__":
    main()

