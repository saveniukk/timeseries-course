from pathlib import Path

import numpy as np

from analysis import (clean_by_sliding_window, detection_quality, estimate_noise_std, evaluate_polynomial,
                      fit_polynomial_lsm)
from filters import alpha_beta_filter, alpha_beta_gamma_filter
from parsing import load_series
from plots import plot_estimation_errors, plot_real_filters, plot_synthetic_filters
from synthesis import add_anomalous_measurements, synthesize_series

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "parsed_minfin_usd_2025.csv"
REFRESH_DATA = False
SEED = 4
TREND_DEGREE = 2
ANOMALY_FRACTION = 0.1
Q_AV = 3
WINDOW_SIZE = 10
THRESHOLD_FACTOR = 3
CHECK_SIZE = 200
CHECK_NOISE_STD = 1.0
CHECK_LEVEL = 3
CHECK_SPEED = 0.5
CHECK_CURVATURE = 0.01
PRINT_PRECISION = 4


def print_header(title):
    print(f"\n----- {title} -----")


def fmt(value):
    return f"{value:.{PRINT_PRECISION}f}"


def load_input_data(rng):
    print_header("1. Вхідні дані")
    frame = load_series(DATA_PATH, REFRESH_DATA)
    real = frame["usd_rate"].to_numpy()
    x = np.linspace(0, 1, len(real))
    coefficients = fit_polynomial_lsm(x, real, TREND_DEGREE)
    ideal = evaluate_polynomial(x, coefficients)
    noise_std = np.std(real - ideal, ddof=1)
    synthetic = synthesize_series(x, coefficients, noise_std, rng)
    contaminated, positions = add_anomalous_measurements(synthetic, ideal, noise_std, ANOMALY_FRACTION, Q_AV, rng)
    print(f"Реальний ряд: {len(real)} спостережень, {frame['date'].min().date()} – {frame['date'].max().date()}")
    print(f"Синтетичний ряд: тренд степеня {TREND_DEGREE}, σ шуму = {fmt(noise_std)}")
    print(f"Додано АВ: {len(positions)} ({ANOMALY_FRACTION:.0%}), Q_AV = {Q_AV}")
    return frame["date"], real, ideal, contaminated, positions


def compensate_anomalies(real, contaminated, positions, ideal):
    print_header("2. Дослідження та компенсація аномалій")
    cleaned, detected = clean_by_sliding_window(contaminated, WINDOW_SIZE, THRESHOLD_FACTOR)
    quality = detection_quality(positions, detected)
    print(f"Синтетичний ряд: виявлено {len(detected)}, вірно {quality['hits']}, хибно {quality['false_alarms']}")
    print(f"Ймовірність вірного виявлення = {fmt(quality['recall'])}, хибної = {fmt(quality['false_rate'])}")
    print(f"СКВ відносно ідеалу: до очищення {fmt(np.std(contaminated - ideal, ddof=1))}, після {fmt(np.std(cleaned - ideal, ddof=1))}")
    real_cleaned, real_detected = clean_by_sliding_window(real, WINDOW_SIZE, THRESHOLD_FACTOR)
    print(f"Реальний ряд: позначено {len(real_detected)} з {len(real)}")
    return cleaned, real_cleaned


def build_filters(series):
    sigma = estimate_noise_std(series)
    adaptive, resets = alpha_beta_filter(series, sigma, THRESHOLD_FACTOR)
    estimates = {
        "αβ": alpha_beta_filter(series)[0],
        "αβγ": alpha_beta_gamma_filter(series),
        "Адаптивний αβ": adaptive,
    }
    return estimates, resets, sigma


def compare_filters_on_synthetic(ideal, cleaned):
    print_header("3. Фільтри на синтетичному ряді")
    estimates, resets, sigma = build_filters(cleaned)
    half = len(ideal) // 2
    print(f"σ шуму, оцінена за даними: {fmt(sigma)}")
    print("Фільтр        | Похибка M | Похибка σ | RMSE 1-ша пол. | RMSE 2-га пол.")
    for name, values in estimates.items():
        error = values - ideal
        print(f"{name:<13} | {fmt(error.mean()):>9} | {fmt(error.std(ddof=1)):>9} | "
              f"{fmt(np.sqrt(np.mean(error[:half] ** 2))):>14} | {fmt(np.sqrt(np.mean(error[half:] ** 2))):>14}")
    print(f"Скидань пам'яті адаптивного αβ: {len(resets)}")
    return estimates, resets, sigma


def check_unbiasedness(ideal, estimates, sigma):
    print_header("4. Незміщеність оцінки")
    for name, values in estimates.items():
        error = values - ideal
        print(f"{name:<13}: середня похибка = {fmt(error.mean())}, відношення до σ шуму = {fmt(error.mean() / sigma)}")


def smooth_real_data(real_cleaned):
    print_header("5. Фільтри на реальному ряді")
    estimates, resets, sigma = build_filters(real_cleaned)
    print(f"σ шуму, оцінена за даними: {fmt(sigma)}")
    print("Фільтр        | M (дані − оцінка) | σ (дані − оцінка)")
    for name, values in estimates.items():
        residual = real_cleaned - values
        print(f"{name:<13} | {fmt(residual.mean()):>17} | {fmt(residual.std(ddof=1)):>17}")
    print(f"Скидань пам'яті адаптивного αβ: {len(resets)}")
    return estimates


def verify_filters(rng):
    print_header("Верифікація: фільтри проти ідеальних трендів та МНК")
    index = np.arange(CHECK_SIZE, dtype=float)
    line = CHECK_LEVEL + CHECK_SPEED * index
    parabola = CHECK_LEVEL + CHECK_SPEED * index + CHECK_CURVATURE * index ** 2
    print(f"αβ на прямій без шуму: макс. похибка = {np.max(np.abs(alpha_beta_filter(line)[0] - line)):.2e}")
    print(f"αβγ на параболі без шуму: макс. похибка = {np.max(np.abs(alpha_beta_gamma_filter(parabola) - parabola)):.2e}")
    print(f"αβ на параболі без шуму: макс. похибка = {np.max(np.abs(alpha_beta_filter(parabola)[0] - parabola)):.2e}")
    noisy = parabola + rng.normal(0, CHECK_NOISE_STD, CHECK_SIZE)
    for name, filtered, degree in (("αβ", alpha_beta_filter(noisy)[0], 1), ("αβγ", alpha_beta_gamma_filter(noisy), 2)):
        lsm = evaluate_polynomial(index[-1:], fit_polynomial_lsm(index, noisy, degree))[0]
        print(f"{name}: остання оцінка {fmt(filtered[-1])}, МНК степеня {degree}: {fmt(lsm)}")


def main():
    rng = np.random.default_rng(SEED)
    dates, real, ideal, contaminated, positions = load_input_data(rng)
    cleaned, real_cleaned = compensate_anomalies(real, contaminated, positions, ideal)
    estimates, resets, sigma = compare_filters_on_synthetic(ideal, cleaned)
    check_unbiasedness(ideal, estimates, sigma)
    real_estimates = smooth_real_data(real_cleaned)
    verify_filters(rng)
    plot_synthetic_filters(ideal, cleaned, estimates)
    plot_estimation_errors(ideal, estimates, resets)
    plot_real_filters(dates, real_cleaned, real_estimates)


if __name__ == "__main__":
    main()
