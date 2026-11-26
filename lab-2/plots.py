import matplotlib.pyplot as plt

FIGURE_SIZE = (10, 5)


def finish_figure():
    plt.tight_layout()
    plt.show()
    plt.close()


def style_axes(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()


def plot_real_forecast(x, y, trend, x_future, forecast, degree):
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(x, y, label="Курс USD (НБУ), 2025")
    plt.plot(x, trend, color="tab:red", linewidth=2, label=f"Тренд МНК, степінь {degree}")
    plt.plot(x_future, forecast, color="tab:green", linewidth=2, label="Прогноз на 0,5 інтервалу")
    plt.axvline(x[-1], color="gray", linestyle="--")
    style_axes("Реальні дані: тренд і прогноз", "Нормований час x", "Курс, грн за 1 USD")
    finish_figure()


def plot_degree_selection(rows, selected_degree):
    degrees = [row["degree"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    axes[0].plot(degrees, [row["r2"] for row in rows], marker="o", label="R²")
    axes[0].set_ylabel("R² на всій вибірці")
    axes[1].plot(degrees, [row["test_rmse"] for row in rows], marker="o", color="tab:orange", label="RMSE на відкладеній частині")
    axes[1].axvline(selected_degree, color="tab:red", linestyle="--", label=f"Обрано: {selected_degree}")
    axes[1].set_ylabel("RMSE, грн")
    for axis in axes:
        axis.set_xlabel("Степінь полінома")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Вибір степеня полінома")
    finish_figure()


def plot_anomaly_cleaning(x, ideal, contaminated, cleaned, true_positions, detected_positions):
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(x, contaminated, color="tab:gray", alpha=0.7, label="Модель з аномальними вимірами")
    plt.plot(x, cleaned, color="tab:blue", label="Після очищення")
    plt.plot(x, ideal, color="black", linewidth=2, label="Ідеальний тренд")
    plt.scatter(x[true_positions], contaminated[true_positions], color="tab:orange", s=30, label="Задані АВ")
    plt.scatter(x[detected_positions], contaminated[detected_positions], facecolors="none", edgecolors="tab:red", s=80,
                label="Виявлені АВ")
    style_axes("Виявлення та очищення аномальних вимірів", "Нормований час x", "Курс, грн за 1 USD")
    finish_figure()


def plot_extrapolation(x, x_future, ideal, ideal_forecast, forecast_raw, forecast_clean, degree):
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(x, ideal, color="black", label="Ідеальний тренд")
    plt.plot(x_future, ideal_forecast, color="black", linestyle="--", label="Ідеальна екстраполяція")
    plt.plot(x_future, forecast_raw, color="tab:orange", label="Прогноз за даними з АВ")
    plt.plot(x_future, forecast_clean, color="tab:green", label="Прогноз за очищеними даними")
    style_axes(f"Прогноз синтезованої моделі (степінь {degree})", "Нормований час x", "Курс, грн за 1 USD")
    finish_figure()

