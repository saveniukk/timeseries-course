import matplotlib.pyplot as plt
import numpy as np

FIGURE_SIZE = (10, 5)
HISTOGRAM_BINS = 25
PDF_GRID_POINTS = 300
PDF_AXIS_MARGIN = 1.5


def finish_figure():
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_series_with_trend(dates, values, trend, degree):
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(dates, values, color="tab:blue", label="Курс USD (НБУ)")
    plt.plot(dates, trend, color="tab:red", linewidth=2, label=f"Тренд МНК, степінь {degree}")
    plt.title("Офіційний курс USD за 2025 рік та тренд")
    plt.xlabel("Дата")
    plt.ylabel("Курс, грн за 1 USD")
    plt.grid(alpha=0.3)
    plt.legend()
    finish_figure()


def plot_degree_selection(rows, selected_degree):
    degrees = [row["degree"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    axes[0].plot(degrees, [row["adjusted_r2"] for row in rows], marker="o", label="Скоригований R²")
    axes[0].set_ylabel("Скоригований R²")
    axes[1].plot(degrees, [row["bic"] for row in rows], marker="o", color="tab:orange", label="BIC")
    axes[1].axvline(selected_degree, color="tab:red", linestyle="--", label=f"Обрано: {selected_degree}")
    axes[1].set_ylabel("BIC")
    for axis in axes:
        axis.set_xlabel("Степінь полінома")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Вибір степеня тренду")
    finish_figure()


def plot_residual_laws(residuals, laws, law_objects):
    grid = np.linspace(residuals.min(), residuals.max(), PDF_GRID_POINTS)
    plt.figure(figsize=FIGURE_SIZE)
    heights, _, _ = plt.hist(residuals, bins=HISTOGRAM_BINS, density=True, alpha=0.5, label="Залишки")
    for item in laws:
        pdf = law_objects[item["law"]].pdf(grid, *item["params"])
        plt.plot(grid, pdf, linewidth=2, label=f"{item['law']} (KS={item['ks']:.4f})")
    plt.ylim(0, heights.max() * PDF_AXIS_MARGIN)
    plt.title("Гістограма залишків та підібрані закони розподілу")
    plt.xlabel("Залишок, грн")
    plt.ylabel("Щільність")
    plt.grid(alpha=0.3)
    plt.legend()
    finish_figure()


def plot_real_vs_synthetic(dates, real, synthetic):
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(dates, real, label="Реальні дані")
    plt.plot(dates, synthetic, alpha=0.8, label="Синтезована модель")
    plt.title("Реальний ряд і синтезована модель")
    plt.xlabel("Дата")
    plt.ylabel("Курс, грн за 1 USD")
    plt.grid(alpha=0.3)
    plt.legend()
    finish_figure()


def plot_histogram_comparison(real_residuals, synthetic_residuals):
    plt.figure(figsize=FIGURE_SIZE)
    plt.hist(real_residuals, bins=HISTOGRAM_BINS, density=True, alpha=0.5, label="Реальні залишки")
    plt.hist(synthetic_residuals, bins=HISTOGRAM_BINS, density=True, alpha=0.5, label="Синтетичні залишки")
    plt.title("Гістограми залишків: реальні дані та модель")
    plt.xlabel("Залишок, грн")
    plt.ylabel("Щільність")
    plt.grid(alpha=0.3)
    plt.legend()
    finish_figure()
