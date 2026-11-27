import matplotlib.pyplot as plt
import numpy as np

FIGURE_SIZE = (10, 6)
HISTOGRAM_BINS = 20
COLORS = plt.get_cmap("tab10")


def finish_figure():
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_decomposition(index, values, parts, title, ylabel):
    figure, axes = plt.subplots(4, 1, figsize=FIGURE_SIZE, sharex=True)
    series = [("Ряд", values), ("Тренд", parts["trend"]), ("Сезонна складова", parts["seasonal"]),
              ("Залишок", parts["residual"])]
    for axis, (name, data) in zip(axes, series):
        axis.plot(index, data)
        axis.set_ylabel(name)
        axis.grid(alpha=0.3)
    axes[0].set_title(f"{title}, {ylabel}")
    axes[-1].set_xlabel("Час")
    finish_figure()


def plot_autocorrelations(autocorrelation, partial, bound):
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    for axis, values, name in zip(axes, (autocorrelation, partial), ("АКФ", "ЧАКФ")):
        axis.bar(range(len(values)), values)
        axis.axhline(bound, color="tab:red", linestyle="--", label="Межа значущості 95 %")
        axis.axhline(-bound, color="tab:red", linestyle="--")
        axis.set_title(f"{name} курсу USD")
        axis.set_xlabel("Лаг, дні")
        axis.set_ylabel(name)
        axis.grid(alpha=0.3)
        axis.legend()
    finish_figure()


def plot_month_clusters(names, features, labels):
    plt.figure(figsize=FIGURE_SIZE)
    for cluster in np.unique(labels):
        mask = labels == cluster
        plt.scatter(features[mask, 0], features[mask, 1], color=COLORS(cluster), s=60, label=f"Кластер {cluster + 1}")
    for name, point in zip(names, features):
        plt.annotate(name[5:], (point[0], point[1]), textcoords="offset points", xytext=(4, 4))
    plt.title("Кластеризація місячних сегментів курсу USD")
    plt.xlabel("Середній курс у місяці, грн")
    plt.ylabel("СКВ у місяці, грн")
    plt.grid(alpha=0.3)
    plt.legend()
    finish_figure()


def plot_usd_real_vs_synthetic(dates, real, synthetic, real_residual, synthetic_residual):
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(dates, real, label="Реальні дані")
    axes[0].plot(dates, synthetic, alpha=0.8, label="Синтезована модель")
    axes[0].set_title("Курс USD: реальний ряд і модель")
    axes[0].set_xlabel("Дата")
    axes[0].set_ylabel("Курс, грн за 1 USD")
    axes[1].hist(real_residual, bins=HISTOGRAM_BINS, density=True, alpha=0.5, label="Реальні залишки")
    axes[1].hist(synthetic_residual, bins=HISTOGRAM_BINS, density=True, alpha=0.5, label="Синтетичні залишки")
    axes[1].set_title("Залишки")
    axes[1].set_xlabel("Залишок, грн")
    axes[1].set_ylabel("Щільність")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.autofmt_xdate()
    finish_figure()


def plot_clustered_series(index, standardized, labels):
    plt.figure(figsize=FIGURE_SIZE)
    for name, label in zip(standardized.columns, labels):
        plt.plot(index, standardized[name], color=COLORS(label), label=f"{name} (кластер {label + 1})")
    plt.title("Нормовані ряди продажів за кластерами")
    plt.xlabel("Квартал")
    plt.ylabel("Нормоване значення")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    finish_figure()


def plot_correlation_heatmap(matrix):
    plt.figure(figsize=FIGURE_SIZE)
    image = plt.imshow(matrix.to_numpy(), vmin=0.5, vmax=1, cmap="viridis")
    plt.colorbar(image, label="Коефіцієнт кореляції Пірсона")
    plt.xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix)), matrix.index)
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            plt.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.title("Кореляційна матриця рядів продажів")
    finish_figure()


def plot_frame_real_vs_synthetic(index, real, synthetic, names):
    figure, axes = plt.subplots(len(names), 1, figsize=FIGURE_SIZE, sharex=True)
    for axis, name in zip(axes, names):
        axis.plot(index, real[name], label="Реальні дані")
        axis.plot(index, synthetic[name], alpha=0.8, label="Синтезована модель")
        axis.set_ylabel(name)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_title("Продажі іпотеки: реальні дані та синтезована модель")
    axes[-1].set_xlabel("Квартал")
    finish_figure()
