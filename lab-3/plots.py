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


def plot_synthetic_filters(ideal, cleaned, estimates):
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(cleaned, color="tab:gray", alpha=0.6, label="Очищені дані")
    plt.plot(ideal, color="black", linewidth=2, label="Ідеальний тренд")
    for name, values in estimates.items():
        plt.plot(values, label=name)
    style_axes("Згладжування синтезованого ряду фільтрами", "Номер вимірювання", "Курс, грн за 1 USD")
    finish_figure()


def plot_estimation_errors(ideal, estimates, resets):
    plt.figure(figsize=FIGURE_SIZE)
    for name, values in estimates.items():
        plt.plot(values - ideal, label=name)
    plt.axhline(0, color="black", linewidth=1)
    for index in resets:
        plt.axvline(index, color="tab:red", alpha=0.15)
    style_axes("Похибка оцінки відносно ідеального тренду (червоні лінії – скидання пам'яті)",
               "Номер вимірювання", "Похибка, грн")
    finish_figure()


def plot_real_filters(dates, cleaned, estimates):
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(dates, cleaned, color="tab:gray", alpha=0.6, label="Курс USD (очищений)")
    for name, values in estimates.items():
        plt.plot(dates, values, label=name)
    style_axes("Згладжування реального ряду USD 2025", "Дата", "Курс, грн за 1 USD")
    finish_figure()
