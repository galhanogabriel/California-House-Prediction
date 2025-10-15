"""Arquivo de construção de gráficos para distribuições de probabilidade.
Construído para o módulo de Estatística do curso de Ciência de Dados ministrado
por Francisco Bustamante.

Pode ser modificado e distribuído livremente, desde que mantida a referência ao autor
a partir deste comentário de cabeçalho.

Autor: Francisco Bustamante
https://www.linkedin.com/in/flsbustamante
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import seaborn as sns
from sklearn.metrics import PredictionErrorDisplay
from src.helpers import RANDOM_STATE

sns.set_theme(palette="bright")

PALETTE = "coolwarm"

SCATTER_ALPHA = 0.2

def plot_coefficients(df_coefficients, title="Coefficients"):
    df_coefficients.plot.barh()

    plt.title(title)
    plt.axvline(x=0, color="1")
    plt.xlabel("Coefficients")

    plt.gca().get_legend().remove()

    plt.show()

def plot_compare_model_metrics(df_results):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    compare_metrics = [
        "time_seconds",
        "test_r2",
        "test_neg_mean_absolute_error",
        "test_neg_root_mean_squared_error",
    ]

    metric_names = [
        "Tempo (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    for ax, metric, name in zip(axs.flatten(), compare_metrics, metric_names):
        sns.boxplot(
            x="model",
            y=metric,
            data=df_results,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(name)
        ax.set_ylabel(name)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.show()

def plot_estimator_residuals(estimator, X, y, eng_formatter=False, sample_fraction=0.25):
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=sample_fraction,
    )

    error_display_02 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="actual_vs_predicted",
        ax=axs[2],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=sample_fraction,
    )

    residuals = error_display_01.y_true - error_display_01.y_pred

    sns.histplot(residuals,kde=True, ax=axs[0])

    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())

    plt.tight_layout()

    plt.show()