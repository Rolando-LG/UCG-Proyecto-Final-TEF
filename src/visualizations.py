from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CLASS_COLORS = {"Legítima": "#4C78A8", "Fraude": "#E45756"}


def _style(fig, title: str):
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=56, b=20),
        legend_title_text="",
    )
    return fig


def fraud_rate_bar(summary: pd.DataFrame, column: str, title: str):
    fig = px.bar(
        summary,
        x=column,
        y="tasa_fraude",
        text="tasa_fraude",
        labels={column: column, "tasa_fraude": "Tasa de fraude (%)"},
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", marker_color="#E45756")
    fig.update_layout(yaxis_title="Tasa de fraude (%)", xaxis_title="")
    return _style(fig, title)


def fraud_count_bar(summary: pd.DataFrame, column: str, title: str):
    fig = px.bar(
        summary,
        x=column,
        y="casos",
        text="casos",
        labels={column: column, "casos": "Casos de fraude"},
    )
    fig.update_traces(textposition="outside", marker_color="#F2A2A2")
    fig.update_layout(xaxis_title="", yaxis_title="Casos")
    return _style(fig, title)


def amount_histogram(df: pd.DataFrame):
    plot_df = df.loc[df["amount"] > 0, ["amount", "clase"]].copy()
    if len(plot_df) > 15000:
        plot_df = plot_df.sample(15000, random_state=42)
    fig = px.histogram(
        plot_df,
        x="amount",
        color="clase",
        nbins=40,
        barmode="overlay",
        log_x=True,
        opacity=0.7,
        color_discrete_map=CLASS_COLORS,
        labels={"amount": "Monto (CLP, escala log)", "clase": "Clase"},
    )
    return _style(fig, "Distribución del monto según la clase")


def numeric_histogram(df: pd.DataFrame, variable: str):
    plot_df = df.loc[:, [variable, "clase"]].dropna()
    if len(plot_df) > 15000:
        plot_df = plot_df.sample(15000, random_state=42)
    fig = px.histogram(
        plot_df,
        x=variable,
        color="clase",
        nbins=30,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map=CLASS_COLORS,
    )
    return _style(fig, f"Distribución de {variable}")


def correlation_heatmap(matrix: pd.DataFrame):
    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlación entre señales numéricas e is_fraud",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=56, b=20))
    return fig


def metrics_bar(metrics: dict):
    frame = pd.DataFrame({
        "métrica": ["Exactitud", "Precisión", "Recall", "F1", "ROC AUC"],
        "valor": [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        ],
    })
    fig = px.bar(frame, x="métrica", y="valor", text="valor", range_y=[0, 1])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_color="#4C78A8")
    return _style(fig, "Métricas en el conjunto de prueba")


def confusion_heatmap(confusion: pd.DataFrame):
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion.values,
            x=list(confusion.columns),
            y=list(confusion.index),
            text=confusion.values,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    return _style(fig, "Matriz de confusión")


def score_distribution(scores: pd.DataFrame):
    fig = px.histogram(
        scores,
        x="probabilidad_fraude",
        color="clase_real",
        nbins=30,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map=CLASS_COLORS,
        labels={"probabilidad_fraude": "Probabilidad estimada de fraude", "clase_real": "Clase real"},
    )
    return _style(fig, "Probabilidad estimada según la clase real")
