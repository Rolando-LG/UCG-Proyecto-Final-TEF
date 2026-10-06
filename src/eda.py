from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import TARGET

# Señales de comportamiento y monto. fraud_type no entra: solo está informado en los fraudes
# y revelaría la etiqueta. Los indicadores de comunicación y cambio de cuenta se exploran
# aparte porque en este dataset sintético describen el escenario del caso.
NUMERIC_FEATURES = [
    "amount",
    "distance_from_home",
    "time_since_last_transaction",
    "sender_account_age",
    "sender_avg_daily_transactions",
    "sender_avg_daily_amount",
    "last_24h_count",
    "amount_vs_avg",
    "recent_tx_ratio",
    "distance_sender_receiver",
    "sender_balance_before",
]
CATEGORICAL_FEATURES = ["transaction_type", "sender_bank"]
SUMMARY_COLUMNS = NUMERIC_FEATURES + [
    TARGET,
    "avg_amount_last_7d",
    "sender_balance_delta",
    "receiver_balance_delta",
]


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "variable": df.columns,
        "nulos": df.isna().sum().values,
        "porcentaje": (df.isna().mean().values * 100).round(2),
    })
    return report.sort_values(["nulos", "variable"], ascending=[False, True]).reset_index(drop=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in SUMMARY_COLUMNS if column in df.columns]
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    if numeric.empty:
        return pd.DataFrame()
    out = numeric.describe().T.reset_index().rename(columns={"index": "variable"})
    return out.round(3)


def fraud_rate_by(df: pd.DataFrame, column: str, chronological: bool = False) -> pd.DataFrame:
    """Calcula transacciones, fraudes y tasa porcentual por segmento."""
    if column not in df.columns or TARGET not in df.columns:
        return pd.DataFrame(columns=[column, "transacciones", "fraudes", "tasa_fraude"])

    work = df.dropna(subset=[column]).copy()
    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce").fillna(0)
    grouped = (
        work.groupby(column, dropna=False)[TARGET]
        .agg(transacciones="count", fraudes="sum", tasa_fraude="mean")
        .reset_index()
    )
    grouped["fraudes"] = grouped["fraudes"].astype(int)
    grouped["transacciones"] = grouped["transacciones"].astype(int)
    grouped["tasa_fraude"] = (grouped["tasa_fraude"] * 100).round(2)
    if chronological:
        return grouped.sort_values(column).reset_index(drop=True)
    return grouped.sort_values(["tasa_fraude", "transacciones"], ascending=[False, False]).reset_index(drop=True)


def fraud_composition(df: pd.DataFrame, column: str = "fraud_type") -> pd.DataFrame:
    """Distribución de una categoría dentro de los casos marcados como fraude."""
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "casos", "porcentaje"])
    frauds = df.loc[df[TARGET] == 1, column].fillna("sin_dato").astype(str)
    if frauds.empty:
        return pd.DataFrame(columns=[column, "casos", "porcentaje"])
    counts = frauds.value_counts().rename_axis(column).reset_index(name="casos")
    counts["porcentaje"] = (counts["casos"] / counts["casos"].sum() * 100).round(2)
    return counts


def amount_by_class(df: pd.DataFrame) -> pd.DataFrame:
    if "amount" not in df.columns or "clase" not in df.columns:
        return pd.DataFrame()
    summary = (
        df.groupby("clase")["amount"]
        .agg(
            transacciones="count",
            mediana="median",
            media="mean",
            percentil_95=lambda series: series.quantile(0.95),
        )
        .reset_index()
        .round(2)
    )
    return summary


def _correlation_columns(df: pd.DataFrame) -> list[str]:
    extra = ["sender_balance_delta", "receiver_balance_delta"]
    return [column for column in NUMERIC_FEATURES + extra + [TARGET] if column in df.columns]


def target_correlations(df: pd.DataFrame) -> pd.DataFrame:
    columns = _correlation_columns(df)
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    if TARGET not in numeric.columns:
        return pd.DataFrame(columns=["variable", "correlacion"])
    corr = numeric.corr(numeric_only=True)[TARGET].drop(labels=[TARGET], errors="ignore")
    return (
        corr.rename("correlacion")
        .reset_index()
        .rename(columns={"index": "variable"})
        .assign(abs_correlacion=lambda frame: frame["correlacion"].abs())
        .sort_values("abs_correlacion", ascending=False)
        .drop(columns="abs_correlacion")
        .reset_index(drop=True)
    )


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    columns = _correlation_columns(df)
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(numeric_only=True).round(2)


def train_fraud_baseline(df: pd.DataFrame) -> dict:
    """Regresión logística balanceada para estimar la probabilidad de fraude.

    Devuelve exactitud, precisión, recall, F1 y ROC AUC de la clase fraude.
    """
    features = [column for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES if column in df.columns]
    missing = [column for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES if column not in df.columns]
    if missing:
        raise ValueError("Faltan columnas requeridas por el modelo: " + ", ".join(missing))

    work = df.dropna(subset=[TARGET]).copy()
    y = work[TARGET].astype(int)
    if y.nunique() < 2 or len(work) < 50:
        raise ValueError("No hay suficientes casos de ambas clases para entrenar.")

    X = work[features]
    numeric = [column for column in NUMERIC_FEATURES if column in X.columns]
    categorical = [column for column in CATEGORICAL_FEATURES if column in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    labels = [0, 1]
    matrix = confusion_matrix(y_test, pred, labels=labels)
    confusion = pd.DataFrame(
        matrix,
        index=["Real legítima", "Real fraude"],
        columns=["Predicha legítima", "Predicha fraude"],
    )

    names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pd.DataFrame({
        "variable": pd.Index(names).str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False),
        "coeficiente": model.named_steps["classifier"].coef_.ravel(),
    })
    coefficients["abs_coeficiente"] = coefficients["coeficiente"].abs()
    coefficients = (
        coefficients.sort_values("abs_coeficiente", ascending=False)
        .drop(columns="abs_coeficiente")
        .head(12)
        .round(3)
        .reset_index(drop=True)
    )

    plot_frame = pd.DataFrame({
        "clase_real": np.where(y_test.to_numpy() == 1, "Fraude", "Legítima"),
        "probabilidad_fraude": proba,
    })
    fraud_scores = plot_frame[plot_frame["clase_real"] == "Fraude"]
    legit_scores = plot_frame[plot_frame["clase_real"] == "Legítima"]
    if len(legit_scores) > 3000:
        legit_scores = legit_scores.sample(3000, random_state=42)
    score_sample = pd.concat([fraud_scores, legit_scores], ignore_index=True)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    return {
        "metrics": {key: round(value, 3) for key, value in metrics.items()},
        "confusion": confusion,
        "coefficients": coefficients,
        "score_sample": score_sample,
        "features_numeric": numeric,
        "features_categorical": categorical,
        "excluded": [
            "fraud_type",
            "sender_communication_channel_flag",
            "sender_recent_account_changes",
            "sender_balance_delta",
            "receiver_balance_delta",
        ],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fraud_test": int(y_test.sum()),
        "threshold": 0.5,
    }


def _records(frame: pd.DataFrame, n: int = 8) -> list[dict]:
    if frame.empty:
        return []
    exported = frame.head(n).copy()
    for column in exported.columns:
        if pd.api.types.is_datetime64_any_dtype(exported[column]):
            exported[column] = exported[column].astype(str)
    return exported.round(3).to_dict(orient="records") if _has_numeric(exported) else exported.to_dict(orient="records")


def _has_numeric(frame: pd.DataFrame) -> bool:
    return not frame.select_dtypes(include="number").empty


def build_agent_context(df: pd.DataFrame, profile: dict, model_result: dict | None) -> dict:
    """Contexto compacto. El agente no recibe el CSV ni identificadores de IP o dispositivo."""
    start = profile.get("period_start")
    end = profile.get("period_end")
    nulls = missing_report(df)
    nulls = nulls[nulls["nulos"] > 0]
    amounts = amount_by_class(df)

    context = {
        "dataset": {
            "nombre": "Dataset sintético de transferencias electrónicas de fondos (TEF), montos en CLP",
            "filas": profile["rows"],
            "columnas": profile["columns"],
            "duplicados": profile["duplicates"],
            "fraudes": profile["frauds"],
            "tasa_fraude_pct": round(profile["fraud_rate"] * 100, 2),
            "nulos_totales": profile["missing"],
            "periodo": f"{start} a {end}" if start is not None else "no disponible",
            "nota": (
                "Dataset sintético de uso académico. Los cálculos los hizo Python. "
                "No es un sistema de detección para operación bancaria."
            ),
        },
        "calidad_nulos": _records(nulls, 5),
        "montos_clp": _records(amounts, 4),
        "tasa_por_tipo_transaccion": _records(fraud_rate_by(df, "transaction_type"), 6),
        "tasa_por_banco": _records(fraud_rate_by(df, "sender_bank"), 8),
        "tasa_por_alerta_comunicacion": _records(fraud_rate_by(df, "alerta_comunicacion"), 4),
        "tasa_por_cambio_cuenta": _records(fraud_rate_by(df, "cambio_cuenta"), 4),
        "composicion_tipos_de_fraude": _records(fraud_composition(df), 8),
        "correlaciones_con_is_fraud": _records(target_correlations(df), 8),
    }
    if model_result:
        context["modelo"] = {
            "algoritmo": "Regresión logística con class_weight=balanced",
            "umbral": model_result["threshold"],
            "metricas_clase_fraude": model_result["metrics"],
            "n_train": model_result["n_train"],
            "n_test": model_result["n_test"],
            "fraudes_en_prueba": model_result["fraud_test"],
            "variables_numericas": model_result["features_numeric"],
            "variables_categoricas": model_result["features_categorical"],
            "variables_excluidas_para_evitar_fuga": model_result["excluded"],
            "coeficientes_mas_altos": _records(model_result["coefficients"], 8),
            "matriz_confusion": model_result["confusion"].to_dict(),
        }
    return context
