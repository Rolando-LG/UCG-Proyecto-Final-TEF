from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE_DIR / "data" / "TEF_chile.csv"

TARGET = "is_fraud"
PREVIEW_COLUMNS = [
    "timestamp",
    "transaction_type",
    "sender_bank",
    "amount",
    "is_fraud",
    "fraud_type",
    "distance_from_home",
    "time_since_last_transaction",
    "last_24h_count",
    "amount_vs_avg",
    "sender_account_age",
    "sender_recent_account_changes",
    "sender_communication_channel_flag",
]
IDENTIFIER_COLUMNS = {
    "transaction_id",
    "sender_id",
    "receiver_id",
    "sender_device_id",
    "receiver_device_id",
    "sender_ip_address",
    "receiver_ip_address",
}
DERIVED_COLUMNS = {"hora", "clase", "alerta_comunicacion", "cambio_cuenta"}


def load_default_data() -> pd.DataFrame:
    """Carga el CSV de TEF incluido en el proyecto."""
    if not DEFAULT_CSV.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {DEFAULT_CSV}")
    return pd.read_csv(DEFAULT_CSV)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Carga un CSV enviado desde la interfaz."""
    return pd.read_csv(uploaded_file)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega campos derivados usados por la exploración. No modifica el archivo fuente."""
    if TARGET not in df.columns:
        raise ValueError("El dataset debe incluir la columna is_fraud (1 = fraude, 0 = legítima).")

    out = df.copy()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    if out[TARGET].dropna().empty:
        raise ValueError("La columna is_fraud no tiene valores numéricos utilizables.")

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["hora"] = out["timestamp"].dt.hour.astype("Int64")
    else:
        out["hora"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    out["clase"] = np.where(out[TARGET] == 1, "Fraude", "Legítima")

    if "sender_communication_channel_flag" in out.columns:
        canal = out["sender_communication_channel_flag"].fillna("none").astype(str)
        out["alerta_comunicacion"] = np.where(canal.eq("none"), "sin_alerta", "con_alerta")
    else:
        out["alerta_comunicacion"] = "no_disponible"

    if "sender_recent_account_changes" in out.columns:
        cambios = out["sender_recent_account_changes"].fillna("none").astype(str)
        out["cambio_cuenta"] = np.where(cambios.eq("none"), "sin_cambio", "con_cambio")
    else:
        out["cambio_cuenta"] = "no_disponible"

    return out


def dataset_profile(df: pd.DataFrame) -> dict:
    """Resume dimensiones, nulos, duplicados y la prevalencia de fraude."""
    fraud_mask = df[TARGET] == 1
    source_columns = [column for column in df.columns if column not in DERIVED_COLUMNS]
    source = df[source_columns]
    timestamp = df["timestamp"] if "timestamp" in df.columns else pd.Series(dtype="datetime64[ns]")
    valid_time = pd.to_datetime(timestamp, errors="coerce").dropna()
    return {
        "rows": int(df.shape[0]),
        "columns": int(len(source_columns)),
        "missing": int(source.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "frauds": int(fraud_mask.sum()),
        "fraud_rate": float(fraud_mask.mean()) if len(df) else 0.0,
        "period_start": valid_time.min() if not valid_time.empty else None,
        "period_end": valid_time.max() if not valid_time.empty else None,
    }


def variable_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de calidad: tipo, nulos y porcentaje, útil para el informe."""
    rows = []
    n = max(len(df), 1)
    columns = [column for column in df.columns if column not in DERIVED_COLUMNS]
    for column in columns:
        if column in IDENTIFIER_COLUMNS:
            kind = "Identificador"
        elif column in {"timestamp"} or pd.api.types.is_datetime64_any_dtype(df[column]):
            kind = "Fecha"
        elif pd.api.types.is_numeric_dtype(df[column]):
            kind = "Numérica"
        else:
            kind = "Categórica"
        nulls = int(df[column].isna().sum())
        rows.append({
            "variable": column,
            "tipo": kind,
            "nulos": nulls,
            "porcentaje_nulos": round(nulls / n * 100, 2),
        })
    report = pd.DataFrame(rows)
    return report.sort_values(["nulos", "variable"], ascending=[False, True]).reset_index(drop=True)


def preview_frame(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Muestra columnas analíticas y omite identificadores de dispositivo, IP y cuenta."""
    columns = [column for column in PREVIEW_COLUMNS if column in df.columns]
    view = df.loc[:, columns].head(n).copy()
    if "timestamp" in view.columns:
        view["timestamp"] = view["timestamp"].astype(str)
    return view
