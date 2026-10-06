from __future__ import annotations

import io

import streamlit as st

from src.agent import AgenteDatos, offline_summary
from src.data_loader import (
    dataset_profile,
    enrich,
    load_default_data,
    load_uploaded_data,
    preview_frame,
    variable_quality,
)
from src.eda import (
    amount_by_class,
    build_agent_context,
    correlation_matrix,
    fraud_composition,
    fraud_rate_by,
    missing_report,
    numeric_summary,
    target_correlations,
    train_fraud_baseline,
)
from src.visualizations import (
    amount_histogram,
    confusion_heatmap,
    correlation_heatmap,
    fraud_count_bar,
    fraud_rate_bar,
    metrics_bar,
    numeric_histogram,
    score_distribution,
)

st.set_page_config(page_title="TEF Chile · Fraude", page_icon="💸", layout="wide")

SEGMENTOS = {
    "Tipo de transacción": ("transaction_type", False),
    "Banco emisor": ("sender_bank", False),
    "Alerta de comunicación": ("alerta_comunicacion", False),
    "Cambio reciente de cuenta": ("cambio_cuenta", False),
    "Hora del día": ("hora", True),
    "Tipo de fraude (casos positivos)": ("fraud_type", False),
}
DISTRIBUCIONES = [
    "distance_from_home",
    "time_since_last_transaction",
    "sender_account_age",
    "amount_vs_avg",
    "last_24h_count",
    "distance_sender_receiver",
]


@st.cache_data(show_spinner="Cargando y analizando el dataset TEF...")
def prepare_default():
    frame = enrich(load_default_data())
    return _with_model(frame)


@st.cache_data(show_spinner="Procesando el CSV cargado...")
def prepare_upload(content: bytes):
    frame = enrich(load_uploaded_data(io.BytesIO(content)))
    return _with_model(frame)


def _with_model(frame):
    try:
        model = train_fraud_baseline(frame)
        error = ""
    except Exception as exc:
        model = None
        error = str(exc)
    return frame, model, error


def build_tools(frame, model_result):
    def calidad():
        report = missing_report(frame)
        return report.loc[report["nulos"] > 0].head(5).to_dict("records")

    def segmentos():
        return {
            "transaction_type": fraud_rate_by(frame, "transaction_type").to_dict("records"),
            "sender_bank": fraud_rate_by(frame, "sender_bank").to_dict("records"),
            "alerta_comunicacion": fraud_rate_by(frame, "alerta_comunicacion").to_dict("records"),
            "cambio_cuenta": fraud_rate_by(frame, "cambio_cuenta").to_dict("records"),
        }

    def modelo():
        if model_result is None:
            return {"estado": "El modelo no está disponible para este archivo."}
        return {
            "metricas": model_result["metrics"],
            "coeficientes": model_result["coefficients"].to_dict("records"),
            "variables_excluidas": model_result["excluded"],
        }

    return {"calidad": calidad, "segmentos": segmentos, "modelo": modelo}


st.title("Explorador de fraude en transferencias electrónicas")
st.caption("TEF Chile · dataset sintético · Streamlit, scikit-learn y Ollama")

with st.sidebar:
    st.header("Configuración")
    uploaded = st.file_uploader("CSV opcional", type=["csv"])
    model_name = st.text_input("Modelo Ollama", value="llama3.2:3b")
    st.caption("Si Ollama no está activo, el análisis y los gráficos se mantienen. El agente usa un resumen calculado en Python.")

try:
    df, model_result, model_error = prepare_upload(uploaded.getvalue()) if uploaded else prepare_default()
except Exception as exc:
    st.error(f"No fue posible cargar el dataset: {exc}")
    st.stop()

profile = dataset_profile(df)
agente = AgenteDatos(modelo=model_name, herramientas=build_tools(df, model_result))
ollama_ok, installed_models, _detail = agente.estado()

with st.sidebar:
    st.markdown("---")
    if ollama_ok:
        st.success("Ollama disponible")
        if installed_models:
            st.caption("Modelos: " + ", ".join(installed_models[:5]))
            if not any(model_name == name or name.startswith(model_name) for name in installed_models):
                st.caption(f"Falta el modelo indicado. En una terminal: ollama pull {model_name}")
        else:
            st.warning("No hay modelos descargados")
            st.caption(f"En una terminal: ollama pull {model_name}")
    else:
        st.warning("Ollama no detectado")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transacciones", f"{profile['rows']:,}".replace(",", "."))
c2.metric("Fraudes", f"{profile['frauds']:,}".replace(",", "."))
c3.metric("Tasa de fraude", f"{profile['fraud_rate'] * 100:.2f} %")
c4.metric("Valores nulos", f"{profile['missing']:,}".replace(",", "."))

tab_datos, tab_exploracion, tab_modelo, tab_agente, tab_arquitectura = st.tabs([
    "Datos", "Exploración", "Modelo", "Agente", "Arquitectura",
])

with tab_datos:
    st.subheader("Dataset de trabajo")
    if uploaded is None:
        st.info(
            "Se usa el archivo incluido data/TEF_chile.csv. El diccionario lo describe como dataset sintético "
            "de transferencias electrónicas de fondos, con montos en CLP. "
            f"Periodo: {profile['period_start']} a {profile['period_end']}."
        )
    else:
        st.info("Se está utilizando el CSV cargado desde la interfaz. Debe incluir la columna is_fraud.")

    st.markdown("#### Muestra analítica")
    st.caption("La vista omite identificadores de cuenta, dispositivo e IP.")
    st.dataframe(preview_frame(df), width="stretch", hide_index=True)

    st.markdown("#### Calidad de datos")
    st.caption("La tabla lista las columnas del CSV. La aplicación agrega hora, clase, alerta_comunicacion y cambio_cuenta para explorar.")
    st.dataframe(variable_quality(df), width="stretch", hide_index=True)

    st.markdown("#### Resumen numérico")
    st.dataframe(numeric_summary(df), width="stretch", hide_index=True)

with tab_exploracion:
    st.subheader("Segmentos y señales de fraude")
    st.caption(
        "Las tasas usan todas las transacciones. Los histogramas de monto y de variables continuas "
        "se dibujan con una muestra reproducible de hasta 15.000 filas para mantener la interfaz ágil. "
        "La alerta de comunicación marca fraude en todos sus casos, y un cambio reciente de cuenta eleva la tasa. "
        "Esas dos señales se exploran aquí y quedan fuera del modelo."
    )
    etiqueta = st.selectbox("Segmento", list(SEGMENTOS))
    columna, cronologico = SEGMENTOS[etiqueta]
    if columna == "fraud_type":
        composicion = fraud_composition(df, columna)
        st.plotly_chart(
            fraud_count_bar(composicion, columna, "Composición de los casos de fraude"),
            width="stretch",
        )
        st.caption("fraud_type solo clasifica los casos ya marcados como fraude. No se usa como variable predictora.")
        st.dataframe(composicion, width="stretch", hide_index=True)
    else:
        tasas = fraud_rate_by(df, columna, chronological=cronologico)
        st.plotly_chart(
            fraud_rate_bar(tasas, columna, f"Tasa de fraude por {etiqueta.lower()}"),
            width="stretch",
        )
        st.dataframe(tasas, width="stretch", hide_index=True)

    st.markdown("#### Monto de la transacción")
    st.dataframe(amount_by_class(df), width="stretch", hide_index=True)
    if "amount" in df.columns:
        st.plotly_chart(amount_histogram(df), width="stretch")

    disponibles = [columna for columna in DISTRIBUCIONES if columna in df.columns]
    if disponibles:
        variable = st.selectbox("Otra variable numérica", disponibles)
        st.plotly_chart(numeric_histogram(df, variable), width="stretch")

    st.markdown("#### Correlación con is_fraud")
    st.caption("Se excluye avg_amount_last_7d de esta matriz porque concentra la mayoría de los nulos.")
    st.dataframe(target_correlations(df).head(10), width="stretch", hide_index=True)
    st.plotly_chart(correlation_heatmap(correlation_matrix(df)), width="stretch")

with tab_modelo:
    st.subheader("Línea base de clasificación")
    st.markdown(
        "Regresión logística con `class_weight='balanced'` y umbral 0.5. "
        "El entrenamiento usa el 75 % de las filas y la evaluación el 25 %, con muestreo estratificado. "
        "El modelo usa monto, distancia, antigüedad, actividad reciente, tipo de transacción y banco. "
        "Quedan fuera `fraud_type`, la alerta de comunicación y el cambio de cuenta, porque en este dataset "
        "describen el escenario etiquetado. También quedan fuera los deltas de saldo: "
        "`sender_balance_delta` coincide con el monto en casi todas las filas."
    )
    if model_error:
        st.warning(model_error)
    elif model_result is None:
        st.warning("No fue posible entrenar el modelo con la configuración actual.")
    else:
        metricas = model_result["metrics"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Exactitud", f"{metricas['accuracy']:.3f}")
        m2.metric("Precisión", f"{metricas['precision']:.3f}")
        m3.metric("Recall", f"{metricas['recall']:.3f}")
        m4.metric("F1", f"{metricas['f1']:.3f}")
        m5.metric("ROC AUC", f"{metricas['roc_auc']:.3f}")
        st.caption(
            f"Entrenamiento: {model_result['n_train']:,} transacciones · "
            f"Prueba: {model_result['n_test']:,} · "
            f"Fraudes en prueba: {model_result['fraud_test']:,}. "
            "La precisión cuenta cuántas alertas del modelo son fraude real. "
            "El recall cuenta cuántos fraudes de la prueba fueron recuperados."
            .replace(",", ".")
        )
        st.plotly_chart(metrics_bar(metricas), width="stretch")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(confusion_heatmap(model_result["confusion"]), width="stretch")
        with right:
            st.plotly_chart(score_distribution(model_result["score_sample"]), width="stretch")
        st.markdown("#### Coeficientes de mayor magnitud")
        st.caption(
            "Coeficientes en log-odds después de escalar las variables numéricas. "
            "Un valor positivo empuja la estimación hacia fraude. "
            "Si dos variables se mueven juntas, el signo de una de ellas se lee en conjunto con la otra."
        )
        st.dataframe(model_result["coefficients"], width="stretch", hide_index=True)

with tab_agente:
    st.subheader("Agente local para interpretación")
    contexto = build_agent_context(df, profile, model_result)
    st.caption("El agente recibe un contexto estadístico y el resultado de funciones autorizadas. No recibe el dataset completo.")

    presets = [
        "Resume los principales hallazgos del dataset.",
        "¿Qué tipos de transacción y bancos muestran mayor tasa de fraude?",
        "¿Qué señales de monto o comportamiento acompañan a las transacciones fraudulentas?",
        "Interpreta la precisión, el recall y el ROC AUC del modelo.",
        "¿Qué límites tiene este análisis por tratarse de un dataset sintético?",
    ]
    choice = st.selectbox("Pregunta sugerida", ["Escribir otra pregunta..."] + presets)
    question = st.text_area(
        "Pregunta",
        value="" if choice == "Escribir otra pregunta..." else choice,
        height=100,
    )
    if st.button("Analizar con el agente", type="primary"):
        if not question.strip():
            st.warning("Escribe una pregunta.")
        elif ollama_ok:
            with st.spinner("Consultando el modelo local..."):
                try:
                    st.markdown(agente.responder(question.strip(), contexto))
                except Exception as exc:
                    st.error(f"Ollama respondió con un error: {exc}")
                    st.markdown(offline_summary(contexto))
        else:
            st.markdown(offline_summary(contexto))
            st.caption("Respuesta de respaldo: Ollama no está disponible en este equipo.")

with tab_arquitectura:
    st.subheader("Arquitectura modular")
    st.code(
        """data/TEF_chile.csv
        ↓
src/data_loader.py
        ↓
src/eda.py
   ↙            ↘
src/visualizations.py   contexto resumido
   ↓                    ↓
Streamlit  ←  src/agent.py  →  src/llm_client.py  →  Ollama local""",
        language="text",
    )
    st.markdown(
        """
- **Imperativo:** validación del archivo, elección de segmento y decisión cuando Ollama no responde.
- **Funcional:** funciones de calidad, tasas de fraude, correlaciones, métricas y gráficos.
- **Orientado a objetos:** la clase `AgenteDatos` guarda el modelo, elige herramientas autorizadas y pide la interpretación a `OllamaClient`.
- **Modularidad:** la interfaz queda en `app.py`. Cada responsabilidad vive en un módulo de `src/`.
        """
    )

st.markdown("---")
st.caption(
    "Trabajo académico sobre un dataset sintético. No es un sistema de prevención de fraude ni una herramienta de decisión bancaria."
)
