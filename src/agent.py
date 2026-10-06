from __future__ import annotations

from collections.abc import Callable

from src.llm_client import OllamaClient, dumps_context

Tool = Callable[[], object]

SYSTEM_PROMPT = (
    "Eres un asistente académico de análisis de datos. "
    "Responde en español, de forma breve y técnica, usando solo el contexto JSON entregado. "
    "Esos números ya fueron calculados por Python: no los recalcules ni inventes variables, tasas o métricas. "
    "Distingue los hechos observados de cualquier hipótesis. "
    "El dataset es sintético y sirve para un trabajo universitario sobre transferencias electrónicas. "
    "No entregues instrucciones para cometer fraude, evadir controles o vulnerar sistemas. "
    "No presentes el resultado como una decisión operativa de un banco."
)


class AgenteDatos:
    """Coordina herramientas analíticas autorizadas y la interpretación del modelo local."""

    def __init__(self, modelo: str, herramientas: dict[str, Tool] | None = None):
        self.modelo = modelo
        self.herramientas = herramientas or {}
        self.cliente = OllamaClient(model=modelo)

    def estado(self) -> tuple[bool, list[str], str]:
        return self.cliente.status()

    def _seleccionar_herramientas(self, pregunta: str) -> list[str]:
        texto = pregunta.lower()
        reglas = [
            ("calidad", ("nulo", "calidad", "faltante", "missing")),
            ("segmentos", ("banco", "tipo", "segment", "tasa", "canal", "cuenta", "hora")),
            ("modelo", ("modelo", "métrica", "metrica", "precisión", "precision", "recall", "auc", "f1")),
        ]
        elegidas = [
            nombre
            for nombre, claves in reglas
            if nombre in self.herramientas and any(clave in texto for clave in claves)
        ]
        if elegidas:
            return elegidas
        return [nombre for nombre in ("segmentos", "modelo", "calidad") if nombre in self.herramientas]

    def responder(self, pregunta: str, contexto: dict) -> str:
        usadas = self._seleccionar_herramientas(pregunta)
        paquete = dict(contexto)
        paquete["herramientas_usadas"] = usadas
        for nombre in usadas:
            paquete[f"resultado_{nombre}"] = self.herramientas[nombre]()
        mensaje = f"CONTEXTO:\n{dumps_context(paquete)}\n\nPREGUNTA:\n{pregunta}"
        return self.cliente.chat(SYSTEM_PROMPT, mensaje)


def _miles(value: object) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(value)


def offline_summary(contexto: dict) -> str:
    """Interpretación de respaldo construida solo con cifras ya calculadas."""
    datos = contexto.get("dataset", {})
    modelo = contexto.get("modelo", {})
    metricas = modelo.get("metricas_clase_fraude", {}) if modelo else {}
    montos = contexto.get("montos_clp", [])
    tipos = contexto.get("composicion_tipos_de_fraude", [])
    nulos = contexto.get("calidad_nulos", [])

    texto = (
        f"El dataset sintético de TEF contiene {_miles(datos.get('filas', 0))} transacciones y "
        f"{_miles(datos.get('fraudes', 0))} fraudes "
        f"({datos.get('tasa_fraude_pct', 0)} % del total)."
    )
    if nulos:
        primero = nulos[0]
        texto += (
            f" La variable con más valores nulos es {primero.get('variable')} "
            f"({primero.get('porcentaje')} %)."
        )
    if montos:
        texto += " Montos en CLP por clase: " + "; ".join(
            f"{fila.get('clase')} mediana {fila.get('mediana')}" for fila in montos
        ) + "."
    if tipos:
        texto += " Entre los fraudes, el tipo más frecuente en el resumen es " + str(tipos[0].get("fraud_type")) + "."
    if metricas:
        texto += (
            f" La regresión logística balanceada, con umbral {modelo.get('umbral')}, obtuvo "
            f"en la clase fraude precisión {metricas.get('precision')}, recall {metricas.get('recall')}, "
            f"F1 {metricas.get('f1')} y ROC AUC {metricas.get('roc_auc')} "
            f"sobre {_miles(modelo.get('n_test'))} transacciones de prueba."
        )
    return texto + " Esta salida usa los cálculos de Python y no sustituye una revisión del caso."
