# Proyecto final · Fraude en transferencias electrónicas (TEF)

Aplicación modular en Streamlit para explorar un dataset sintético de Transferencias Electrónicas de Fondos en Chile, entrenar una línea base de clasificación y consultar un agente local con Ollama.

El trabajo pertenece al módulo **Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos**. El análisis es académico: no es un sistema de prevención de fraude ni una herramienta de decisión bancaria.

## Propósito

Describir la calidad de los datos, comparar transacciones legítimas y fraudulentas, visualizar segmentos relevantes y pedir a un modelo local que interprete cifras ya calculadas por Python.

## Resultados calculados sobre `data/TEF_chile.csv`

Estas cifras salen del código del proyecto y sirven para completar el informe.

| Indicador | Resultado |
|---|---|
| Transacciones | 94.593 |
| Columnas del CSV | 36 |
| Periodo | 31 de diciembre de 2023 a 29 de junio de 2024 |
| Moneda | CLP |
| Duplicados de fila | 0 |
| Fraudes (`is_fraud = 1`) | 2.081 (2,20 %) |
| Única columna con nulos | `avg_amount_last_7d`: 65.724 nulos (69,48 %) |
| Mediana del monto legítimo | 15.582,5 CLP |
| Mediana del monto fraudulento | 86.879 CLP |
| Tipo de fraude más frecuente | Vishing/Phishing: 509 casos (24,46 % de los fraudes) |
| Banco con mayor tasa | BancoEstado: 3,13 % (1.373 fraudes en 43.801 transacciones) |
| Alerta de comunicación | 854 de 854 transacciones con alerta están etiquetadas como fraude |
| Cambio reciente de cuenta | tasa de 13,23 % con cambio y 0,91 % sin cambio |

Línea base: regresión logística con `class_weight='balanced'`, umbral 0,5, 75 % de entrenamiento y 25 % de prueba estratificada (`random_state=42`).

| Métrica en prueba (23.649 transacciones, 520 fraudes) | Valor |
|---|---|
| Exactitud | 0,829 |
| Precisión de la clase fraude | 0,084 |
| Recall de la clase fraude | 0,681 |
| F1 | 0,149 |
| ROC AUC | 0,826 |

Matriz de confusión en prueba: 19.250 legítimas correctas, 3.879 legítimas marcadas como fraude, 166 fraudes no detectados y 354 fraudes recuperados.

El modelo no usa `fraud_type`, la alerta de comunicación ni el cambio de cuenta, porque en este dataset describen el escenario etiquetado. Tampoco usa los deltas de saldo: `sender_balance_delta` reproduce el monto en casi todas las filas.

## Estructura

```text
proyecto_final/
├── app.py
├── README.md
├── requirements.txt
├── run_app.bat
├── .gitignore
├── data/
│   ├── TEF_chile.csv
│   └── diccionario_variables_tef.md
├── assets/
└── src/
    ├── data_loader.py      # carga, perfil y campos derivados
    ├── eda.py              # nulos, segmentos, correlaciones y modelo
    ├── visualizations.py   # gráficos Plotly
    ├── llm_client.py       # conexión HTTP con Ollama
    └── agent.py            # clase AgenteDatos y resumen de respaldo
```

`app.py` solo arma la interfaz. Las funciones de análisis viven en `src/`.

## Requisitos

- Python 3.11 o superior. El desarrollo se probó con Python 3.13.
- Conexión a internet solo para instalar dependencias la primera vez.
- Ollama es opcional para los gráficos y las tablas. Es necesario para la respuesta generativa del agente.

## Entorno virtual

### Windows

```powershell
cd proyecto_final
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
cd proyecto_final
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Ollama

Instalar Ollama desde su sitio oficial y, con el servicio activo, descargar un modelo:

```bash
ollama pull llama3.2:3b
```

En un equipo con poca memoria:

```bash
ollama pull llama3.2:1b
```

Si se usa la versión de 1B, hay que escribir ese nombre en el campo **Modelo Ollama** de la barra lateral.

## Ejecución

```bash
python -m streamlit run app.py
```

En Windows también se puede usar `run_app.bat`. El script crea el entorno, instala `requirements.txt` y abre la aplicación.

El dataset ya está en `data/TEF_chile.csv`. La interfaz permite cargar otro CSV si incluye la columna `is_fraud`. El diccionario de variables está en `data/diccionario_variables_tef.md`.

## Uso de la interfaz

1. Revisar en **Datos** la muestra, la calidad y el resumen numérico.
2. En **Exploración**, cambiar el segmento (tipo de transacción, banco, alerta, cambio de cuenta, hora o tipo de fraude) y revisar monto, distribuciones y correlaciones.
3. En **Modelo**, leer precisión, recall, F1, ROC AUC, la matriz de confusión y los coeficientes.
4. En **Agente**, elegir una pregunta sugerida o escribir otra. Con Ollama activo, la clase `AgenteDatos` elige herramientas de calidad, segmentos o modelo y envía ese contexto al modelo local. Si Ollama no responde, aparece un resumen armado con las mismas cifras.

Los histogramas de variables continuas usan una muestra reproducible de hasta 15.000 filas. Las tasas, la calidad y el modelo usan el dataset completo.

## Paradigmas que evidencia el código

- Imperativo: validaciones de carga y la decisión de usar Ollama o el resumen de respaldo.
- Funcional: funciones reutilizables de perfil, tasas, correlaciones, métricas y gráficos.
- Orientado a objetos: `AgenteDatos` y `OllamaClient`.

## Problemas conocidos

- `avg_amount_last_7d` tiene 69,48 % de nulos, por eso no entra a la matriz de correlación ni al modelo.
- La alerta de comunicación separa fraude y no fraude de forma perfecta en este archivo sintético. Usarla como predictor repetiría la etiqueta.
- La precisión de 0,084 muestra muchas alertas sobre transacciones legítimas. El recall de 0,681 recupera cerca de dos tercios de los fraudes de la prueba.
- Un Ollama apagado, o un modelo no descargado, deja el agente en modo de respaldo. El resto de la aplicación sigue funcionando.
- El CSV pesa unos 43 MB. GitHub acepta ese tamaño. No hay que subir el modelo descargado por Ollama ni la carpeta `.venv`.

## Fuente de los datos

El archivo fue incorporado al proyecto junto con su diccionario, que lo describe como dataset sintético de TEF. En el informe hay que citar la URL pública de origen (Kaggle u otra fuente) además de este repositorio.
