# 📊 Dashboard de Validación: Riesgo Diabetes T2

Este repositorio contiene el código fuente de la herramienta de análisis estadístico y visualización para el protocolo de investigación **"Diseño de un formulario basado en probabilidad y estadística para evaluar el riesgo de padecimiento de diabetes tipo 2"**.

El sistema se conecta a una base de datos en la nube (Firebase), procesa las encuestas en tiempo real y genera métricas de validación clínica (Sensibilidad, Especificidad, Curva ROC) para evaluar el desempeño del modelo predictivo.

## 🚀 Funcionalidades Principales

1.  **Conexión Segura:** Extracción de datos cifrada desde Google Firebase Firestore.
2.  **Procesamiento de Datos (ETL):** Limpieza y transformación de datos crudos utilizando `pandas`.
3.  **Cálculo de Riesgo:** Implementación del algoritmo de regresión logística ponderada (Score propio).
4.  **Validación Inferencial:**
    * Generación automática de **Matriz de Confusión**.
    * Cálculo de **Curva ROC** y **AUC**.
    * Correlación visual entre Puntaje, Glucosa e IMC.
5.  **Interfaz Interactiva:** Dashboard web construido con `Streamlit` y `Plotly`.

## 🛠️ Requisitos del Sistema

* **Python 3.8** o superior.
* Archivo de credenciales `serviceAccountKey.json` (No incluido en el repositorio por seguridad).
* Conexión a internet para acceder a Firebase.

## 📦 Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto en tu entorno local:

### 1. Clonar o descargar el repositorio
Descarga los archivos del proyecto en tu carpeta de trabajo.

### 2. Configurar el entorno virtual (Recomendado)
Para evitar conflictos de dependencias, crea y activa un entorno virtual:

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
