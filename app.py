import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Validación Riesgo Diabetes", layout="wide", page_icon="🩺")

# --- 1. CONEXIÓN A FIREBASE ---
@st.cache_resource
def init_connection():
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json") 
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_connection()

# --- 2. OBTENCIÓN DE DATOS ---
@st.cache_data(ttl=600)
def get_data():
    docs = db.collection('encuestas').stream()
    items = [doc.to_dict() for doc in docs]
    return items

# --- 3. INTERFAZ GRÁFICA ---
st.title("📊 Tablero de Validación: Modelo de Riesgo DM2")
st.markdown(" Análisis en tiempo real de la prueba piloto en CDMX.")

try:
    raw_data = get_data()
    if not raw_data:
        st.error("No hay datos en Firebase.")
        st.stop()
    
    df = pd.DataFrame(raw_data)

    # --- PROCESAMIENTO Y TRADUCCIÓN DE VARIABLES ---
    # 1. Convertir numéricos antes de renombrar
    cols_num_map = {
        'bmi': 'IMC',
        'glucose_level': 'Glucosa',
        'score': 'Puntaje',
        'waist_size': 'Cintura'
    }
    
    # Convertimos a número las columnas originales si existen
    for eng_col in cols_num_map.keys():
        if eng_col in df.columns:
            df[eng_col] = pd.to_numeric(df[eng_col], errors='coerce')

    # 2. RENOMBRAR COLUMNAS AL ESPAÑOL (Esto cambia todo en las gráficas)
    df.rename(columns={
        'bmi': 'IMC',
        'risk_level': 'Nivel de Riesgo',
        'glucose_level': 'Glucosa',
        'score': 'Puntaje',
        'sex': 'Sexo',
        'age_range': 'Rango de Edad',
        'acanthosis_nigricans': 'Acantosis',
        'waist_size': 'Cintura'
    }, inplace=True)

    # ---------------------------------------------------------
    # 🔥 DEFINICIÓN DE LA VERDAD Y PREDICCIÓN
    # ---------------------------------------------------------
    if 'Glucosa' in df.columns and 'Puntaje' in df.columns:
        # Glucosa >= 100 es Riesgo Real
        df['Realidad'] = (df['Glucosa'] >= 100).astype(int)
        # Score >= 10 es Predicción de Riesgo
        df['Prediccion'] = (df['Puntaje'] >= 10).astype(int)

    # --- TABS EN ESPAÑOL ---
    tab1, tab2, tab3 = st.tabs(["📈 Descriptiva (Sec 9.1)", "⚠️ Factores de Riesgo (Sec 9.4)", "🧪 Validación Inferencial (Sec 9.5)"])

    # === TAB 1: DESCRIPTIVA ===
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Participantes", len(df))
        if 'Sexo' in df.columns:
            c2.metric("Hombres", len(df[df['Sexo'] == 'Hombre']))
            c3.metric("Mujeres", len(df[df['Sexo'] == 'Mujer']))
        if 'Rango de Edad' in df.columns:
            moda_edad = df['Rango de Edad'].mode()[0]
            c4.metric("Edad Frecuente", moda_edad)

        st.divider()
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribución por Género")
            if 'Sexo' in df.columns:
                fig_sex = px.pie(df, names='Sexo', title='Participantes por Sexo', hole=0.4)
                st.plotly_chart(fig_sex, use_container_width=True)
        
        with col_b:
            if 'Nivel de Riesgo' in df.columns:
                st.subheader("Niveles de Riesgo Detectados")
                # Ordenamos para que salga Bajo -> Medio -> Alto
                orden_riesgo = ['Bajo', 'Medio', 'Alto']
                fig_risk = px.bar(df['Nivel de Riesgo'].value_counts().reindex(orden_riesgo).reset_index(), 
                                  x='Nivel de Riesgo', y='count', 
                                  color='Nivel de Riesgo',
                                  title='Conteo por Nivel de Riesgo',
                                  labels={'count': 'Cantidad de Personas'},
                                  color_discrete_map={'Alto':'red', 'Medio':'orange', 'Bajo':'green'})
                st.plotly_chart(fig_risk, use_container_width=True)

    # === TAB 2: FACTORES DE RIESGO ===
    with tab2:
        st.info("Análisis de variables clave: IMC y Acantosis Nigricans.")
        
        c1, c2 = st.columns(2)
        with c1:
            # Boxplot IMC
            if 'IMC' in df.columns and 'Nivel de Riesgo' in df.columns:
                st.subheader("IMC vs Nivel de Riesgo")
                fig_box = px.box(df, x='Nivel de Riesgo', y='IMC', color='Nivel de Riesgo',
                                 title="Distribución de IMC según Riesgo",
                                 color_discrete_map={'Alto':'red', 'Medio':'orange', 'Bajo':'green'},
                                 points="all") 
                st.plotly_chart(fig_box, use_container_width=True)
        
        with c2:
            # Acantosis vs Glucosa
            if 'Acantosis' in df.columns and 'Glucosa' in df.columns:
                st.subheader("Impacto de Acantosis Nigricans")
                fig_acant = px.strip(df, x='Acantosis', y='Glucosa', 
                                     color='Acantosis',
                                     title="Glucosa según presencia de Acantosis")
                st.plotly_chart(fig_acant, use_container_width=True)

    # === TAB 3: VALIDACIÓN INFERENCIAL ===
    with tab3:
        st.markdown("### 🧬 Evaluación del Desempeño del Modelo")

        col_izq, col_der = st.columns([2, 1])

        with col_izq:
            # Scatter: Puntaje vs Glucosa
            if 'Puntaje' in df.columns and 'Glucosa' in df.columns:
                st.subheader("Correlación: Puntaje vs. Glucosa")
                fig_scatter = px.scatter(df, x='Puntaje', y='Glucosa',
                                         color='Nivel de Riesgo', size='IMC',
                                         hover_data=['Sexo', 'Rango de Edad'],
                                         title="Puntaje del Modelo vs. Glucosa Real (Tamaño = IMC)",
                                         color_discrete_map={'Alto':'red', 'Medio':'orange', 'Bajo':'green'})
                
                fig_scatter.add_hline(y=100, line_dash="dot", annotation_text="Glucosa 100 mg/dL")
                fig_scatter.add_vline(x=10, line_dash="dot", annotation_text="Corte Riesgo (10 pts)")
                st.plotly_chart(fig_scatter, use_container_width=True)

        with col_der:
            # Matriz de Confusión
            st.subheader("Matriz de Confusión")
            tn, fp, fn, tp = confusion_matrix(df['Realidad'], df['Prediccion']).ravel()
            
            st.write(f"✅ **Verdaderos Positivos:** {tp}")
            st.write(f"❌ **Falsos Negativos:** {fn}")
            st.write(f"🛡️ **Verdaderos Negativos:** {tn}")
            st.write(f"⚠️ **Falsos Positivos:** {fp}")
            
            sensibilidad = tp / (tp + fn) if (tp+fn)>0 else 0
            especificidad = tn / (tn + fp) if (tn+fp)>0 else 0
            
            st.metric("Sensibilidad", f"{sensibilidad:.1%}")
            st.metric("Especificidad", f"{especificidad:.1%}")

        st.divider()

        # Curva ROC
        st.subheader("Curva ROC")
        fpr, tpr, thresholds = roc_curve(df['Realidad'], df['Puntaje'])
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f'Curva ROC (AUC = {roc_auc:.3f})',
            labels=dict(x='1 - Especificidad (Falsos Positivos)', y='Sensibilidad (Verdaderos Positivos)'),
            width=700, height=500
        )
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        
        c_roc1, c_roc2 = st.columns([3, 1])
        with c_roc1:
            st.plotly_chart(fig_roc, use_container_width=True)
        with c_roc2:
            st.success(f"**AUC: {roc_auc:.3f}**")
            if roc_auc >= 0.75:
                st.balloons()
                st.markdown("🎉 **¡Hipótesis Aceptada!**")
            else:
                st.warning("Requiere calibración.")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Verifica que tu 'serviceAccountKey.json' esté en la carpeta.")