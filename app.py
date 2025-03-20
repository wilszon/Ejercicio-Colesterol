import streamlit as st
import joblib
import pandas as pd

# --- Configuración de la página ---
st.set_page_config(page_title="Predicción de Problemas Cardiacos", layout="centered")

# --- TÍTULO Y SUBTÍTULO ---
st.title("🩺 Modelo de Predicción de Problemas Cardíacos con IA")
st.subheader("Realizado por Wilson Suarez")

# --- INTRODUCCIÓN ---
st.write("""
Esta aplicación permite predecir si una persona tiene riesgos de problemas cardíacos.  
Ingresa tu edad y nivel de colesterol para obtener una predicción basada en inteligencia artificial.
""")

# --- IMAGEN ---
image_url = "https://images.ecestaticos.com/aTyeFebpQ-BqHJ7FIQjnjzcN2og=/334x4:1953x1213/1200x900/filters:fill(white):format(jpg)/f.elconfidencial.com%2Foriginal%2Fb9e%2Fd37%2F516%2Fb9ed3751689578efdbb19ed1b8b401e9.jpg"
st.image(image_url, caption="Imagen de referencia sobre problemas cardíacos", use_container_width=True)

# --- ENTRADA DE DATOS ---
st.sidebar.header("📊 Ingrese los datos")

edad = st.sidebar.slider("Edad", min_value=20, max_value=80, value=40, step=1)
colesterol = st.sidebar.slider("Colesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1)

# Crear DataFrame con los datos ingresados
data = pd.DataFrame({"edad": [edad], "colesterol": [colesterol]})
st.write("### Datos ingresados:")
st.write(data)

# --- CARGAR MODELOS ---
scaler = joblib.load("scaler.bin")
knn_model = joblib.load("knn_model.bin")

# --- NORMALIZACIÓN ---
data_scaled = scaler.transform(data)

# --- PREDICCIÓN ---
prediccion = knn_model.predict(data_scaled)[0]

# --- MOSTRAR RESULTADO ---
st.write("### Resultado de la Predicción:")

if prediccion == 0:
    st.success("✅ No tiene problemas cardíacos", icon="😃")
    st.markdown('<div style="background-color:#ADD8E6; padding:10px; border-radius:5px;">'
                '<h3 style="color:#000000;">Resultado: No tiene problemas cardíacos</h3></div>',
                unsafe_allow_html=True)
else:
    st.error("⚠️ Tiene riesgo de problemas cardíacos", icon="⚠️")
    st.markdown('<div style="background-color:#FF7F7F; padding:10px; border-radius:5px;">'
                '<h3 style="color:#000000;">Resultado: Tiene riesgo de problemas cardíacos</h3></div>',
                unsafe_allow_html=True)

# --- LÍNEA DIVISORIA ---
st.markdown("---")

# --- COPYRIGHT ---
st.markdown("© Unab2025")
