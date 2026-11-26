import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import gzip

# 1. Configuración básica de la página
st.set_page_config(page_title="Detector de Phishing", page_icon="🔒")
st.title("🔒 Detector de Phishing en URLs")
st.write("Pega una URL y el modelo te dirá si es legítima o sospechosa de phishing.")

# 2. Cargar el modelo, el escalador y las columnas
# Usamos @st.cache_resource para que la app no recargue los modelos cada vez que el usuario hace clic
@st.cache_resource
def cargar_modelos():
    with gzip.open('modelo_phishing.pkl.gz', 'rb') as f:
        modelo = joblib.load(f)
    scaler = joblib.load('scaler_phishing.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return modelo, scaler, feature_cols

modelo, scaler, feature_cols = cargar_modelos()

# 3. La misma función de extracción de features que usaste en Colab
def extraer_features(url):
    url = str(url)
    return {
        'longitud': len(url),
        'num_puntos': url.count('.'),
        'num_guiones': url.count('-'),
        'num_barras': url.count('/'),
        'num_digitos': sum(c.isdigit() for c in url),
        'num_at': url.count('@'),
        'num_igual': url.count('='),
        'num_query': url.count('?'),
        'tiene_https': int('https' in url.lower()),
        'tiene_ip': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
        'tiene_palabra_sospechosa': int(any(p in url.lower() for p in
            ['login', 'verify', 'secure', 'bank', 'account', 'update', 'confirm', 'signin'])),
    }

# 4. Interfaz de usuario
url_input = st.text_input("URL a analizar:", placeholder="ej: https://www.google.com")

if st.button("Analizar"):
    if not url_input.strip():
        st.warning("Por favor ingresa una URL válida.")
    else:
        # Extraer features
        feats = extraer_features(url_input)
        
        # Crear el array de X en el mismo orden que el entrenamiento
        X = np.array([[feats[c] for c in feature_cols]])
        
        # Escalar las features con el mismo escalador
        X_scaled = scaler.transform(X)
        
        # Hacer la predicción
        prediccion = modelo.predict(X_scaled)[0]
        probabilidades = modelo.predict_proba(X_scaled)[0]
        
        # 5. Mostrar el resultado
        if prediccion == 1:
            st.error(f"⚠️ **POSIBLE PHISHING** (Confianza: {probabilidades[1]*100:.2f}%)")
        else:
            st.success(f"✅ **URL LEGÍTIMA** (Confianza: {probabilidades[0]*100:.2f}%)")
            
        # 6. Mostrar las features extraídas (para transparencia)
        st.subheader("Análisis de características de la URL")
        df_feats = pd.DataFrame([feats])
        st.dataframe(df_feats)
