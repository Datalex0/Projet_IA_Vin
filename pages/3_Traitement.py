import streamlit as st
from config import style

# Style adaptatif pour le mode clair/sombre
st.markdown(style, unsafe_allow_html=True)

# En-tête principal
st.markdown("""
    <div class="main-header">
        <h1>🧹 Nettoyage et Traitement des données</h1>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            Nettoyage et Traitement des données
        </p>
    </div>
    """, unsafe_allow_html=True)

if "df" not in st.session_state:
    st.warning("❌ Veuiller importer des données avant de pouvoir effectuer le nettoyage.")
    st.stop()
else:
    df = st.session_state["df"]