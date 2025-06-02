
import streamlit as st
from config import style
from routes import routes

# Style adaptatif pour le mode clair/sombre
st.markdown(style, unsafe_allow_html=True)

# En-tête principal avec image
st.markdown("""
    <div class="main-header">
        <h1>Application d'Analyse et de Prédiction de Données</h1>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            Bienvenue sur notre pipeline d'analyse et prédiction
        </p>
    </div>
    """, unsafe_allow_html=True)

st.info(
    "Utilisez le menu de gauche pour naviguer dans l'application :\n\n"
    "- Import\n"
    "- Exploration\n"
    "- Traitement\n"
    "- Entraînement\n"
    "- Evaluation\n"
    "- Prédiction"
)
