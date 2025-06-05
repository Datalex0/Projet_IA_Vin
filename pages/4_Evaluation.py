import streamlit as st
from config import style
import pandas as pd
from modules.module_eval import afficher_evaluation, fonctionnement
from routes import redirection

# Style adaptatif pour le mode clair/sombre
st.markdown(style, unsafe_allow_html=True)

# En-tête principal
st.markdown("""
    <div class="main-header">
        <h1>📝 Évaluation du modèle</h1>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            Analyse des performances du modèle entraîné et/ou optimisé
        </p>
    </div>
    """, unsafe_allow_html=True)

    
# Vérification des éléments nécessaires
if not all(k in st.session_state for k in ["modele_final", "X_test", "y_test", "task"]):
    st.warning("❌ Veuillez d'abord entraîner un modèle (et splitter les données) dans l'onglet 3.")
    st.stop()
else:
    # Récupération des objets depuis la session
    modele = st.session_state["modele_final"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    task_type = st.session_state["task"]  # "classification" ou "régression"

    # Expander fonctionnement
    fonctionnement(task_type)
    
    # Évaluation
    afficher_evaluation(modele, X_test, y_test, task_type)
    
    # Redirection page suivante
    redirection("🔮 Prédictions", "5_Predictions")