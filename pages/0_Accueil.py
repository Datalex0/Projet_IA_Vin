
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

with st.expander("ℹ️ Fonctionnement"):
    st.info("""
        Tout au long de votre parcours, des onglets rétractables comme celui-ci vous aideront dans l'utilisation de l'application.\n
        Vous utiliserez le menu de gauche pour naviguer dans les pages de l'application et certaines pages comportent également des onglets.
    """)
    
st.markdown("""
            ## Dans cette application, voici ce que vous allez pouvoir faire dans les différentes pages :\n
            - ### 📥 Import des données : 
                * Importer vos données ou utiliser le dataset déjà enregistré et supprimer les colonnes non désirées.      
            - ### 🔍 Exploration et Traitements : 
                * Choisir la colonne cible et si vous voulez faire de la Classification ou de la Régression,
                * Observer la distribution des variables,
                * Encoder la cible si besoin puis observer les corrélations et choisir les colonnes à conserver en fonction,
                * Effectuer la gestion des valeurs manquantes et des valeurs aberrantes,
                * Standardiser les données si nécessaire,
                * Exporter le résultat en CSV ou XLSX et générer un rapport PDF des observations et traitements effectués.                
            - ### 🦾 Entraînement d'un modèle : 
                * Effectuer la séparation du jeu de données (entraînement/test) puis sélectionner le meilleur modèle pour votre modélisation,
                * Entraîner le modèle sélectionné et l'exporter au format pickles,
                * Optimiser automatiquement les Hyperparamètres puis exporter le modèle optimisé au format pickles.                
            - ### 📝 Évaluations : 
                * Évaluer les performances du modèle
            - ### 🔮 Prédictions : 
                * Effectuer des Prédictions sur de nouvelles données
            """)


    
    