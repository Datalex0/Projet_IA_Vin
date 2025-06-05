from streamlit import Page
import streamlit as st

routes = [
    Page("pages/0_Accueil.py", title="Accueil", icon="🏠"),
    Page("pages/1_Import.py", title="Import des données", icon="📥"),
    Page("pages/2_Exploration.py", title="Exploration et Traitements", icon="🔍"),
    Page("pages/3_Machine Learning.py", title="Entraînement d'un modèle", icon="🦾"),
    Page("pages/4_Evaluation.py", title="Évaluations", icon="📝"),
    Page("pages/5_Predictions.py", title="Prédictions", icon="🔮")
]

def redirection(titre, nom_de_page):
    # Redirection page suivante
    st.write("***")
    st.markdown("Vous pouvez maintenant passer à la page")
    if st.button(titre):
        st.switch_page(page=f"pages/{nom_de_page}.py")