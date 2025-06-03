from streamlit import Page

routes = [
    Page("pages/0_Accueil.py", title="Accueil", icon="🏠"),
    Page("pages/1_Import.py", title="Import des données", icon="📥"),
    Page("pages/2_Exploration.py", title="Exploration et Traitements", icon="🔍"),
    Page("pages/4_Machine Learning.py", title="Entraînement d'un modèle", icon="🦾"),
    Page("pages/5_Evaluation.py", title="Évaluation", icon="📝"),
]
