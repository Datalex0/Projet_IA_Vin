import streamlit as st
import pandas as pd
from modules.module_import import state_write, import_fichier
from config import style

# Style adaptatif pour le mode clair/sombre
st.markdown(style, unsafe_allow_html=True)

# En-tête principal
st.markdown("""
    <div class="main-header">
        <h1>📥 Import des données</h1>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            Importez votre fichier Excel ou CSV
        </p>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            ou cochez directement la case sur votre gauche pour utiliser le dataframe préenregistré sur le vin
        </p>
    </div>
    """, unsafe_allow_html=True)


use_df_vin =st.sidebar.checkbox("Utiliser le dataframe sur le vin")

# Si l'utilisateur veut utiliser le df sur le vin
if use_df_vin :
    df = pd.read_csv("vin.csv")
    # Enregistrement dans session_state
    st.success(f'✅ Données issues de "vin.csv" importées avec succès !')
    state_write(df)

# Sinon il charge son csv ou xlsx
else :
    import_fichier()

# Empêche l'accès aux autres pages si le fichier n'est pas encore chargé
if "df" not in st.session_state:
    st.warning("Veuillez importer un fichier avant de continuer.")
    st.stop()
else:
    df = st.session_state["df"]
    try:
        st.markdown(f"### Affichage des 10 premières lignes du Dataframe :")
        st.write(df.head(10))
        # Métriques du dataset dans un container
        with st.container():
            st.markdown("### 📊 Statistiques du Dataset")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Lignes",
                    f"{len(df):,}",
                    help="Nombre total de lignes dans le dataset"
                )

            with col2:
                st.metric(
                    "Colonnes",
                    len(df.columns),
                    help="Nombre total de colonnes dans le dataset"
                )
    except:
        st.stop()