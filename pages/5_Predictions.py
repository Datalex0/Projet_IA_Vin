import streamlit as st
from config import style
import pandas as pd
from modules.module_predict import appliquer_modele, charger_modele_externe, saisir_donnees_manuellement, import_fichier

# Style adaptatif pour le mode clair/sombre
st.markdown(style, unsafe_allow_html=True)

# En-tête principal
st.markdown("""
    <div class="main-header">
        <h1>🔮 Prédictions</h1>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            Importez ou saisissez vos nouvelles données pour obtenir des prédictions à l'aide du modèle entraîné
        </p>
    </div>
    """, unsafe_allow_html=True)



# Chargement du modèle
st.subheader("📦 Modèle à utiliser")
choix_modele = st.radio(
    "Souhaitez-vous utiliser le modèle déjà entraîné ou en importer un autre ?", 
    ["Modèle en mémoire", "Importer un fichier .pkl"]
)

modele = None

if choix_modele == "Modèle en mémoire":
    modele = st.session_state.get("modele_final")
    if modele:
        st.success("✅ Modèle récupéré depuis la session")
    else:
        st.warning("❌ Aucun modèle trouvé dans la session")

elif choix_modele == "Importer un fichier .pkl":
    fichier_pkl = st.file_uploader("Importer un modèle .pkl", type=["pkl"])
    if fichier_pkl:
        modele = charger_modele_externe(fichier_pkl)
        st.session_state["modele_final"] = modele
        st.success("✅ Modèle importé avec succès")

if not modele:
    st.warning("❌ Aucun modèle disponible. Entraînez ou importez un modèle pour continuer.")
    st.stop()
    

# Choix de la méthode d'entrée
st.subheader("🧾 Méthode d'entrée des données")
methode = st.radio(
    "Comment souhaitez-vous fournir les données à prédire ?", 
    ["Fichier CSV ou XLSX", "Saisie manuelle"]
)

# CSV ou XLSX
if methode == "Fichier CSV ou XLSX":
    df_new = import_fichier()
    # uploaded_file = st.file_uploader("Uploader un fichier CSV avec les mêmes colonnes que le modèle (hors cible)", type="csv")

    # if uploaded_file is not None:
    #     df_new = pd.read_csv(uploaded_file)
        # st.write("Aperçu des données importées :")
        # st.dataframe(df_new.head())

    if st.button("⚡ Appliquer le modèle pour prédire", key="predict_csv"):
        df_new = st.session_state.get("df_new", None)
        if df_new is not None:
            try:
                df_resultats = appliquer_modele(modele, df_new)
                st.success("✅ Prédictions effectuées avec succès")
                st.dataframe(df_resultats.head())

                csv = df_resultats.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Télécharger les prédictions", data=csv, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'application du modèle : {e}")
    else:
        st.warning("Veuillez d'abord importer un fichier avant de lancer la prédiction.")

# Saisie manuelle
elif methode == "Saisie manuelle":
    if "X_train" in st.session_state:
        X_train = st.session_state["X_train"]
        colonnes = X_train.columns.tolist()
                
        df_input = saisir_donnees_manuellement(colonnes)

        if st.button("⚡ Prédire", key="predict_form"):
            try:
                df_resultats = appliquer_modele(modele, df_input)
                st.success("✅ Prédiction effectuée :")
                st.dataframe(df_resultats)
            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction : {e}")
    else:
        st.warning("❗ Pour la saisie manuelle, les colonnes attendues doivent être disponibles dans X_train.")
