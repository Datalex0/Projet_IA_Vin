import streamlit as st
from config import style
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.module_explo import select_distribution,select_pairplot, generer_rapport_pdf, telecharger_donnees, detecter_outliers_zscore, detecter_outliers_iqr, traiter_outliers, detection_outliers 
from sklearn.impute import SimpleImputer
import os

# Style adaptatif pour le mode clair/sombre
st.markdown(style, unsafe_allow_html=True)

# En-tête principal
st.markdown("""
    <div class="main-header">
        <h1>🔍 Exploration des données</h1>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            Exploration des données
        </p>
    </div>
    """, unsafe_allow_html=True)

# Récupération du DF
if "df" not in st.session_state:
    st.warning("❌ Veuiller importer des données pour pouvoir explorer.")
    st.stop()
else:
    df = st.session_state["df"]


# Onglets de navigation
onglet1, onglet2, onglet3, onglet4, onglet5 = st.tabs([
    "🧬 Types & Cible",
    "📊 Distributions",
    "📈 Corrélations",
    "🧹 NaN & Outliers",
    "🎯 Préparation finale"
])


### Onglet 1 - Types & Cible
with onglet1:
    with st.expander("ℹ️ Fonctionnement"):
        st.markdown("""
        Cette section permet d'identifier les types de chaque variable (numérique, catégorielle...) et de choisir la variable cible pour la modélisation.
        En fonction de la cible choisie, l'application propose automatiquement un type de tâche : **classification** (si 10 modalités ou moins) ou **régression** (valeurs continues).
        Vous pouvez forcer ce choix manuellement.
        """)
    
    # Affichage du type de données pour chaque colonne
    st.subheader("🧬 Types de données")
    st.dataframe(df.dtypes.reset_index().rename(columns={0: "Type", "index": "Colonne"}))

    # L'utilisateur choisit sa colonne cible
    st.subheader("🎯 Sélection de la variable cible")
    target = st.selectbox("Choisissez la cible", df.columns)

    # Si la cible comporte jusqu'à 10 valeurs différente, on estime que c'est de la Classification sinon Régression
    st.subheader("🧠 Type de tâche")
    # Autodétection
    auto_detected_task = "classification" if df[target].nunique() <= 10 else "régression"
    st.write(f"Suggestion automatique : **{auto_detected_task}**")

    # Choix par l'utilisateur avec le choix autodétecté par défaut
    task = st.radio("Choisissez le type de tâche", ["classification", "régression"], index=0 if auto_detected_task == "classification" else 1)
    st.success(f"Type de tâche sélectionné : **{task}**")


### Onglet 2 - Distributions
with onglet2:
    st.subheader("📊 Distribution des variables")
    
    with st.expander("ℹ️ Fonctionnement"):
        st.markdown("""
        - Pour les colonnes **numériques**, un histogramme avec une courbe de densité (KDE) montre la forme de la distribution (normale, asymétrique, etc).
        - Pour les colonnes **catégorielles**, un diagramme à barres indique la fréquence de chaque modalité.

        Cela aide à détecter les valeurs aberrantes, les déséquilibres ou les transformations à appliquer avant le traitement.
        """)
    
    # Sélection de la colonne à analyser
    selected_col = st.selectbox("Sélectionnez une colonne", df.columns)

    # Si la colonne est de type numérique : histplot + kde
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
    else:
        # Sinon, barplot
        counts = df[selected_col].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(v), ha='center', va='bottom')
        plt.xticks(rotation=45)
        st.pyplot(fig)

### Onglet 3 - Corrélations
with onglet3:
    with st.expander("ℹ️ Fonctionnement"):
        st.markdown("""
        Cette section permet de visualiser les relations entre les variables numériques grâce à une matrice de corrélation.
        Elle aide à détecter les variables redondantes ou très corrélées, qu'il peut être utile d'exclure pour éviter les multicolinéarités.
        Elle permet également de sélectionner les colonnes à conserver pour la modélisation, en se basant sur cette matrice.
        """)
        
    st.subheader("📈 Matrice de corrélation")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📉 Colonnes trop corrélées (> 0.95)")
    to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.95:
                to_drop.add(corr.columns[i])
    if len(to_drop)>0 :
        st.write(list(to_drop))
    else:
        st.write("Pas de colonne trop corrélée")
        
    st.subheader("✅ Sélection guidée des colonnes")
    if task == "régression":
        good_corr_cols = corr[target][abs(corr[target]) > 0.2].drop(target).index.tolist()
    else:
        good_corr_cols = numeric_cols
    selected_cols_corr = st.multiselect("Colonnes à conserver pour la modélisation", numeric_cols, default=good_corr_cols)
    

# Onglet 4 - NaN & Outliers
with onglet4:
    with st.expander("ℹ️ Fonctionnement"):
        st.markdown("""
        Cette section permet :
        - d'analyser les valeurs manquantes (NaN) et de choisir une méthode d'imputation (moyenne, médiane, etc.)
        - de détecter les valeurs aberrantes (outliers) grâce au score Z, afin de les supprimer ou ajuster au besoin.
        """)
        
    # Gestion des NaN
    st.subheader("🧹 Gestion des NaN")
    nan_summary = df.isna().sum()
    st.dataframe(nan_summary[nan_summary > 0])

    selected_nan = st.multiselect("Colonnes à traiter", nan_summary[nan_summary > 0].index.tolist())
    imputation_label = st.selectbox("Méthode d'imputation", ["La Moyenne", "La Médiane", "La valeur la plus fréquente"])
    translate = {
        "La Moyenne":"mean",
        "La Médiane":"median",
        "La valeur la plus fréquente":"most_frequent"
    }
    imputation_strategy = translate[imputation_label]

    if st.button("Remplacer les NaN"):
        imputer = SimpleImputer(strategy=imputation_strategy)
        df[selected_nan] = imputer.fit_transform(df[selected_nan])
        st.success("NaN remplacés avec succès")

    st.write("***")

    # Gestion des  Outliers
    st.subheader("🚨 Détection des outliers")

    methode_outlier = st.radio("Méthode de détection", ["Z-score", "IQR"])

    outliers_bool = detection_outliers(df, numeric_cols, methode_outlier, key="zscore1")
    
    st.write("Nombre d'outliers par colonne :")
    st.write(outliers_bool.sum())

    traitement_outlier = st.selectbox("Action", ["Aucune", "Supprimer les lignes", "Remplacer par la médiane"])
    
    if traitement_outlier != "Aucune":
        df = traiter_outliers(df, numeric_cols, outliers_bool, traitement_outlier)
        st.success("Traitement des outliers effectué.")
        outliers_bool_clean = detection_outliers(df, numeric_cols, methode_outlier, key="zscore2")
        st.write(outliers_bool_clean.sum())


# Onglet 5 - Préparation finale
with onglet5:
    st.subheader("🎯 Sélection des colonnes finales")
    to_keep = st.multiselect("Colonnes à conserver", df.columns.tolist(), default=[c for c in df.columns if c not in to_drop])

    if st.button("✅ Enregistrer le jeu de données nettoyé"):
        df_clean = df[to_keep].copy()
        st.session_state["df_clean"] = df_clean
        st.success("Jeu de données nettoyé enregistré avec succès dans session_state['df_clean']")
        st.write(df_clean.head())

    st.write("***")

    # Téléchargements seulement si df_clean est défini
    if "df_clean" in st.session_state:
        df_clean = st.session_state["df_clean"]
        
        # Téléchargement des données au format CSV ou XLSX
        telecharger_donnees(df_clean)
        
        st.write("***")
        
        # Téléchargement d'un Rapport d'explo en PDF
        st.subheader("📝 Générer un rapport de l'exploration et des traitements en PDF")

        if st.button("📄 Générer le rapport PDF"):
            pdf_path, images = generer_rapport_pdf(df, df_clean, target, to_keep, task, to_drop, corr)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Télécharger le rapport PDF",
                    data=f.read(),
                    file_name="rapport_exploration.pdf",
                    mime="application/pdf"
                )

            try:
                os.remove(pdf_path)
                for img in images:
                    os.remove(img)
            except Exception as e:
                st.warning(f"Erreur lors de la suppression des fichiers temporaires : {e}")












# # Ajout d'une image pour personnaliser la vue de la première partie sur l'analyse exploratoire
# col1, col2 = st.columns([1, 10])
# with col1:
#     # st.image("Image_1.png", width=50)
#     st.write("image1")
# with col2:
#     st.markdown("# Exploration descriptives des données")

# # Création d'un menu interactif pour l'affichage des données
# st.header("Vous pouvez choisir les options parmis la liste suivante: ")

# if st.checkbox("Afficher quelques lignes au hasard du dataframe "):
#     st.subheader("Lignes du dataframe")
#     st.dataframe(df.sample(5))

# if st.checkbox("Afficher les noms des colonnes "):
#     st.subheader("Noms des colonnes")
#     st.dataframe(df.columns)

# if st.checkbox("Afficher la taille du dataframe "):
#     st.write(f"Nombre de lignes : {len(df)}")

# if st.checkbox("Afficher les types de données par colonnes "):
#     st.subheader("Formats de donnée")
#     st.dataframe(df.dtypes)

# if st.checkbox("Afficher les statistiques descriptives "):
#     st.subheader(" Statistiques descriptives")
#     st.dataframe(df.describe().T.round(3).style.background_gradient())

# if st.checkbox("Afficher le nombre de catégorie de vin"):
#     st.subheader("Nombre de catégorie")
#     st.dataframe(df["target"].value_counts())
#     st.subheader("Répartition des catégories")
#     st.dataframe(df["target"].value_counts(normalize=True))

# if st.checkbox("Afficher le nombre de valeurs manquantes"):
#     st.subheader("Valeurs manquante")
#     st.dataframe(df.isnull().sum())

# # Ajout d'une image pour personnaliser la vue de la seconde partie
# col3, col4 = st.columns([1, 10])
# with col3:
#     # st.image("Image_2.png", width=70)
#     st.write("image2")
# with col4:
#     st.markdown("# Visualisation des données")

# st.header("Vous pouvez choisir les options de visualisations suivante: ")
# if st.checkbox("Sélectionner les distributions à afficher"):
#     select_distribution(df)

# if st.checkbox("Sélectionner les pairplots à afficher"):
#     st.write("Pour afficher le pairplot, veuillez sélectionner la target et l'associer à une autre variable")
#     select_pairplot(df)

# if st.checkbox("Créer un graphique de corrélation"):
#     corr = df.drop("target", axis=1).corr()
#     mask = np.triu(np.ones_like(corr, dtype=bool))
#     plot = sns.heatmap(corr, mask=mask,cmap=sns.diverging_palette(230, 20, as_cmap=True))
#     f, ax = plt.subplots(figsize=(11, 9))
#     #sns.heatmap(corr, mask=mask,cmap=sns.color_palette("Spectral", as_cmap=True))
#     st.pyplot(plot.get_figure())