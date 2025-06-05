import streamlit as st
from config import style
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.module_explo import generer_rapport_pdf, telecharger_donnees, traiter_outliers, detection_outliers, encoder_cible, standardisation
from sklearn.impute import SimpleImputer
import os
from routes import redirection

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
onglet1, onglet2, onglet3, onglet4, onglet5, onglet6 = st.tabs([
    "🧬 Types & Tâche",
    "📊 Distributions",
    "📈 Corrélations",
    "🧹 NaN & Outliers",
    "⚖️ Standardisation",
    "🎯 Résumé & Exports"
])


### Onglet 1 - Types & Cible
with onglet1:

    with st.expander("ℹ️ Fonctionnement"):
        st.info("""
        Cette section permet d'identifier les types de chaque variable (numérique, catégorielle...) et de choisir la variable cible pour la modélisation. \n
        En fonction de la cible choisie, l'application propose automatiquement un type de tâche : **classification** (si 10 modalités ou moins) ou **régression** (valeurs continues).\n
        Vous pouvez forcer ce choix manuellement.
        """)
    
    # Affichage du type de données pour chaque colonne
    st.subheader("🧬 Types de données")
    st.dataframe(df.dtypes.reset_index().rename(columns={0: "Type", "index": "Colonne"}))

    st.write("***")
    
    # L'utilisateur choisit sa colonne cible
    st.subheader("🎯 Sélection de la variable cible")
    target = st.selectbox("Choisissez la colonne cible", df.columns)

    st.write("***")
    
    # Si la cible comporte jusqu'à 10 valeurs différente, on estime que c'est de la Classification sinon Régression
    st.subheader("🧠 Type de tâche")
    # Autodétection
    auto_detected_task = "classification" if df[target].nunique() <= 10 else "régression"
    st.write(f"Suggestion automatique : **{auto_detected_task}**")

    # Choix par l'utilisateur avec le choix autodétecté par défaut
    task = st.radio("Choisissez le type de tâche", ["classification", "régression"], index=0 if auto_detected_task == "classification" else 1)
    st.success(f"Type de tâche sélectionné : **{task}**")
    st.session_state['task']=task

    st.write("***")
    st.markdown("### Vous pouvez maintenant passer à l'onglet suivant : 📊 Distributions")

### Onglet 2 - Distributions
with onglet2:
    st.subheader("📊 Distribution des variables")
    
    with st.expander("ℹ️ Fonctionnement"):
        st.info("""
        Cet onglet permet d'explorer la distribution des valeurs d'une variable :
        - Pour les colonnes **numériques**, un histogramme avec une courbe de densité (KDE) montre la forme de la distribution (normale, asymétrique, etc).
        - Pour les colonnes **catégorielles**, un diagramme à barres indique la fréquence de chaque modalité.

        Cela aide à détecter les valeurs aberrantes, les déséquilibres ou les transformations à appliquer avant le traitement.
        """)

    col1, col2, col3 = st.columns([1, 2, 1])  # colonne centrale plus large pour le graph
    with col1:
        # Sélection de la colonne à analyser
        selected_col = st.selectbox("Sélectionnez une colonne", df.columns)
    with col2:
        # Si la colonne est de type numérique : histplot + kde
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        else:
            # Sinon, barplot
            counts = df[selected_col].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=counts.index, y=counts.values, ax=ax)
            for i, v in enumerate(counts.values):
                ax.text(i, v, str(v), ha='center', va='bottom')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
    st.write("***")
    st.markdown("### Vous pouvez maintenant passer à l'onglet suivant : 📈 Corrélations")

### Onglet 3 - Corrélations
with onglet3:
    with st.expander("ℹ️ Fonctionnement"):
        st.info("""
        Cette section permet de visualiser les relations entre les variables numériques grâce à une matrice de corrélation.\n
        Elle aide à détecter les variables redondantes ou très corrélées, qu'il peut être utile d'exclure pour éviter les multicolinéarités.\n
        Elle permet également de sélectionner les colonnes à conserver pour la modélisation, en se basant sur cette matrice.
        """)
        

    # ENCODAGE Si Classification et target non numérique
    df_corr = st.session_state.get("df_corr", df.copy())
    target_corr = st.session_state.get("target_corr", target)
    
    if task == "classification" and not pd.api.types.is_numeric_dtype(df_corr[target]):
        # Si on est sur une classification, l'utilisateur choisit entre 3 méthodes d'encodage
        choix_encoder = st.selectbox("Encodage de la cible", ["Label Encoding", "One-Hot Encoding", "get_dummies"])
        if choix_encoder == "Label Encoding":
            drop_first = False
        else:
            drop_first = st.checkbox("Supprimer la première modalité (évite la multicolinéarité)", value=False)
        
        # Application de l'encodage choisi
        if st.button("Appliquer l'encodage"):
            df_corr = df.copy()
            df_corr, encoded_target_name = encoder_cible(df_corr, target, choix_encoder, drop_first)
            df_corr = df_corr.loc[:, ~df_corr.columns.duplicated()]
            st.session_state["df_corr"] = df_corr
            st.session_state["target_corr"] = encoded_target_name
            st.session_state["choix_encoder"] = choix_encoder
            st.success("Encodage appliqué avec succès.")
            target_corr = encoded_target_name
        # Affichage de l'encoder appliqué actuellement
        if "choix_encoder" in st.session_state:
            st.info(f"Encodage actuellement appliqué : `{st.session_state['choix_encoder']}`")
    else:
        st.session_state["df_corr"] = df_corr
        st.session_state["target_corr"] = target_corr
    
    numeric_cols = df_corr.select_dtypes(include=np.number).columns.tolist()
    target_cols = st.session_state["target_corr"]
    target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
    cols_to_include = [col for col in numeric_cols if col in df_corr.columns]
    corr = df_corr[cols_to_include].corr()

    st.subheader("🔢 Aperçu des données")
    st.dataframe(df_corr.sample(n=10))
    
    ### MATRICE DE CORRELATION
    st.subheader("📈 Matrice de corrélation")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("***")
    
    # Si régression, on conserve les colonnes ayant une corrélation >0.2 avec la target sinon >0.1
    seuil_bas = 0.2 if task == "régression" else 0.1
    # Dans les deux cas, on conserve les colonnes ayant une corrélation <0.95 avec la target
    seuil_haut = 0.95
    
    st.subheader(f"📉 Colonnes pas assez corrélées (< {seuil_bas})")
    to_drop = []
    for col in corr.columns:
        if col != target_corr:
            val_corr = corr.at[col, target_corr]
            if abs(val_corr) < seuil_bas:
                to_drop.append({"Colonne": col, "Corrélation avec la cible": corr.at[col, target_corr]})
                
    if to_drop:
        df_to_drop = pd.DataFrame(to_drop).sort_values(by="Corrélation avec la cible")
        st.dataframe(df_to_drop, hide_index=True)
    else:
        st.write("Pas de colonne faiblement corrélée à la cible")
    
    
    st.subheader(f"📉 Colonnes trop corrélées (> {seuil_haut})")
    # to_drop = set()
    corr_trop_fortes = []
    for i in range(len(corr.columns)):
        for j in range(i):
            # if abs(corr.iloc[i, j]) > seuil_haut:
            #     to_drop.add(corr.columns[i])
            val_corr = corr.iloc[i, j]
            if abs(val_corr) > seuil_haut:
                corr_trop_fortes.append({
                    "Colonne 1": corr.columns[i],
                    "Colonne 2": corr.columns[j],
                    "Corrélation": round(val_corr, 3)
                })
    # if to_drop :
    #     df_to_drop = pd.DataFrame(list(to_drop), columns=["Colonnes à supprimer (trop corrélées)"])
    if corr_trop_fortes:
        df_corr_fortes = pd.DataFrame(corr_trop_fortes).sort_values(by="Corrélation", ascending=False)
        st.dataframe(df_to_drop, hide_index=True)
    else:
        st.write("Pas de colonne trop corrélée")
    
    st.write("***")
    
    st.subheader("✅ Sélection guidée des colonnes")
    
    # if task == "régression":
    #     # Si régression, on conserve les colonnes ayant une corrélation >0.2 avec la target
    #     good_corr_cols = corr[target_corr][abs(corr[target_corr]) > 0.2].drop(target_corr).index.tolist()
    # else:
    #     # Si classification, on conserve les colonnes ayant une corrélation >0.1 avec la target
    #     if target_corr in corr.columns:
    #         good_corr_cols = corr[target_corr][abs(corr[target_corr]) > 0.1].drop(target_corr).index.tolist()
    #     else:
    #         st.warning(f"La colonne cible '{target_corr}' n'est plus présente dans les données.")
    #         good_corr_cols = []
    
    # if target_corr in corr.columns:
    #     # Si régression, on conserve les colonnes ayant une corrélation >0.2 avec la target
    #     if task == "régression":
    #         seuil_bas = 0.2
    #     # Si classification, on conserve les colonnes ayant une corrélation >0.1 avec la target
    #     else:
    #         seuil_bas = 0.1
    #     # Dans les deux cas, on conserve les colonnes ayant une corrélation <0.95 avec la target
    #     seuil_haut = 0.95
        
    #     # Colonnes corrélées à la target entre 0.1 ou 0.2 et 0.95
    #     filtres_target = (abs(corr[target_corr]) > seuil_bas) & (abs(corr[target_corr]) < seuil_haut)
    #     # good_corr_cols = corr[target_corr][filtres_target].drop(target_corr).index.tolist()
    #     good_corr_cols = corr[target_corr][filtres_target].index.tolist()
    
    
    # Vérifier que toutes les colonnes cible sont bien dans la matrice
    if isinstance(target_corr, list):
        cibles_valides = [c for c in target_corr if c in corr.columns]
        if not cibles_valides:
            st.warning("Aucune colonne cible encodée n'est présente dans la matrice de corrélation.")
            good_corr_cols = []
        else:
            # Agrégation des corrélations sur les colonnes cibles
            corr_scores = pd.Series(0, index=corr.columns)
            for col in cibles_valides:
                corr_scores += abs(corr[col])
            corr_scores /= len(cibles_valides)

            filtres_target = (corr_scores > seuil_bas) & (corr_scores <= seuil_haut)
            good_corr_cols = corr_scores[filtres_target].drop(labels=cibles_valides, errors="ignore").index.tolist()
    else:
        if target_corr not in corr.columns:
            st.warning(f"La colonne cible '{target_corr}' n'est plus présente dans les données.")
            good_corr_cols = []
        
        else:
            # Si régression, on conserve les colonnes ayant une corrélation >0.2 avec la target
            if task == "régression":
                seuil_bas = 0.2
            # Si classification, on conserve les colonnes ayant une corrélation >0.1 avec la target
            else:
                seuil_bas = 0.1
            
            # Colonnes corrélées à la target entre 0.1 ou 0.2 et 0.95
            filtres_target = (abs(corr[target_corr]) > seuil_bas) & (abs(corr[target_corr]) <= 0.95)
            good_corr_cols = corr[target_corr][filtres_target].index.tolist()
        
        # Élimination des colonnes trop corrélées entre elles (>0.95)
        if good_corr_cols:
            corr_matrix = df[good_corr_cols].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            good_corr_cols = [col for col in good_corr_cols if col not in to_drop]
        else:
            to_drop=[]
            
    selected_cols_corr = st.multiselect("Colonnes à conserver pour la modélisation", numeric_cols, default=good_corr_cols)
    
    # # initialiser df_clean dans le session_state pour l'onglet 4
    if st.button("✅ Valider cette version comme jeu de données nettoyé"):
        df_corr_validated = df_corr[selected_cols_corr + [target_corr]].copy()
        st.session_state["df_clean"] = df_corr_validated
        st.success("La version avec encodage a été validée comme jeu de données nettoyé (df_clean).")

        st.write("***")
        st.markdown("### Vous pouvez maintenant passer à l'onglet suivant : 🧹 NaN & Outliers")

# Onglet 4 - NaN & Outliers
with onglet4:
    if "df_clean" in st.session_state:
        df=st.session_state["df_clean"].copy()
        
        with st.expander("ℹ️ Fonctionnement"):
            st.info("""
            Cette onglet permet :
            - d'analyser les valeurs manquantes (NaN) et de choisir une méthode d'imputation (moyenne, médiane, valeur la plus fréquente)
            - de détecter les valeurs aberrantes (outliers), afin de les supprimer ou ajuster au besoin.
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
            if not selected_nan:
                st.warning("Veuillez sélectionner au moins une colonne à traiter.")
            else:
                imputer = SimpleImputer(strategy=imputation_strategy)
                df[selected_nan] = pd.DataFrame(
                        imputer.fit_transform(df[selected_nan]),
                        columns=selected_nan,
                        index=df.index
                    )
                st.success("NaN remplacés avec succès")
                st.session_state["df_clean"] = df
            
        st.write("***")

        # Gestion des  Outliers
        st.subheader("🚨 Détection des outliers")

        # L'utilisateur choisit la méthode de détection des outliers 
        methode_outlier = st.radio("Méthode de détection", ["Z-score", "IQR"])

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()  # recalcul après potentiel suppression de colonnes
        outliers_bool = detection_outliers(df, numeric_cols, methode_outlier, key="zscore1")

        st.write("Nombre d'outliers par colonne :")
        st.write(outliers_bool.sum())

        traitement_outlier = st.selectbox("Action", ["Aucune", "Supprimer les lignes", "Remplacer par la médiane"])
        
        if st.button("Remplacer les Outliers"):
            if traitement_outlier != "Aucune":
                df = traiter_outliers(df, numeric_cols, outliers_bool, traitement_outlier)
                st.success("Traitement des outliers effectué.")
                st.session_state["df_clean"] = df
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                outliers_bool_clean = detection_outliers(df, numeric_cols, methode_outlier, key="zscore2")
                st.write(outliers_bool_clean.sum())
        
        st.write("***")
        st.markdown("### Vous pouvez maintenant passer à l'onglet suivant : ⚖️ Standardisation")
            
# Onglet 5 - Standardisation
with onglet5:
    if "df_clean" in st.session_state:
        df_clean = st.session_state["df_clean"].copy()
        
        with st.expander("ℹ️ Fonctionnement"):
            st.info("""
                Cette onglet permet de standardiser les données afin que certaines colonnes n'aient pas un poids supérieur à d'autres dans la modélisation.\n
                Il vous sera indiqué si vos données semblent déjà standardisées ou non et si ce n'est pas le cas, vous pourrez cocher la case pour le faire.
            """)
            
        colonne_target = st.session_state["target_corr"]
        
        st.subheader("⚖️ Standardisation des données")
        standardisation(df_clean, colonne_target)    
        
        st.write("***")
        st.markdown("### Vous pouvez maintenant passer à l'onglet suivant : 🎯 Résumé & Exports")
    
# Onglet 6 - Résumé & Exports
with onglet6:
    if "df_clean" in st.session_state:
        df_clean = st.session_state["df_clean"].copy()
        
        with st.expander("ℹ️ Fonctionnement"):
            st.info("""
            Cette onglet permet :
            - d'afficher un résumé des observations et traitements réalisés,
            - de télécharger les données au format CSV ou XLSX,
            - de générer et télécharger un rapport d'exploration et traitement au format PDF.
            """)
        
        # Résumé des traitements effectués
        st.subheader("📌 Résumé des traitements effectués")
        # 1. Nombre de lignes et colonnes
        n_lignes, n_colonnes = df_clean.shape
        # 2. Colonnes supprimées pour forte corrélation
        nb_col_corr_suppr = len(to_drop)
        # 3. Colonnes conservées
        nb_colonnes_conservees = len(selected_cols_corr)
        # 4. Type de tâche
        type_tache_resume = task
        # 5. Méthode d'encodage
        encoder_resume = st.session_state.get("choix_encoder", "Aucun encodage (cible numérique)")
        # Affichage
        st.markdown(f"""
        - ✅ **{n_lignes} lignes** et **{n_colonnes} colonnes** finales
        - 🧠 **Type de tâche** sélectionné : **{type_tache_resume}**
        - 🎯 **Variable cible encodée avec** : `{encoder_resume}`
        - 🧹 **Colonnes supprimées pour forte corrélation** (> 0.95) : {nb_col_corr_suppr}
        - 📊 **Nombre de colonnes conservées** pour la modélisation : {nb_colonnes_conservees}
        """)
        if st.session_state.get("standardized", False):
            nb_std_cols = len(st.session_state.get("standardized_columns", []))
            st.markdown(f"- ⚖️ **Standardisation appliquée sur {nb_std_cols} colonnes**")
            with st.expander("🔍 Détail des colonnes standardisées (moyenne et écart-type)"):
                st.dataframe(st.session_state.get("standardized_stats", pd.DataFrame()))
        else:
            st.markdown("- ⚖️ **Aucune standardisation appliquée**")

        st.write("***")
        
        # Téléchargement des données au format CSV ou XLSX
        telecharger_donnees(df_clean)
        
        st.write("***")
        
        # Téléchargement d'un Rapport d'explo en PDF
        st.subheader("📝 Générer un rapport de l'exploration et des traitements en PDF")

        if st.button("📄 Générer le rapport PDF"):
            pdf_path, images = generer_rapport_pdf(df, df_clean, target_corr, selected_cols_corr, task, to_drop, corr)

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
        
        # Redirection page suivante
        redirection("🦾 Entraînement d'un modèle", "3_Machine Learning")