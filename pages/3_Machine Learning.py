import streamlit as st
from config import style
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
import joblib
from modules.module_ML import (
    modeles_map,
    selection_meilleur_modele_par_cv,
    entrainer_modele,
    exporter_modele,
    grille_options
)
from routes import redirection


# Style adaptatif pour le mode clair/sombre
st.markdown(style, unsafe_allow_html=True)

# En-tête principal
st.markdown("""
    <div class="main-header">
        <h1>🦾 Entraînement d'un modèle de Machine Learning</h1>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            Entraînement d'un modèle de Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

if "df_clean" not in st.session_state:
    st.warning("❌ Veuiller importer des données avant de pouvoir entraîner un modèle.")
    st.stop()
else:
    df = st.session_state["df_clean"].copy()
    colonne_target = st.session_state["target_corr"]
    type_modele = st.session_state['task'] # "classification" ou "régression"
    X = df.select_dtypes('number').drop(colonne_target, axis = 1)
    y= df[colonne_target]
    
    # # Si la cible a été encodée par LabelEncoder, recréer le mapping
    # if isinstance(colonne_target, str) and colonne_target.startswith("target_encoded"):
    #     # On tente d’inférer le mapping à partir des types
    #     classes_possibles = df[colonne_target].unique()
    #     if "df" in st.session_state and "target_corr" in st.session_state:
    #         original_target = st.session_state["target_corr"]
    #         if isinstance(original_target, str) and original_target in df.columns:
    #             # on peut retrouver le mapping
    #             try:
    #                 from sklearn.preprocessing import LabelEncoder
    #                 encoder = LabelEncoder()
    #                 encoder.fit(st.session_state["df"][original_target])
    #                 mapping = {i: label for i, label in enumerate(encoder.classes_)}
    #                 st.session_state["mapping_target"] = mapping
    #             except Exception as e:
    #                 st.warning(f"Impossible de reconstruire le mapping de la target : {e}")


# Onglets de navigation
onglet1, onglet2, onglet3 = st.tabs([
    "🥇 Train Test Split & Comparaison des modèles",
    "🏋️‍♂️ Entraînement & Export",
    "⚙️ Optimisation des Hyperparamètres"
])



### Onglet 1 - 🥇 Train Test Split & Comparaison des modèles
with onglet1:
    with st.expander("ℹ️ Fonctionnement"):
        st.info("""
        Cette section permet de :
        - **Séparer le jeu de données** en un jeu d'entraînement/test
        - Lancer une **comparaison automatique de différents modèles** en fonction du type de tâche (régression ou classification) choisie dans la page précédente
        - Vous aurez pour cela le choix entre **LazyPredict** et **la Validation Croisée**
        """)
    
    
    
    ### TRAIN TEST SPLIT ###
    
    st.subheader("✂️ Séparation du jeu de données")
    # Sélecteur de taille du jeu de test
    taille_test=st.select_slider("Taille du jeu de test", options=["5%", "10%", "15%", "20%", "25%", "30%"], value="20%")
    translate = {"5%":0.05, "10%":0.1, "15%":0.15, "20%":0.2, "25%":0.25, "30%":0.3}
    test_size = translate[taille_test]
    
    if st.button("Séparer les données"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.success(f"✅ Le jeu de données à bien été scindé. Le jeu de test représente {taille_test} du jeu de données total")
        
        st.markdown(f"""
            - 🔍 **Jeu d'entraînement** : {X_train.shape[0]} lignes
            - 🧪 **Jeu de test** : {X_test.shape[0]} lignes
            - 🧠 **Tâche** : {type_modele}
            """)

    
    st.write("***")
    
    
    ### SELECTION DU MEILLEUR MODELE ###
    if "X_train" in st.session_state:
        st.subheader("🥇 Sélection du meilleur modèle")
        modele_selection = st.radio(
        "Méthode de sélection du meilleur modèle",
        ["Validation croisée", "LazyPredict"],
        index=0
        )
        st.session_state["modele_selection"] = modele_selection
        
        with st.expander("ℹ️ Aide"):
            st.info("✅ La validation croisée donne une estimation plus fiable des performances des modèles.")
            st.warning("⚠️ LazyPredict donne une estimation plus rapide mais moins robuste.")
            
        # si l'utilisateur choisit LazyPredict
        if modele_selection == "LazyPredict":
            if st.button("Exécuter LazyPredict"):
                
                X_train = st.session_state["X_train"]
                X_test = st.session_state["X_test"]
                y_train = st.session_state["y_train"]
                y_test = st.session_state["y_test"]
                    
                if type_modele == 'régression':
                    #Regression
                    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None )
                    models,predictions = reg.fit(X_train, X_test, y_train, y_test)
                else:
                    #Classification
                    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
                    models,predictions = clf.fit(X_train, X_test, y_train, y_test)

                models = models.reset_index(['Model'])
                st.dataframe(models)
                
                # Identifier le meilleur modèle (selon la métrique par défaut de LazyPredict)
                metric = "R^2" if type_modele == "régression" else "Accuracy"
                meilleur_modele = models.loc[models[metric].idxmax()]
                nom_meill_modl = meilleur_modele["Model"]
                score_meill_model = round(meilleur_modele[metric], 4)
                
                st.session_state["meilleur_modele"] = nom_meill_modl
                st.success(f"🏆 Meilleur modèle : **{nom_meill_modl}** avec un score de **{score_meill_model}** ({metric})")
                

        # si l'utilisateur choisit la validation croisée
        elif modele_selection == "Validation croisée":
            # Choix du nombre de folds
            nb_folds=st.slider("Nombre de folds", min_value=2, max_value=10, value=5)
                
            if st.button("Exécuter la Validation Croisée"):
                
                if type_modele == "régression":
                    modeles_a_tester = {
                        "LinearRegression": LinearRegression(),
                        "Ridge": Ridge(),
                        "Lasso": Lasso(),
                        "DecisionTreeRegressor": DecisionTreeRegressor(),
                        "RandomForestRegressor": RandomForestRegressor()
                    }
                    scoring = "neg_root_mean_squared_error"  # ou "r2"
                else:
                    modeles_a_tester = {
                        "LogisticRegression": LogisticRegression(max_iter=1000),
                        "SVC": SVC(),
                        "DecisionTreeClassifier": DecisionTreeClassifier(),
                        "RandomForestClassifier": RandomForestClassifier()
                    }
                    scoring = "accuracy"

                df_cv = selection_meilleur_modele_par_cv(modeles_a_tester, X, y, scoring, nb_folds)

                st.dataframe(df_cv)
                
                if not df_cv.empty:
                    meilleur_modele = df_cv.iloc[0]
                    nom_meill_modl = meilleur_modele["Modèle"]
                    score_meill_model = meilleur_modele[f"Score moyen ({scoring})"]

                    st.session_state["meilleur_modele"] = nom_meill_modl
                    st.success(f"🏆 Meilleur modèle : **{nom_meill_modl}** avec un score moyen de **{score_meill_model}** sur {nb_folds}-folds")
        
        st.write("***")
        st.markdown("### Vous pouvez maintenant passer à l'onglet suivant : 🏋️‍♂️ Entraînement & Export")
        
### Onglet 2 - 🏋️‍♂️ Entraînement & Export
with onglet2:
    
    
    with st.expander("ℹ️ Fonctionnement"):
        st.info("""
        Cette section permet :
        - **D'entraîner** le modèle sélectionné précédemment
        - **D'exporter** le modèle en fichier pickles au format "nom_du_modele.pkl"
        """)
    
    st.write("***")
    
    if "meilleur_modele" in st.session_state:
        meilleur_modele = st.session_state["meilleur_modele"]
        st.markdown("### 🏋️‍♂️ Entraînement final")
        if st.button(f"Lancer l'entraînement du modèle {meilleur_modele}"):
            entrainer_modele(meilleur_modele, X, y)
                
        st.write("***")
        
        # Export PKL
        if "modele_final" in st.session_state:  # déclenchement de l'export quand modele_final dans session_state 
            modele_final=st.session_state["modele_final"]
            exporter_modele(modele_final, meilleur_modele, key_suffix="entraine")
        
        st.write("***")
        st.markdown("### Vous pouvez maintenant passer à l'onglet suivant : ⚙️ Optimisation des Hyperparamètres")

### Onglet 3 - ⚙️ Optimisation des Hyperparamètres
with onglet3:
    st.subheader("⚙️ Optimisation des Hyperparamètres")
    
    if "modele_final" not in st.session_state:
        st.warning("❌ Veuillez d’abord sélectionner un modèle à optimiser dans l’onglet 1.")
        st.stop()

    modele = st.session_state["modele_final"]
    modele = str(modele).split("(")[0].strip()
    st.info(f"Modèle sélectionné à optimiser : **{modele}**")
    
    if modele not in grille_options:
        st.warning("⚠️ Ce modèle n’a pas encore de grille d’optimisation préconfigurée.")
        st.stop()

    # Choix de la méthode à utiliser et du nombre de folds
    methode = st.radio("Méthode d’optimisation", ["GridSearchCV", "RandomizedSearchCV"])
    nb_folds=st.slider("Nombre de folds pour la validation croisée", min_value=2, max_value=10, value=5)

    model_select = grille_options[modele]["modele"]
    param_grille = grille_options[modele]["params"]

    if st.button("🚀 Lancer l’optimisation"):
        st.markdown("⏳ Optimisation en cours...")

        if methode == "GridSearchCV":
            search = GridSearchCV(model_select, param_grille, cv=nb_folds, scoring="r2" if type_modele == "régression" else "accuracy")
        else:
            search = RandomizedSearchCV(model_select, param_grille, n_iter=10, cv=nb_folds, scoring="r2" if type_modele == "régression" else "accuracy", random_state=42)

        search.fit(X, y)
        meilleur_modele_optimise = search.best_estimator_
        meilleurs_params = search.best_params_
        meilleur_score = search.best_score_

        st.success("✅ Optimisation terminée !")
        st.markdown(f"🏆 **Meilleurs hyperparamètres :** `{meilleurs_params}`")
        st.markdown(f"📈 **Score moyen CV :** `{round(meilleur_score, 4)}`")
        
        st.session_state["meilleur_modele_optimise"] = meilleur_modele_optimise # pour déclencher ensuite l'export 
        
        st.write("***")
        
        # Export PKL
        if "meilleur_modele_optimise" in st.session_state:  # déclenchement de l'export quand meilleur_modele_optimise dans session_state 
            meilleur_modele_optimise=st.session_state["meilleur_modele_optimise"]
            exporter_modele(meilleur_modele_optimise, modele, key_suffix="optimise")

    # Redirection page suivante
    redirection("📝 Évaluations", "4_Evaluation")