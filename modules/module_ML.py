
import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR



### Onglet 1 - Sélection du modèle ###


# Mapping nom : classe de modèle
modeles_map = {
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "SVR": SVR
}


# Fonction : validation croisée pour sélection

def selection_meilleur_modele_par_cv(modeles_a_tester: dict, X, y, scoring: str, nb_folds: int) -> pd.DataFrame:
    scores_cv = {}
    for nom, modele in modeles_a_tester.items():
        try:
            score = cross_val_score(modele, X, y, cv=nb_folds, scoring=scoring)
            scores_cv[nom] = round(score.mean(), 4)
        except Exception as e:
            print(f"Erreur avec {nom} : {e}")
    return pd.DataFrame(scores_cv.items(), columns=["Modèle", f"Score moyen ({scoring})"]).sort_values(f"Score moyen ({scoring})", ascending=False)



### Onglet 2 - Entraînement & Export ###


# Fonction : entraînement d'un modèle

def entrainer_modele(nom_modele: str, X, y):
    if nom_modele not in modeles_map:
        raise ValueError("❌ Ce modèle n'est pas encore supporté pour l'entraînement")
    modele_final = modeles_map[nom_modele]()
    modele_final.fit(X, y)
    st.success("✅ Le modèle a bien été entraîné")
    st.session_state["modele_final"] = modele_final  # pour déclencher ensuite l'export 


# Fonction export d’un modèle au format .pkl (valable aussi onglet 3)

def exporter_modele(modele, nom_fichier, key_suffix: str = ""):
    """
    Exporte un modèle .pkl et propose le téléchargement
    
    Args:
        modele: Le modèle scikit-learn à sauvegarder.
        nom_fichier: Le nom du fichier exporté (.pkl).
        nom_visible: Le nom affiché à l'utilisateur (optionnel).
    """
    st.markdown("### 📤 Export")
    joblib.dump(modele, f"{nom_fichier}.pkl")
    with open(f"{nom_fichier}.pkl", "rb") as fichier:
        st.download_button(
            label=f"⬇️ Télécharger le modèle {'optimisé' if 'optim' in key_suffix else 'entraîné'} (.pkl)",
            data=fichier,
            file_name=f"{nom_fichier}_{key_suffix}.pkl",
            mime="application/octet-stream",
            key=f"download_{nom_fichier}_{key_suffix}"  # Clé unique pour éviter les doublons entre les 2 exports, personnaliser le label et le nom de fichier
        )



### Onglet 3 - Optimisation Hyperparamètres ###


# Mapping des modèles les plus courants pour l'optimisation des Hyperparamètres

grille_options = {
    "RandomForestClassifier": {
        "modele": RandomForestClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "RandomForestRegressor": {
        "modele": RandomForestRegressor(),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "LogisticRegression": {
        "modele": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "Ridge": {
        "modele": Ridge(),
        "params": {
            "alpha": [0.01, 0.1, 1, 10, 100],
            "solver": ["auto", "svd", "cholesky", "saga"]
        }
    },
    "Lasso": {
        "modele": Lasso(max_iter=10000),
        "params": {
            "alpha": [0.01, 0.1, 1, 10],
            "selection": ["cyclic", "random"]
        }
    },
    "ElasticNet": {
        "modele": ElasticNet(max_iter=10000),
        "params": {
            "alpha": [0.01, 0.1, 1, 10],
            "l1_ratio": [0.1, 0.5, 0.9]
        }
    },
    "SVC": {
        "modele": SVC(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        }
    },
    "SVR": {
        "modele": SVR(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "epsilon": [0.1, 0.2, 0.5]
        }
    },
    "DecisionTreeClassifier": {
        "modele": DecisionTreeClassifier(),
        "params": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "DecisionTreeRegressor": {
        "modele": DecisionTreeRegressor(),
        "params": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "KNeighborsClassifier": {
        "modele": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }
    },
    "KNeighborsRegressor": {
        "modele": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }
    },
    "GradientBoostingClassifier": {
        "modele": GradientBoostingClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10]
        }
    },
    "GradientBoostingRegressor": {
        "modele": GradientBoostingRegressor(),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10]
        }
    }
}

