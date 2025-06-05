
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

def evaluation_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Rapport sous forme de DataFrame
    rapport = classification_report(y_test, y_pred, output_dict=True)
    df_rapport = pd.DataFrame(rapport).transpose().round(2)
    # Affichage stylisé
    st.dataframe(df_rapport.style.background_gradient(cmap="Blues", axis=1).format(precision=2))
    
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Valeurs prédites")
    ax.set_ylabel("Valeurs réelles")
    ax.set_title("Matrice de confusion")
    
    col1, col2, col3 = st.columns([2, 1, 1])  # colonne centrale plus large pour le graph
    with col1:
        st.pyplot(fig)

def evaluation_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown("### 📈 Résultats de la régression")
    st.markdown(f"- **RMSE** : `{rmse:.2f}`")
    st.markdown(f"- **R²** : `{r2:.2f}`")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Valeurs réelles")
    ax.set_ylabel("Prédictions")
    ax.set_title("Valeurs réelles vs. Prédictions")
    st.pyplot(fig)

def afficher_evaluation(model, X_test, y_test, type_tache):
    if type_tache == "classification":
        evaluation_classification(model, X_test, y_test)
    else:
        evaluation_regression(model, X_test, y_test)


def fonctionnement(type_tache):
    if type_tache == "classification":
        st.markdown("### 📋 Rapport de Classification")
        with st.expander("ℹ️ Fonctionnement"):
            st.info("""
                Cette section affiche des métriques d'évaluation du modèle. Ces métriques seront différentes en fonction du type de tâche (Classification ou Régression) effectuée.\n
                Nous sommes ici sur de la **Classification**.\n
                - Interprétation des lignes :\n
                    * 0, 1, 2... représentent les **Classes Prédictives**\n
                    * accuracy représente l’**Exactitude Globale** : proportion totale de bonnes prédictions\n
                    * macro avg	représente la **Moyenne Non Pondérée** des métriques par classe\n
                    * weighted avg représente la **Moyenne Pondérée** par le support (nombre d’exemples de chaque classe)\n
                - Interprétation des colonnes :\n
                    * precision représente le nombre de prédictions correctes parmi les prédictions positives pour cette classe,\n
                    * recall montre combien des vrais éléments de cette classe ont été bien retrouvés,\n
                    * f1-score représente la **Moyenne Harmonique** entre précision et rappel (équilibre entre les deux)\n
                    * le support affiche le nombre de cas réels pour chaque classe
            """)
    else:
        st.markdown("### 📋 Rapport de Régression")
        with st.expander("ℹ️ Fonctionnement"):
            st.info("""
                Cette section affiche des métriques d'évaluation du modèle. Ces métriques seront différentes en fonction du type de tâche (Classification ou Régression) effectuée.\n
                Nous sommes ici sur de la **Régression**.\n
                Interprétation des métriques :
                    * RMSE (Root Mean Squared Error) : Racine de la moyenne des carrés des erreurs\n
                        Plus bas = meilleures prédictions (en unités de la variable cible)
                    * MAE (Mean Absolute Error) : Moyenne des erreurs absolues\n
                        Plus bas = erreurs faibles, moins sensibles aux outliers
                    * R² (Coefficient de détermination) : Proportion de variance expliquée par le modèle\n
                        1.0 = parfait, 0 = pire qu'une moyenne constante, peut être < 0 si catastrophique
                    """)