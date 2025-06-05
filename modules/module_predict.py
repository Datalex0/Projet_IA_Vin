
import pandas as pd
import joblib
import streamlit as st
import csv


def charger_modele_externe(fichier):
    """
    Importe un modèle au format .pkl
    """
    try:
        modele = joblib.load(fichier)
        st.success("✅ Modèle chargé avec succès.")
        return modele
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle : {e}")
        return None
    
    
def appliquer_modele(modele, df_input):
    """
    Applique un modèle à un DataFrame d'entrée
    """
    df_resultats = df_input.copy()

    if df_resultats.isnull().values.any():
        st.error("❌ Certaines valeurs saisies sont invalides ou manquantes. Veuillez corriger les entrées.")
        df_resultats["Prédiction"] = None
        return df_resultats

    try:
        # Appliquer la même standardisation si elle existe
        if "scaler" in st.session_state and "standardized_columns" in st.session_state:
            scaler = st.session_state["scaler"]
            cols = st.session_state["standardized_columns"]
            try:
                df_resultats[cols] = scaler.transform(df_resultats[cols])
                st.success("🔄 Standardisation appliquée automatiquement avant la prédiction")
            except Exception as e:
                st.warning(f"⚠️ Erreur lors de l'application du scaler : {e}")

        pred = modele.predict(df_resultats)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        df_resultats["Prédiction"] = None
        return df_resultats

    # Décodage si mapping inverse
    if "mapping_target" in st.session_state:
        mapping = st.session_state["mapping_target"]
        if isinstance(mapping, dict):
            try:
                pred_series = pd.Series(pred).astype(int)
                pred = pred_series.map(mapping).fillna(pred_series).tolist()
            except Exception as e:
                st.warning(f"Erreur lors du décodage des prédictions : {e}")

    df_resultats["Prédiction"] = pred
    return df_resultats


def saisir_donnees_manuellement(colonnes, nb_colonnes_par_ligne=3):
    """
    Crée un formulaire pour la saisie manuelle de chaque colonne
    """
    valeurs = {}
    for i in range(0, len(colonnes), nb_colonnes_par_ligne):
        cols = st.columns(nb_colonnes_par_ligne)
        for j, col in enumerate(colonnes[i:i+nb_colonnes_par_ligne]):
            with cols[j]:
                val = st.text_input(f"{col}", key=f"saisie_{col}")
                val = val.strip().replace(",", ".")
                try:
                    val = float(val)
                except:
                    val = None
                valeurs[col] = val

    return pd.DataFrame([valeurs])


def import_fichier():
    '''Module permettant d'importer un fichier CSV ou XLSX'''
    
    file = st.file_uploader("Uploader un fichier (.csv ou .xlsx) avec les mêmes colonnes que le modèle (hors cible)", type=["csv", "xlsx"])
    button = st.button('importer')

    if button and file is not None:

        try:
            if file.name.endswith(".csv"):
                content = file.getvalue().decode("utf-8")
                
                # Utiliser csv.Sniffer pour identifier automatiquement le séparateur
                dialect = csv.Sniffer().sniff(content)
                separator = dialect.delimiter

                # Lire le fichier CSV dans un DataFrame pandas en utilisant le séparateur identifié
                df = pd.read_csv(file, sep=separator)

            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
            st.success(f'✅ Données issues de "{file.name}" importées avec succès !')
            st.write("Aperçu des données importées :")
            st.dataframe(df.head())
            st.session_state["df_new"] = df
        
        except Exception as e:
            st.error(f"Erreur lors de l'import : {e}")
            st.session_state["df_new"] = None
            
        
        