import streamlit as st
import pandas as pd
import csv

### Ajout du df au session_state
def state_write(df):
    st.session_state['df'] = df

### Upload csv ou xlsx
def import_fichier():
    file = st.file_uploader("Uploader un fichier (.csv ou .xlsx)", type=["csv", "xlsx"])
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

        except Exception as e:
            st.error(f"Erreur lors de l'import : {e}")

        # Enregistrement dans session_state
        state_write(df)
