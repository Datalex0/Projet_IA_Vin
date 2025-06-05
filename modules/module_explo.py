import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import tempfile
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

### Onglet 3 - Corrélations ###

def encoder_cible(df, target, methode, drop_first=False):
    """Encode la target selon la méthode choisie par l'utilisateur"""
    df_copy = df.copy()
    encoded_target_name = "target_encoded"

    if methode == "Label Encoding":
        encoder = LabelEncoder()
        df_copy["target_encoded"] = encoder.fit_transform(df_copy[target])
        # Stockage du mapping inverse pour Label Encoding pour afficher les valeurs initiales avant encodage lors de la prédiction
        st.session_state["mapping_target"] = {i: label for i, label in enumerate(encoder.classes_)}

    elif methode == "One-Hot Encoding":
        ohe = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None)
        encoded_data = ohe.fit_transform(df_copy[[target]])
        encoded_cols = ohe.get_feature_names_out([target])
        df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_copy.index)
        # df_copy = pd.concat([df_copy.drop(columns=[target]), df_encoded], axis=1)
        df_copy = pd.concat([df_copy, df_encoded], axis=1)
        df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()].copy()
        encoded_target_name = encoded_cols.tolist()  # Liste de colonnes encodées
        
        # Stockage du mapping inverse pour One-Hot pour afficher les valeurs initiales avant encodage lors de la prédiction
        mapping_inv = {}
        for full_col in encoded_cols:
            # Exemple : 'target_classname'
            if "_" in full_col:
                original_value = full_col.split("_", 1)[1]
                mapping_inv[full_col] = original_value
        st.session_state["mapping_target"] = mapping_inv

    elif methode == "get_dummies":
        df_copy = pd.get_dummies(df_copy, columns=[target], drop_first=drop_first)
        df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()].copy()
        encoded_cols = [col for col in df_copy.columns if col.startswith(target + "_")]
        encoded_target_name = encoded_cols  # Liste des colonnes générées
        
        # Stockage du mapping inverse pour get_dummies
        mapping_inv = {}
        for col_name in encoded_cols:
            original_value = col_name.split("_", 1)[1]
            mapping_inv[col_name] = original_value
        st.session_state["mapping_target"] = mapping_inv

    else:
        raise ValueError("Méthode d'encodage non reconnue.")

    st.session_state["target_corr"] = encoded_target_name  # MAJ de la variable
    return df_copy, encoded_target_name



### Onglet 4 - NaN et Outliers ###

def detecter_outliers_zscore(df, seuil=3.0):
    """Retourne un DataFrame booléen où True indique un outlier selon le Z-score."""
    z_scores = np.abs((df - df.mean()) / df.std(ddof=0))
    return z_scores > seuil

def detecter_outliers_iqr(df, seuil=1.5):
    """Retourne un DataFrame booléen où True indique un outlier selon la méthode IQR."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - seuil * IQR
    upper_bound = Q3 + seuil * IQR
    return (df < lower_bound) | (df > upper_bound)

def detection_outliers(df, numeric_cols, methode_outlier, key=None):
    if methode_outlier == "Z-score":
        seuil = st.slider("Seuil Z-score", 2.0, 5.0, 3.0, key=key)
        outliers_bool = detecter_outliers_zscore(df[numeric_cols], seuil)
    else:
        seuil = 1.5
        outliers_bool = detecter_outliers_iqr(df[numeric_cols], seuil)
    return outliers_bool

def traiter_outliers(df, numeric_cols, outliers_bool, action="Supprimer les lignes"):
    """Traite les outliers en fonction de l'action choisie."""
    if action == "Supprimer les lignes":
        to_drop = outliers_bool.any(axis=1)
        return df[~to_drop]
    elif action == "Remplacer par la médiane":
        df_copy = df.copy()
        for col in numeric_cols:
            median = df_copy[col].median()
            df_copy.loc[outliers_bool[col], col] = median
        return df_copy
    return df



### Onglet 5 - Standardisation ###

def standardisation(df, colonne_target):
    # Affichage du df avant standardisation
    st.markdown("**Aperçu du Dataframe AVANT Standardisation**")
    st.dataframe(df.head())
    st.write("***")
    # définir un seuil de proximité de 0
    threshold = 0.1
    # tester si la deviation std et la moyenne sont proche de 0
    cols_to_test = [col for col in df.columns if col != colonne_target and pd.api.types.is_numeric_dtype(df[col])]
    close_to_zero_std = (df[cols_to_test].std().sub(1).abs() < threshold).all()
    close_to_zero_mean = (df[cols_to_test].mean().abs() < threshold).all()
    
    # Si les 2 sont déjà proches de 0 
    if close_to_zero_std and close_to_zero_mean:
        st.write("Vos données semblent déjà standardisées")
    else:
        st.write("Vos données ne semblent pas standardisées")
        standard_box = st.checkbox('Standardiser', key="appli_standardisation")
        if standard_box:
            standardize_data(df, colonne_target)

def standardize_data(df, colonne_target):
    # Convertir en liste si besoin
    if isinstance(colonne_target, str):
        colonne_target = [colonne_target]
    
    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    non_standardizable_columns = [col for col in non_numeric_columns if col not in colonne_target]
    
    if non_standardizable_columns:
        st.write("Colonnes qui ne sont pas numériques et ne peuvent pas être standardisées :")
        for col in non_standardizable_columns:
            st.write(col)
            
    standardizable_columns = [
        col for col in df.columns 
        if col not in non_standardizable_columns 
        and col not in colonne_target
        and pd.api.types.is_numeric_dtype(df[col])
                              ]
    
    if standardizable_columns:
        scaler = StandardScaler()
        df[standardizable_columns] = scaler.fit_transform(df[standardizable_columns])
        st.success("✅ Les colonnes standardisables ont été standardisées avec succès.")
        # Stockage des session_states pour le résumé et pour la prédiction sur les nouvelles données
        st.session_state["standardized"] = True
        st.session_state["scaler"] = scaler
        st.session_state["standardized_columns"] = standardizable_columns
        st.session_state["standardized_stats"] = df[standardizable_columns].agg(['mean', 'std']).T.round(2)
        # Affichage du df après standardisation
        st.write("***")
        st.markdown("**Aperçu du Dataframe APRÈS Standardisation**")
        st.dataframe(df[standardizable_columns].head())
    else:
        st.warning("Aucune colonne standardisable n'a été trouvée.")
        st.session_state["standardized"] = False
    
    st.session_state["df_clean"] = df
            
            

    ### Onglet 6 - Résumé & Exports ###

def telecharger_donnees(df_clean):
    st.subheader("Télécharger les données")
    format_choisi = st.selectbox("Quel format désirez vous ?", ["CSV", "Excel (.xlsx)"])

    if format_choisi == "CSV":
        data = df_clean.to_csv(index=False).encode('utf-8')
        file_name="donnees_nettoyees.csv"
        mime="text/csv"
        
    else:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_clean.to_excel(writer, sheet_name="données nettoyées", index=False)
        excel_buffer.seek(0)
        data=excel_buffer
        file_name="donnees_nettoyees.xlsx"
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    st.success(f"✅ Fichier {file_name} prêt à être téléchargé.")
    st.download_button("📥 Télécharger les données nettoyées", data=data, file_name=file_name, mime=mime)

    
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, 'Rapport d\'Exploration de Données', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, text):
        self.set_font("Arial", '', 10)
        self.multi_cell(0, 10, text)
        self.ln()

def generer_rapport_pdf(df, df_clean, target, to_keep, task, to_drop, corr):
    pdf = PDF()
    pdf.add_page()

    tmpimg_paths = []

    # Graphique de distribution
    if pd.api.types.is_numeric_dtype(df[target]):
        fig1, ax1 = plt.subplots()
        sns.histplot(df[target].dropna(), kde=True, ax=ax1)
        ax1.set_title(f"Distribution de {target}")
        fig1.tight_layout()
        tmpimg = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig1.savefig(tmpimg.name)
        plt.close(fig1)
        pdf.image(tmpimg.name, w=180)
        tmpimg_paths.append(tmpimg.name)

    # Heatmap de corrélation
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax2)
    ax2.set_title("Matrice de corrélation")
    fig2.tight_layout()
    tmpimg2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig2.savefig(tmpimg2.name)
    plt.close(fig2)
    pdf.image(tmpimg2.name, w=180)
    tmpimg_paths.append(tmpimg2.name)

    # Bar chart d'une variable catégorielle
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        top_cat = max(cat_cols, key=lambda c: df[c].nunique())
        counts = df[top_cat].value_counts()
        fig3, ax3 = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax3)
        ax3.set_title(f"Répartition de {top_cat}")
        plt.xticks(rotation=45)
        fig3.tight_layout()
        tmpimg3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig3.savefig(tmpimg3.name)
        plt.close(fig3)
        pdf.image(tmpimg3.name, w=180)
        tmpimg_paths.append(tmpimg3.name)

    # Ajout des sections texte
    pdf.chapter_title("1. Informations générales")
    pdf.chapter_body(f"Nombre de lignes : {df_clean.shape[0]}\nNombre de colonnes : {df_clean.shape[1]}")

    pdf.chapter_title("2. Colonnes conservées")
    pdf.chapter_body(", ".join(to_keep))

    pdf.chapter_title("3. Type de tâche choisi")
    pdf.chapter_body(task)

    pdf.chapter_title("4. Colonnes très corrélées proposées à l'exclusion")
    pdf.chapter_body(", ".join(to_drop) if to_drop else "Aucune")

    pdf.chapter_title("5. Statistiques descriptives")
    desc = df_clean.describe().round(2).T
    pdf.set_font("Arial", "", 8)

    # Entêtes
    col_names = desc.columns.tolist()
    pdf.cell(30, 10, "Statistique", border=1)
    for col in col_names:
        pdf.cell(30, 10, str(col), border=1)
    pdf.ln()

    # Valeurs
    for index, row in desc.iterrows():
        pdf.cell(30, 10, str(index), border=1)
        for val in row:
            pdf.cell(30, 10, str(val), border=1)
        pdf.ln()

    # Sauvegarde du PDF
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)

    return tmp_pdf.name, tmpimg_paths
