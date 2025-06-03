import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import tempfile
from io import BytesIO

def select_distribution(df):
   ''' Fonction permettant d'afficher la distributions des variables en sélectionnant 
   celles que l'on souhaite afficher '''
   colonnes = ["alcohol","malic_acid","ash","alcalinity_of_ash","magnesium","total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins","color_intensity","hue","od280/od315_of_diluted_wines","proline"]
   selected_cols = st.multiselect("Sélectionnez les colonnes à afficher :", colonnes, default=["alcohol"], key="col_selector_1")
   for col in selected_cols:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(df[col], kde=True, color='skyblue')
    ax.set_title(f"Distribution de {col}")
    st.pyplot(fig)


def select_pairplot(df):
   ''' Fonction permettant d'afficher les pairplots des variables en sélectionnant 
   celles que l'on souhaite afficher '''
   pairplot_cols = ["alcohol","malic_acid","ash","alcalinity_of_ash","magnesium","total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins","color_intensity","hue","od280/od315_of_diluted_wines","proline", "target"]
   selected_cols_pairplot = st.multiselect("Sélectionnez les colonnes à afficher :", pairplot_cols, default=["target", "alcohol"], key="col_selector_2")
   for col in selected_cols_pairplot:
    fig, ax = plt.subplots(figsize=(5, 5))
    fig = sns.pairplot(df[selected_cols_pairplot], hue="target" if "target" in selected_cols_pairplot else None)
    ax.set_title(f"Distribution de {col}")
    st.pyplot(fig)



### Onglet 4 ###

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

        
### Onglet 5 ###

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
    
    st.success(f"✅ Fichier {file_name} prêt à être téléchargé avec succès.")
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
        tmpimg = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig1.savefig(tmpimg.name)
        plt.close(fig1)
        pdf.image(tmpimg.name, w=180)
        tmpimg_paths.append(tmpimg.name)

    # Heatmap de corrélation
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax2)
    ax2.set_title("Matrice de corrélation")
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
