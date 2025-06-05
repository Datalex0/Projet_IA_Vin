#
#  📈 Application Pipeline de Machine Learning 🤖

#
## 💡 Présentation de l'Application
Pipeline de Machine Learning permettant :

🧹 d'analyser et nettoyer le jeu de données fourni par l’utilisateur,

🏋️‍♂️ d’entraîner des modèles de Machine Learning adaptés au jeu de données

🔮 de faire des prédictions sur de nouvelles données

# 
## 📌 Les éléments importants :
- [Le lien vers l'application](https://pipeline-machine-learning.streamlit.app) (ctrl + clic pour l'ouvrir dans un nouvel onglet, si besoin cliquer sur le bouton pour "réveiller" l'application)
- Les scripts Python
- Le fichier vin.csv utilisé comme exemple pour le pipeline
- [La Documentation Technique](https://github.com/Datalex0/Projet_IA_Vin/blob/6195a9bd69ad0ae27a58218909459656c6c9e104/Documentation%20Technique.pdf)

#
## 🛤️ Votre parcours sur l'application
Vous serez guidé tout au long de votre parcours et passerez par les étapes suivantes :
* Page 0 - 🏠 Accueil : Présentation générale de l’application
  
* Page 1 - 📥 Import des données : 
  -	chargement CSV ou XLSX ou utilisation du dataset proposé, 
  -	suppression des colonnes inutiles
* Page 2 – 🔍 Exploration et Traitements :
  -	Choix de la cible
  -	Détection et choix du type de tâche (classification ou régression)
  -	Visualisation de la distribution des colonnes
  -	Encodage, 
  -	Analyse des corrélations et choix des colonnes à conserver
  -	Gestion des NaN,
  -	Gestion des outliers,
  -	Standardisation
  -	Export des données nettoyées en CSV ou XLSX
  -	Téléchargement d’un rapport PDF des analyses et traitements effectués
* Page 3 – 🦾 Entraînement du modèle : 
  -	Séparation du jeu de données en un jeu d’entraînement et un jeu de test,
  -	sélection automatique du meilleur modèle (LazyPredict / Cross-Validation),
  -	Entraînement et export du modèle au format pickles
  -	Optimisation des Hyperparamètres (GridSearchCV/RandomizedSearchCV) et nouvel export du modèle optimisé au format pickles
* Page 4 – 📝 Évaluation : 
  -	rapport de classification / régression
* Page 5 – 🔮 Prédictions : 
  -	Import de nouvelles données par saisie manuelle ou import CSV/XLSX
  -	Utilisation du modèle en mémoire ou import d’un autre modèle au format pickles pour effectuer des prédictions sur les nouvelles données

#
## 🚨 Points d'attention
*	Bien valider df_clean pour que l’entraînement fonctionne
*	Vérifier le bon encodage des cibles pour les prédictions
*	LabelEncoder doit être utilisé si on veut faire le décodage lors de la prédiction


