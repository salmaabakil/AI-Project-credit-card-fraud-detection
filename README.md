# 💳 AI Project: Détection de Fraude sur Cartes de Crédit

Ce projet propose une application interactive basée sur **Streamlit**, permettant de détecter des fraudes sur des transactions par carte bancaire à l'aide de modèles d'apprentissage automatique.

## 🚀 Présentation

Les fraudes par carte de crédit représentent une menace majeure pour les institutions financières et les consommateurs. Ce projet vise à développer un outil de détection performant et explicable, tout en permettant une exploration interactive des données et des performances des modèles.

## 🧰 Technologies utilisées

- **Python (pandas, numpy, scikit-learn, XGBoost, Random Forest, Logistic Regression)**
- **Streamlit** pour l'interface web interactive
- **SMOTE** pour le rééquilibrage des classes
- **Plotly & Seaborn** pour la visualisation des données
- **Docker** (optionnel) pour l'encapsulation

## 🗂️ Fonctionnalités principales

- 📊 **Exploration interactive des données** (histogrammes, matrices de corrélation, etc.)
- 🧠 **Sélection et comparaison de modèles** :
  - XGBoost
  - Random Forest
  - Régression Logistique
- ✅ **Évaluation détaillée** :
  - Classification report
  - F1-score (validation croisée)
  - AUC-ROC
  - Matrice de confusion
  - Courbe ROC
- 📂 **Upload de fichier CSV personnalisé** pour tester vos propres transactions
- 💾 Sauvegarde automatique du **meilleur modèle** (`best_model.pkl`)

## ⚙️ Prétraitement

- Normalisation des montants
- Suppression de la colonne "Time"
- Utilisation de **SMOTE** pour traiter le déséquilibre des classes
- Échantillonnage de 20 % des données pour accélérer la démo

## 🖥️ Utilisation

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/salmaabakil/AI-Project-credit-card-fraud-detection.git
cd AI-Project-credit-card-fraud-detection
