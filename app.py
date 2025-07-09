# === IMPORTATIONS ===
import pandas as pd
import numpy as np
import time
import pickle
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import f1_score, roc_auc_score
import plotly.express as px
import plotly.figure_factory as ff
import os

# === CONFIGURATION DE LA PAGE ===
st.set_page_config(page_title="Détection de Fraude", layout="wide")

# === SIDEBAR ===
with st.sidebar:
    st.title("🛡️ Projet IA & Cybersécurité")
    st.markdown("""
    **Auteur** : Salma Abakil  
    **Projet** : Détection de Fraude par Carte de Crédit  
    **Module** : Intelligence Artificielle et Cybersécurité
    """)
    st.markdown("---")
    st.info("Développé avec Streamlit")

# === FONCTION POUR CHARGER LES DONNÉES ===
@st.cache_data
def load_data():
    df = pd.read_csv("c:/Users/HP/Documents/Docker-projet-ai/creditcard.csv")
    return df

df = load_data()

# === RÉSUMÉ RAPIDE ===
st.title("💳 Détection de Fraudes par Carte de Crédit")
st.write("""
Bienvenue sur l'application de détection de fraudes.  
Choisissez un modèle, explorez ses performances, comparez les résultats, 
et testez vos propres fichiers de transactions !
""")

# === EXPLORATION INTERACTIVE DES DONNÉES ===
st.markdown("### 📊 Exploration Interactive des Données")

# Répartition des classes
st.markdown("#### Répartition des classes (0 = Non Fraude, 1 = Fraude)")
fig_class = px.histogram(df, x='Class', color='Class',
                         labels={'Class': 'Classe'},
                         color_discrete_map={0: 'lightblue', 1: 'salmon'},
                         title="Nombre de transactions par classe")
st.plotly_chart(fig_class, use_container_width=True)

# Heatmap de corrélation
st.markdown("#### Matrice de corrélation (échantillon de 1000 lignes)")
sample_corr = df.sample(n=1000, random_state=42)
corr = sample_corr.corr()
z = corr.values
x = y = corr.columns.tolist()
fig_corr = ff.create_annotated_heatmap(
    z=z, x=x, y=y,
    annotation_text=[[f"{val:.2f}" for val in row] for row in z],
    colorscale='Viridis', showscale=True
)
st.plotly_chart(fig_corr, use_container_width=True)


# === PRÉTRAITEMENT ===
df = df.sample(frac=0.2, random_state=42)
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
df.drop(['Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# === APPLICATION DE SMOTE ===
X_smote_sample, _, y_smote_sample, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42, stratify=y_train)
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_smote_sample, y_smote_sample)

# === DICTIONNAIRE DES MODÈLES ===
models = {
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# === SÉLECTION DU MODÈLE ===
model_choice = st.selectbox("🧠 Choisissez un modèle :", ["XGBoost", "Random Forest", "Logistic Regression"])

# === FONCTION D'ÉVALUATION DU MODÈLE ===
def evaluate_model(model_name):
    model = models[model_name]
    model.fit(X_resampled, y_resampled)

    start_time = time.time()
    y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time

    st.success(f"⏱️ Temps de prédiction : {elapsed_time:.4f} secondes")

    # Rapport de classification
    st.subheader("📋 Rapport de Classification")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.style.format(precision=2))

    # AUC-ROC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
        st.info(f"📈 AUC-ROC : {auc_score:.4f}")

    # F1-score (Validation Croisée)
    scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='f1')
    st.info(f"✅ F1-score moyen (CV 5-fold) : {scores.mean():.4f}")

    # Matrice de confusion
    st.subheader("🔍 Matrice de Confusion")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"Matrice - {model_name}", fontsize=10)
    ax.set_xlabel("Prédit", fontsize=9)
    ax.set_ylabel("Réel", fontsize=9)
    st.pyplot(fig)

    # Courbe ROC
    st.subheader("📊 Courbe ROC")
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})", color='blue')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlabel("Taux de Faux Positifs")
        ax.set_ylabel("Taux de Vrais Positifs")
        ax.set_title("Courbe ROC")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# === FONCTION DE COMPARAISON DES MODÈLES ===
def compare_models():
    best_model_name = ""
    best_f1_score = 0
    best_auc = 0
    best_model = None

    st.subheader("🔎 Résultats comparatifs des modèles (Validation croisée 5-fold)")

    for name, model in models.items():
        # Validation croisée F1-score sur les données réséchantillonnées
        f1_scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='f1')
        mean_f1 = f1_scores.mean()

        # Entraîner le modèle sur tout le jeu rééchantillonné pour calculer l’AUC
        model.fit(X_resampled, y_resampled)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_proba)
        else:
            auc_score = 0.0

        st.write(f"- **{name}** — F1-score moyen (CV) : `{mean_f1:.4f}` | AUC-ROC sur test : `{auc_score:.4f}`")

        if mean_f1 > best_f1_score or (auc_score > best_auc and mean_f1 >= best_f1_score):
            best_model_name = name
            best_f1_score = mean_f1
            best_auc = auc_score
            best_model = model

    st.success(f"🥇 Meilleur modèle (selon F1-CV) : **{best_model_name}**")
    st.write(f"📈 AUC-ROC sur test : `{best_auc:.4f}`")
    st.write(f"✅ F1-score moyen (validation croisée) : `{best_f1_score:.4f}`")

    # Sauvegarder le meilleur modèle
    if best_model:
        with open("best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        st.info("Meilleur modèle sauvegardé sous `best_model.pkl`")

# === BOUTON POUR COMPARER ===
with st.expander("📊 Comparer tous les modèles", expanded=False):
    if st.button("Lancer la comparaison"):
        with st.spinner("Comparaison en cours..."):
            compare_models()

# === AFFICHAGE DES RÉSULTATS DU MODÈLE SÉLECTIONNÉ ===
if model_choice:
    evaluate_model(model_choice)

# === UPLOAD ET TEST DE FICHIERS CSV ===
st.subheader("📂 Tester un fichier CSV de transactions")

uploaded_file = st.file_uploader("Uploader votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        file_size = uploaded_file.size
        if file_size > 5 * 1024 * 1024:  # Limite de 5 Mo
            st.error("❌ Le fichier est trop volumineux (max 5 Mo).")
        else:
            test_data = pd.read_csv(uploaded_file)

            # Vérifier les colonnes
            missing_cols = [col for col in X.columns if col not in test_data.columns]
            if missing_cols:
                st.error(f"❌ Colonnes manquantes : {missing_cols}")
            else:
                st.success("✅ Fichier valide. Prédiction en cours...")

                # Charger meilleur modèle ou celui sélectionné
                if os.path.exists("best_model.pkl"):
                    with open("best_model.pkl", "rb") as f:
                        model = pickle.load(f)
                    st.info("✅ Utilisation du meilleur modèle sauvegardé (best_model.pkl).")
                else:
                    model = models[model_choice]
                    model.fit(X_resampled, y_resampled)
                    st.info("ℹ️ Utilisation du modèle sélectionné dans l’interface.")

                predictions = model.predict(test_data)

                # Ajouter prédictions
                test_data['Prediction'] = predictions

                # Afficher résultats
                st.subheader("📄 Résultats de prédiction")
                st.dataframe(test_data)

                # Résumé des fraudes
                num_frauds = np.sum(predictions)
                st.warning(f"Transactions frauduleuses détectées : {num_frauds}")

                # Bouton de téléchargement
                csv = test_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les résultats",
                    data=csv,
                    file_name='predictions_fraude.csv',
                    mime='text/csv'
                )

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
