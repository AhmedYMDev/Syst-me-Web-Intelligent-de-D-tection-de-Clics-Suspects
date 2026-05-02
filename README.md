# Système Web Intelligent de Détection de Clics Suspects

## Description du projet

Ce projet est un système intelligent de détection de clics suspects dans un contexte web/e-commerce. Il applique le processus complet **KDD — Knowledge Discovery in Databases** afin d'extraire des connaissances exploitables à partir de données comportementales liées aux clics utilisateurs.

L'objectif principal est d'identifier les comportements anormaux pouvant indiquer une fraude au clic, un trafic artificiel ou une activité suspecte. Le projet combine plusieurs approches de data mining et de machine learning :

* **Isolation Forest** pour la détection non supervisée d'anomalies.
* **DBSCAN** pour l'identification de groupes de comportements similaires et de points bruités.
* **XGBoost** pour la classification supervisée des clics suspects.
* Une comparaison finale des performances des différentes approches.
* Une analyse métier permettant d'interpréter les résultats et leur impact potentiel.

Le projet inclut également une architecture modulaire avec notebooks, scripts Python, modèles sauvegardés, API temps réel, dashboard et génération de résultats.

---

## Objectifs

Les objectifs du projet sont :

1. Respecter toutes les étapes du processus **KDD**.
2. Nettoyer, transformer et enrichir les données de clics.
3. Détecter les comportements suspects à l'aide d'algorithmes adaptés.
4. Comparer plusieurs approches de machine learning.
5. Fournir une interprétation claire des résultats obtenus.
6. Préparer une base exploitable pour une intégration dans un système web.
7. Visualiser les résultats via un dashboard interactif.

---

## Processus KDD appliqué

Le projet suit les cinq grandes étapes du processus KDD :

### 1. Sélection des données

Les données utilisées proviennent d'une source open access, notamment Kaggle ou une plateforme équivalente. Elles représentent des événements utilisateurs tels que des clics, des impressions, des conversions ou des interactions web.

### 2. Prétraitement

Cette étape comprend :

* Chargement des données.
* Analyse des colonnes disponibles.
* Gestion des valeurs manquantes.
* Suppression ou traitement des doublons.
* Vérification des types de données.
* Nettoyage des incohérences.

### 3. Transformation

Les données sont transformées afin de créer des variables exploitables par les modèles :

* Extraction des variables temporelles.
* Encodage des variables catégorielles.
* Normalisation ou standardisation si nécessaire.
* Construction de variables comportementales.
* Préparation des jeux d'entraînement et de test.

### 4. Data Mining

Trois approches principales sont utilisées :

#### Isolation Forest

Isolation Forest est utilisée comme méthode non supervisée pour détecter les observations anormales. Elle est adaptée lorsque les comportements frauduleux sont rares par rapport au trafic normal.

#### DBSCAN

DBSCAN permet de regrouper les comportements similaires et d'identifier les points isolés comme potentiellement suspects. Cette méthode est utile pour découvrir des structures cachées dans les données sans utiliser directement une variable cible.

#### XGBoost

XGBoost est utilisé comme modèle supervisé pour prédire si un clic est suspect ou non. Il est performant sur les données tabulaires et permet d'obtenir une évaluation précise à travers plusieurs métriques.

### 5. Interprétation et évaluation

Les résultats sont évalués à l'aide de métriques adaptées :

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Matrice de confusion
* Comparaison entre modèles
* Analyse des variables importantes

L'interprétation ne se limite pas aux scores. Elle vise aussi à comprendre les comportements détectés comme suspects.

---

## Structure du projet

```text
project_data_mining/
│
├── data/
│   └── click_fraud_dataset.csv
│
├── notebooks/
│   └── 01_kdd_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── realtime_api.py
│   ├── roi_analysis.py
│   └── database.py
│
├── web/
│   └── website_simulation/
│
├── dashboard/
│   └── streamlit_dashboard.py
│
├── models/
│   └── fraud_model.pkl
│
├── outputs/
│   ├── figures/
│   ├── fraud_scores.csv
│   ├── roi_report.csv
│   └── final_report.pdf
│
├── LICENSE
└── README.md
```

---

## Technologies utilisées

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit
* FastAPI ou Flask
* Jupyter Notebook
* Git et GitHub

---

## Installation

### 1. Cloner le repository

```bash
git clone https://github.com/AhmedYMDev/Syst-me-Web-Intelligent-de-D-tection-de-Clics-Suspects.git
```

### 2. Accéder au dossier du projet

```bash
cd Syst-me-Web-Intelligent-de-D-tection-de-Clics-Suspects
```

### 3. Créer un environnement virtuel

```bash
python3 -m venv venv
```

### 4. Activer l'environnement virtuel

Sur macOS ou Linux :

```bash
source venv/bin/activate
```

Sur Windows :

```bash
venv\Scripts\activate
```

### 5. Installer les dépendances

```bash
pip install -r requirements.txt
```

Si le fichier `requirements.txt` n'existe pas encore, les dépendances principales peuvent être installées avec :

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit fastapi uvicorn jupyter
```

---

## Utilisation

### Lancer le notebook KDD

```bash
jupyter notebook notebooks/01_kdd_analysis.ipynb
```

Le notebook contient les étapes principales :

* Analyse exploratoire des données.
* Nettoyage.
* Feature engineering.
* Entraînement des modèles.
* Évaluation.
* Comparaison finale.

---

### Entraîner le modèle

```bash
python src/train_model.py
```

Cette commande entraîne les modèles et sauvegarde le meilleur modèle dans le dossier :

```text
models/
```

---

### Lancer l'API temps réel

Si le projet utilise FastAPI :

```bash
uvicorn src.realtime_api:app --reload
```

L'API sera disponible à l'adresse :

```text
http://127.0.0.1:8000
```

---

### Lancer le dashboard Streamlit

```bash
streamlit run dashboard/streamlit_dashboard.py
```

Le dashboard permet de visualiser :

* Les clics normaux et suspects.
* Les scores de fraude.
* Les métriques des modèles.
* Les graphiques d'analyse.
* Les résultats métier.

---

## Modèles utilisés

### Isolation Forest

Modèle non supervisé utilisé pour isoler les observations rares. Les clics qui s'écartent fortement du comportement normal sont considérés comme suspects.

### DBSCAN

Algorithme de clustering basé sur la densité. Les points qui ne font partie d'aucun cluster dense sont considérés comme bruit ou anomalies.

### XGBoost

Modèle supervisé performant pour la classification binaire. Il est utilisé pour prédire la probabilité qu'un clic soit frauduleux ou suspect.

---

## Évaluation des performances

Les modèles sont comparés à l'aide de plusieurs métriques :

| Modèle           | Type          | Objectif                          |
| ---------------- | ------------- | --------------------------------- |
| Isolation Forest | Non supervisé | Détection d'anomalies             |
| DBSCAN           | Non supervisé | Clustering et détection de bruit  |
| XGBoost          | Supervisé     | Classification des clics suspects |

Les métriques utilisées incluent :

* Precision
* Recall
* F1-score
* ROC-AUC
* Matrice de confusion
* Courbe ROC
* Analyse des faux positifs et faux négatifs

---

## Résultats attendus

Le système doit produire :

* Un score de suspicion pour chaque clic.
* Une classification entre clic normal et clic suspect.
* Des visualisations des comportements anormaux.
* Une comparaison claire entre les modèles.
* Un rapport final contenant les résultats et l'interprétation.

Les fichiers de sortie sont stockés dans :

```text
outputs/
```

---

## Analyse métier

La détection des clics suspects peut aider à :

* Réduire les pertes publicitaires.
* Identifier les sources de trafic anormal.
* Améliorer la qualité des campagnes marketing.
* Protéger les annonceurs contre la fraude au clic.
* Optimiser le retour sur investissement publicitaire.

---

## Limites du projet

Certaines limites peuvent exister selon le dataset utilisé :

* Absence possible d'adresse IP réelle.
* Données anonymisées.
* Déséquilibre important entre clics normaux et clics frauduleux.
* Difficulté à valider certaines anomalies sans contexte métier réel.
* Résultats dépendants de la qualité des variables disponibles.

Ces limites sont prises en compte dans l'interprétation finale.

---

## Améliorations futures

Les améliorations possibles incluent :

* Ajouter des données réseau comme l'adresse IP, le pays ou le navigateur.
* Intégrer un système de monitoring en temps réel.
* Ajouter une base de données pour stocker les prédictions.
* Améliorer l'explicabilité avec SHAP ou LIME.
* Déployer l'API sur un serveur cloud.
* Ajouter une interface web complète pour tester le système.


## Licence

Ce projet est distribué sous licence **MIT**.

La licence MIT permet l'utilisation, la modification et la distribution du code, sous réserve de conserver la notice de copyright et la licence originale.

---

## Auteur

**AhmedYMDev**
Ma Binôme **Manalnafea**

Projet réalisé dans le cadre d'un travail académique en Data Mining.

---

## Statut du projet

Projet en cours de développement.

Les principales fonctionnalités prévues sont :

* Analyse KDD complète.
* Entraînement des modèles.
* Évaluation comparative.
* Dashboard Streamlit.
* API de prédiction temps réel.
* Rapport final.


