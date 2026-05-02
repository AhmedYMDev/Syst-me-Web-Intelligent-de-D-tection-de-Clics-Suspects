# Système web intelligent de détection de clics suspects

## Objectif

Ce projet vise à détecter des clics suspects dans des campagnes publicitaires digitales et à estimer l’impact du filtrage sur le ROI marketing.

Le système combine :

- analyse KDD
- machine learning
- API FastAPI
- site web de simulation
- dashboard Streamlit
- analyse ROI

## Dataset

Dataset utilisé : TalkingData AdTracking Fraud Detection Challenge.

Colonnes principales :

- ip
- app
- device
- os
- channel
- click_time
- attributed_time
- is_attributed

## Architecture

```text
Dataset Kaggle
↓
Notebook KDD
↓
Feature engineering
↓
Modèle XGBoost
↓
API FastAPI
↓
Website de simulation
↓
Dashboard Streamlit
↓
Analyse ROI