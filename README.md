# Bike Sharing Forecast Project - POC (Capital Bikeshare)

Projet de prédiction de la demande horaire de vélos en libre-service à horizon **48 heures (J+2)**.

## Structure du Projet

```text
bike-sharing-forecast/
├── README.md                    # Instructions d'exécution
├── NOTE_METHODE.md              # Note méthodologique (POV)
├── requirements.txt             # Dépendances Python
│
├── notebooks/                   # Analyse exploratoire
│   └── exploration.ipynb        # Visualisation et feature discovery
│
├── data/                        # Données (non versionné)
│   ├── raw/                     # Téléchargements des données brutes
│   │   ├── zips/                # Données Zip
│   │   └── csv/                 # Données CSV
│   └── processed/               # CSV agrégé
│
├── src/                         # Code source
│   ├── data_loader.py           # Orchestration du téléchargement
│   ├── feature_engineering.py   # Création des Lags et Features temporelles
│   ├── baseline.py              # Modèle Seasonal Naive (Baseline)
│   ├── improved_model.py        # Modèle LightGBM (Improved)
│   ├── evaluation.py            # Calcul MAE, RMSE, sMAPE
│   └── utils.py                 # Utilitaires
│
├── scripts/                     # Scripts d'exécution
│   ├── run_baseline.py          # Exécution Baseline (Naive)
│   ├── run_improved.py          # Entraînement LightGBM
│   └── compare_models.py        # Rapport de performance
│
└── results/                     # Sorties
    ├── metrics.json             # Métriques par modèle
    └── model_comparison.csv     # Tableau final
```

## Installation

### 1. Création de l'environnement virtuel
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Installations des dépendances
```bash
pip install -r requirements.txt
```

> [!TIP]
> **Sur Linux (Ubuntu/Debian)** : Si vous rencontrez l'erreur `libgomp.so.1`, installez la dépendance système avec :  
> `sudo apt-get update && sudo apt-get install -y libgomp1`

## Pipeline de Reproduction

### 1. Préparation des données
Télécharge et agrège les données 2024-2025.
```bash
python src/data_loader.py
```

### 2. Exécuter la Baseline (Seasonal Naive)
Calcul des prédictions basées sur J-7.
```bash
python scripts/run_baseline.py
```

### 3. Entraîner le Modèle Amélioré (LightGBM)
```bash
python scripts/run_improved.py
```

### 4. Comparer les résultats
Génère un tableau comparatif dans `results/model_comparison.csv`.
```bash
python scripts/compare_models.py
```

## Performances Résumées (Sample Test > 2025-11-01)

| Modèle | MAE | RMSE | sMAPE |
| :--- | :--- | :--- | :--- |
| **Seasonal Naive** | 156.35 | 262.51 | 35.57% |
| **LightGBM (Improved)**| **116.32** | **184.30** | **29.08%** |




