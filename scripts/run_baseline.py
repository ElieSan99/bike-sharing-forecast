#!/usr/bin/env python3
"""
Script pour entraîner et évaluer le modèle baseline Prophet
Usage: python scripts/run_baseline.py
"""

import sys
sys.path.append('src')

from data_loader import load_and_prepare_data, create_temporal_split
from feature_engineering import add_temporal_features
from baseline import SeasonalNaiveForecaster

from evaluation import calculate_metrics, analyze_errors_by_conditions
from pathlib import Path
import json
import logging


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Baseline Seasonal Naive ===")
    
    # 1. Charger données
    logger.info("Chargement données...")
    df = load_and_prepare_data('data/processed/bikeshare_aggregated.csv')
    
    # 2. Feature Engineering
    logger.info("Feature Engineering...")
    df = add_temporal_features(df)
    
    # 3. Split temporel (train, val, test)
    # On utilise les dates par défaut définies dans data_loader.py
    train, val, test = create_temporal_split(df)


    logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    # 3. Prédictions
    logger.info("Application du Seasonal Naive (Lag 168h)...")
    model = SeasonalNaiveForecaster(seasonality=168)
    
    # On prédit sur le test set
    # Note: comme c'est un Naive, il n'y a pas d'entraînement complexe
    y_pred_all = model.predict(test)
    
    # On enlève les NaNs initiaux dus au shift pour le calcul des métriques
    mask = y_pred_all.notna()
    y_pred = y_pred_all[mask]
    y_true = test.loc[mask, 'demand']
    df_eval = test.loc[mask].copy()
    
    # 4. Évaluer
    metrics = calculate_metrics(y_true.values, y_pred.values)
    logger.info("Métriques globales (Test) :")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.2f}")
    
    # Analyse de robustesse
    logger.info("Analyse de robustesse...")
    error_analysis, worst_errors = analyze_errors_by_conditions(df_eval, y_true.values, y_pred.values)
    
    # 5. Sauvegarder
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'baseline_metrics.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'robustness': error_analysis
        }, f, indent=2)
    
    logger.info(f"Résultats sauvegardés dans {results_dir}")

if __name__ == '__main__':
    main()
