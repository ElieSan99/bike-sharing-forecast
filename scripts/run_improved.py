#!/usr/bin/env python3
"""
Script pour entraîner et évaluer le modèle amélioré LightGBM
Usage: python scripts/run_improved.py
"""

import sys
import os
from pathlib import Path
sys.path.append('src')

from data_loader import load_and_prepare_data, create_temporal_split
from feature_engineering import add_temporal_features, add_lag_features
from improved_model import LightGBMModel
from evaluation import calculate_metrics, analyze_errors_by_conditions
import json
import logging


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Improved LightGBM ===")
    
    # 1. Charger données
    logger.info("Chargement données...")
    filepath = 'data/processed/bikeshare_aggregated.csv'
    if not Path(filepath).exists():
        logger.error(f"Erreur: {filepath} n'existe pas. Lancez data_loader.py d'abord.")
        return
        
    df = load_and_prepare_data(filepath)
    
    # 2. Feature Engineering
    logger.info("Feature Engineering...")
    df = add_temporal_features(df)
    df = add_lag_features(df)
    
    # 3. Split temporel (train, val, test)
    train, val, test = create_temporal_split(df)
    logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    # 4. Entraîner
    logger.info("Entraînement LightGBM avec Early Stopping...")
    model = LightGBMModel()
    model.train(train, df_val=val)

    
    # 5. Prédire
    logger.info("Prédictions...")
    y_pred = model.predict(test)
    y_true = test['demand'].values
    
    # 6. Évaluer
    metrics = calculate_metrics(y_true, y_pred)
    logger.info("Métriques globales :")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.2f}")
    
    # Analyse d'erreurs approfondie
    logger.info("Analyse de robustesse...")
    error_analysis, worst_errors = analyze_errors_by_conditions(test, y_true, y_pred)
    logger.info("Erreurs par conditions (MAE) :")
    for k, v in error_analysis.items():
        logger.info(f"  {k}: {v:.2f}")
    
    # 7. Sauvegarder
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'improved_metrics.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'robustness': error_analysis
        }, f, indent=2)
    
    worst_errors.to_csv(results_dir / 'worst_errors.csv', index=False)
    
    logger.info(f"Résultats sauvegardés dans {results_dir}")

if __name__ == '__main__':
    main()

