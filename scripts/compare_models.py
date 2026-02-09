#!/usr/bin/env python3
"""
Script pour comparer les performances de la baseline et du modèle amélioré
Usage: python scripts/compare_models.py
"""

import json
import pandas as pd
from pathlib import Path

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    print("=== Comparaison des Modèles ===")
    
    results = []
    
    # Charger métriques Baseline
    baseline_path = Path('results/baseline_metrics.json')
    if baseline_path.exists():
        data = load_json(baseline_path)
        metrics = data['metrics']
        metrics['Model'] = 'Seasonal Naive (Baseline)'
        results.append(metrics)
    
    # Charger métriques Improved
    improved_path = Path('results/improved_metrics.json')
    if improved_path.exists():
        data = load_json(improved_path)
        metrics = data['metrics']
        metrics['Model'] = 'LightGBM (Improved)'
        results.append(metrics)

    
    if not results:
        print("Aucun résultat trouvé. Lancez les scripts run_*.py d'abord.")
        return
        
    df_results = pd.DataFrame(results)
    # Réorganiser colonnes
    cols = ['Model', 'MAE', 'RMSE', 'sMAPE']
    df_results = df_results[cols]
    
    print("\nTableau Récapitulatif :")
    print(df_results.to_string(index=False))
    
    # Sauvegarder rapport final
    df_results.to_csv('results/model_comparison.csv', index=False)
    print("\n✓ Rapport sauvegardé dans results/model_comparison.csv")

if __name__ == '__main__':
    main()
