import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """Calcule toutes les métriques"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = calculate_smape(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'sMAPE': smape
    }

def calculate_smape(y_true, y_pred):
    """Symmetric MAPE"""
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )

def analyze_errors_by_conditions(df_test, y_true, y_pred):
    """Analyse erreurs par conditions"""
    df_test['error'] = np.abs(y_true - y_pred)
    
    analysis = {
        'overall_mae': df_test['error'].mean(),
        'weekend_mae': df_test[df_test['is_weekend']==1]['error'].mean(),
        'weekday_mae': df_test[df_test['is_weekend']==0]['error'].mean(),
        'rush_hour_mae': df_test[df_test['is_rush_hour']==1]['error'].mean(),
        # Ajouter jours fériés, météo si disponible
    }
    
    # Top 10 pires erreurs
    worst_errors = df_test.nlargest(10, 'error')[['datetime', 'error']]
    
    return analysis, worst_errors