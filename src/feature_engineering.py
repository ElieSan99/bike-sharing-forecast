import pandas as pd

def add_temporal_features(df):
    """Ajoute des features temporelles de base"""
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Heures de pointe (ex: 7h-9h et 17h-19h en semaine)
    df['is_rush_hour'] = (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) | 
        ((df['hour'] >= 17) & (df['hour'] <= 19))
    ).astype(int) & (df['is_weekend'] == 0).astype(int)
    
    return df

def add_lag_features(df, target_col='demand'):
    """Ajoute des lags temporels (t-24h, t-168h)"""
    # Lag 1 jour (24h)
    df['lag_24h'] = df[target_col].shift(24)
    # Lag 1 semaine (168h)
    df['lag_168h'] = df[target_col].shift(168)
    
    # Suppression des lignes avec NaN créées par les lags
    return df.dropna()


if __name__ == "__main__":
    # Test rapide
    from pathlib import Path
    processed_path = Path("data/processed/bikeshare_aggregated.csv")
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        df = add_temporal_features(df)
        print("Features ajoutées avec succès.")
        print(df.head())
    else:
        print("Fichier agrégé non trouvé. Lancez d'abord data_loader.py.")
