import pandas as pd


class SeasonalNaiveForecaster:
    """
    Baseline Seasonal Naive :
    y_hat(t) = y(t - seasonality)

    Exemple :
    - données horaires -> seasonality = 168 (24h * 7 jours)
    """

    def __init__(self, seasonality: int = 168):
        self.seasonality = seasonality

    def predict(self, df: pd.DataFrame, y_col: str = "demand") -> pd.Series:
        if y_col not in df.columns:
            raise ValueError(f"Colonne cible '{y_col}' absente du DataFrame")

        return df[y_col].shift(self.seasonality)
