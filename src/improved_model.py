import lightgbm as lgb

class LightGBMModel:
    def __init__(self, features=None):
        self.model = None
        self.features = features or [
            "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
            "lag_24h", "lag_168h"
        ]

    def train(self, df_train, df_val=None):
        """Entraîne LightGBM avec validation temporelle + early stopping."""
        X_train = df_train[self.features]
        y_train = df_train["demand"]

        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": 42,
        }

        train_data = lgb.Dataset(X_train, label=y_train)

        if df_val is not None:
            X_val = df_val[self.features]
            y_val = df_val["demand"]
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=2000,
                valid_sets=[train_data, val_data],
                valid_names=["train", "val"],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
            )
        else:
            # fallback sans validation (moins recommandé)
            self.model = lgb.train(params, train_data, num_boost_round=500)

        return self

    def predict(self, df_test):
        X_test = df_test[self.features]
        return self.model.predict(X_test)
