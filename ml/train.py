
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def apply_adstock(series, decay_rate):
    """
    Applies a geometric adstock transformation to a pandas Series.
    """
    adstocked_series = np.zeros_like(series, dtype=float)
    for i in range(len(series)):
        if i == 0:
            adstocked_series[i] = series.iloc[i]
        else:
            adstocked_series[i] = series.iloc[i] + decay_rate * adstocked_series[i-1]
    return pd.Series(adstocked_series, index=series.index)

def train_model(df, sku):
    """
    Trains a Ridge regression model for a given SKU with adstock and log-log transformation.
    """
    print(f"ML Layer: Training model for {sku}...")
    sku_df = df[df['SKU'] == sku].copy().reset_index(drop=True)

    # Adstock rates (can be customized or tuned)
    adstock_rates = {'TV_Spend': 0.5, 'Digital_Spend': 0.3, 'Print_Spend': 0.1}
    
    # Feature Engineering (Adstock)
    for channel, decay in adstock_rates.items():
        sku_df[channel] = apply_adstock(sku_df[channel], decay)

    X = sku_df[['TV_Spend', 'Digital_Spend', 'Print_Spend']]
    y = sku_df['Sales']

    # Log-transform features and target
    X_log = np.log(X + 1)
    y_log = np.log(y + 1)

    # Hyperparameter tuning for Ridge
    param_grid = {'alpha': np.logspace(-4, 4, 20)}
    
    # Dynamically set the number of CV splits to avoid errors with small datasets
    n_samples = len(sku_df)
    cv_splits = min(n_samples, 5)
    
    # Ensure cv_splits is at least 2 if there's enough data
    if n_samples < 2:
        print(f"ML Layer: Not enough data for SKU {sku} to train a model. Skipping.")
        return None

    grid_search = GridSearchCV(Ridge(), param_grid, cv=cv_splits, scoring='neg_mean_squared_error')
    grid_search.fit(X_log, y_log)

    print(f"ML Layer: Model for {sku} trained successfully. Best alpha: {grid_search.best_params_['alpha']:.2f}")
    return grid_search.best_estimator_

def save_model(model, file_path):
    """
    Saves the trained model to a file.
    """
    print(f"ML Layer: Saving model to {file_path}...")
    joblib.dump(model, file_path)
    print("ML Layer: Model saved successfully.")
