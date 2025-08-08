
import pandas as pd
import numpy as np
import joblib

def load_model(file_path):
    """
    Loads a trained model from a file.
    """
    print(f"ML Layer: Loading model from {file_path}...")
    model = joblib.load(file_path)
    print("ML Layer: Model loaded successfully.")
    return model

def predict_sales(model, spend_data):
    """
    Predicts sales and breaks down the contribution of each channel.
    """
    print("ML Layer: Making predictions with contribution breakdown...")

    adstock_rates = {'TV_Spend': 0.5, 'Digital_Spend': 0.3, 'Print_Spend': 0.1}
    
    X_pred = pd.DataFrame([spend_data])
    
    adstocked_spend = {}
    for channel, decay_rate in adstock_rates.items():
        if channel in X_pred.columns:
            # Ensure we are using the scalar value, not the Series
            spend_value = X_pred[channel].iloc[0]
            adstocked_spend[channel] = spend_value / (1 - decay_rate) if decay_rate < 1 else spend_value

    X_adstocked = pd.DataFrame([adstocked_spend])
    X_log = np.log(X_adstocked + 1)

    # Decompose the prediction
    log_base_sales = model.intercept_
    
    channel_contributions_log = {}
    # Ensure model.feature_names_in_ exists and matches columns
    if hasattr(model, 'feature_names_in_'):
        for i, feature in enumerate(model.feature_names_in_):
            if feature in X_log.columns:
                channel_contributions_log[feature] = model.coef_[i] * X_log[feature].iloc[0]
    else: # Fallback for models without feature_names_in_
        for i, feature in enumerate(X_log.columns):
             channel_contributions_log[feature] = model.coef_[i] * X_log[feature].iloc[0]


    # Combine contributions in log space
    total_log_sales = log_base_sales + sum(channel_contributions_log.values())

    # Convert from log scale to absolute numbers
    base_sales = np.exp(log_base_sales) - 1
    total_sales = np.exp(total_log_sales) - 1
    
    channel_contributions_abs = {}
    # Calculate absolute contribution by finding the difference
    temp_base = np.exp(log_base_sales)
    for channel, log_contrib in channel_contributions_log.items():
        channel_contributions_abs[channel] = (np.exp(log_base_sales + log_contrib)) - temp_base


    prediction_breakdown = {
        "total_predicted_sales": total_sales,
        "base_sales": base_sales,
        "channel_contributions": channel_contributions_abs
    }
    
    print("ML Layer: Prediction complete.")
    return prediction_breakdown
