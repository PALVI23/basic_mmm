
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List
from etl.pipeline import run_etl_pipeline
from ml.train import train_model, save_model
from ml.predict import load_model, predict_sales
import pandas as pd

app = FastAPI()

# In-memory storage for models and data
models: Dict[str, object] = {}
dataframes: Dict[str, pd.DataFrame] = {}

class SpendData(BaseModel):
    sku: str
    spend_data: Dict[str, float]

@app.post("/upload_and_process/")
async def upload_and_process(file: UploadFile = File(...)):
    """
    This endpoint receives a CSV file, runs the ETL and ML pipelines,
    and stores the trained models and data.
    """
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    df = run_etl_pipeline(temp_file_path)
    dataframes['original'] = df

    skus = df['SKU'].unique().tolist()
    trained_skus = []
    for sku in skus:
        model = train_model(df, sku)
        if model: # Check if model was trained successfully
            model_path = f"{sku}_model.pkl"
            save_model(model, model_path)
            models[sku] = load_model(model_path)
            trained_skus.append(sku)

    os.remove(temp_file_path)

    return {"trained_skus": trained_skus}

@app.post("/predict/")
async def predict(data: SpendData):
    """
    Predicts sales for a given SKU and spend data, returning a full breakdown.
    """
    model = models.get(data.sku)
    if not model:
        return {"error": "Model not found for this SKU"}
    
    prediction_breakdown = predict_sales(model, data.spend_data)
    return {"sku": data.sku, "prediction_breakdown": prediction_breakdown}

@app.get("/get_average_spend/{sku}")
async def get_average_spend(sku: str):
    """
    Returns the average spend for a given SKU.
    """
    if 'original' not in dataframes:
        return {"error": "Data not uploaded yet"}
    
    df = dataframes['original']
    sku_df = df[df['SKU'] == sku]
    if sku_df.empty:
        return {"error": "SKU not found"}
        
    avg_spend = {
        'TV_Spend': sku_df['TV_Spend'].mean(),
        'Digital_Spend': sku_df['Digital_Spend'].mean(),
        'Print_Spend': sku_df['Print_Spend'].mean(),
    }
    return {"sku": sku, "average_spend": avg_spend}

@app.get("/get_historical_data/{sku}")
async def get_historical_data(sku: str):
    """
    Returns historical sales and spend data for a given SKU.
    """
    if 'original' not in dataframes:
        return {"error": "Data not uploaded yet"}
    
    df = dataframes['original']
    sku_df = df[df['SKU'] == sku]
    if sku_df.empty:
        return {"error": "SKU not found"}
        
    # Convert to a JSON-serializable format
    return sku_df.to_dict(orient='records')

@app.get("/get_all_sku_summary/")
async def get_all_sku_summary():
    """
    Returns a summary of total sales and spend for all SKUs.
    """
    if 'original' not in dataframes:
        return {"error": "Data not uploaded yet"}
    
    df = dataframes['original']
    summary = df.groupby('SKU').agg({
        'Sales': 'sum',
        'TV_Spend': 'sum',
        'Digital_Spend': 'sum',
        'Print_Spend': 'sum'
    }).reset_index()
    
    return summary.to_dict(orient='records')
