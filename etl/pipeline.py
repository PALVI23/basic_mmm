
import pandas as pd
import os

def load_data(file_path):
    """
    Loads data from a CSV or Excel file, detecting the type by extension.
    """
    print(f"ETL Stage: Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
        elif file_extension == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        print("ETL Stage: Data loaded successfully.")
        return df
    except Exception as e:
        print(f"ETL Stage: Error loading data file. {e}")
        raise

def preprocess_data(df):
    """
    Performs basic preprocessing on the data.
    """
    print("ETL Stage: Preprocessing data...")
    # Clean up column names by stripping leading/trailing whitespace
    df.columns = df.columns.str.strip()
    print(f"ETL Stage: Cleaned column headers: {df.columns.tolist()}")
    
    # Example preprocessing steps:
    # - Handling missing values
    # - Feature engineering
    # - etc.
    print("ETL Stage: Data preprocessing complete.")
    return df

def run_etl_pipeline(file_path):
    """
    Runs the full ETL pipeline.
    """
    print("ETL Pipeline Started.")
    df = load_data(file_path)
    df_processed = preprocess_data(df)
    print("ETL Pipeline Finished.")
    return df_processed
