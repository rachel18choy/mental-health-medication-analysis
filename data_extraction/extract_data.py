import requests
import pandas as pd
import json
from datetime import datetime
import os

def extract_api_data(api_url="http://localhost:8000/data", save_to_file=True):
    """
    Extract data from the FastAPI endpoint and optionally save to file
    """
    try:
        # Make GET request to the API
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Convert response to DataFrame
        data = response.json()
        df = pd.DataFrame(data)
        
        print(f"Successfully extracted {len(df)} records with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        # Save to CSV and JSON for model training
        if save_to_file:
            # Create data directory if it doesn't exist
            os.makedirs('../data', exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save to CSV
            csv_filename = f"../data/api_extracted_data_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Data saved to: {csv_filename}")
            
            # Save to JSON
            json_filename = f"../data/api_extracted_data_{timestamp}.json"
            df.to_json(json_filename, orient='records', indent=2)
            print(f"Data saved to: {json_filename}")
            
            # Also save a clean version without null placeholders
            clean_df = df.replace({None: pd.NA})
            clean_csv = f"../data/api_extracted_data_clean_{timestamp}.csv"
            clean_df.to_csv(clean_csv, index=False)
            print(f"Clean data saved to: {clean_csv}")
        
        return df
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure your FastAPI server is running.")
        print("Run this command first: uvicorn app.main:app --reload")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def data_summary(df):
    """
    Generate summary statistics for the extracted data
    """
    if df is not None:
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Total records: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        
        print("\nColumn information:")
        for col in df.columns:
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            unique_count = df[col].nunique()
            print(f"  {col}: {non_null} non-null, {null_count} null, {unique_count} unique values")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nData types:")
        print(df.dtypes)

def explore_data_for_modeling(df):
    """
    Explore data specifically for classification modeling
    """
    if df is not None:
        print("\n" + "="*50)
        print("MODELING PREPARATION ANALYSIS")
        print("="*50)
        
        # Identify potential target columns (columns that might be used for classification)
        print("\nPotential target columns for classification:")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 10:  # Reasonable number of classes for classification
                print(f"  {col}: {len(unique_values)} unique values - {list(unique_values[:5])}")
        
        # Check for missing values
        print("\nMissing values per column:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric columns ({len(numeric_cols)}): {list(numeric_cols)}")
            print("Summary statistics:")
            print(df[numeric_cols].describe())

if __name__ == "__main__":
    # Extract data from API
    print("Extracting data from FastAPI...")
    df = extract_api_data()
    
    # Generate summaries
    if df is not None:
        data_summary(df)
        explore_data_for_modeling(df)
        
        print("\nNext steps for model training:")
        print("1. Review the data summaries above")
        print("2. Identify your target variable for classification")
        print("3. Create feature engineering script")
        print("4. Split data into train/test sets")
        print("5. Train your classification model")