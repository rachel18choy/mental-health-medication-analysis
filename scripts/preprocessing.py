import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import yaml
import os

def load_config():
    """Load configuration from YAML file"""
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_and_preprocess_data(save_for_unsupervised=False):
    """Load and preprocess the data"""
    config = load_config()
    
    # Load data
    df = pd.read_csv(config['data']['raw_path'])
    
    print(f"Original data shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    # Identify target column
    target_col = None
    for col in df.columns:
        if 'antipsychotics' in col.lower():
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[6]
    
    print(f"Target column: {target_col}")
    
    # Extract target variable
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Convert target to binary (0: Not helpful, 1: Helpful)
    y_binary = y.apply(lambda x: 1 if x == 1 else 0)
    
    print(f"\nTarget distribution:")
    print(f"  Helpful (1): {(y_binary == 1).sum()} samples")
    print(f"  Not helpful (0): {(y_binary == 0).sum()} samples")
    
    # Clean column names
    X.columns = [str(col).strip() for col in X.columns]
    
    # 1. Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        if col in X.columns:
            # Fill NaN with mode
            mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
            X[col] = X[col].fillna(mode_val)
            
            # Encode
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # 2. Convert all columns to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 3. Handle specific columns
    # Process study years column
    for col in X.columns:
        if 'study' in str(col).lower() and 'year' in str(col).lower():
            print(f"Processing study years column: {col}")
            
            # Map common study year formats
            study_years_mapping = {
                '0 years': 0, '0-1 years': 0.5, '1-2 years': 1.5,
                '2-3 years': 2.5, '3-4 years': 3.5, '4-5 years': 4.5,
                '5-6 years': 5.5, '6-7 years': 6.5
            }
            
            # Try to map if values are strings
            if X[col].dtype == 'object':
                X[col] = X[col].astype(str)
                X[col] = X[col].map(study_years_mapping)
    
    # 4. Impute NaN values
    print(f"\nNaN values before imputation: {X.isnull().sum().sum()}")
    
    if X.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)
        print(f"NaN values after imputation: {X.isnull().sum().sum()}")
    
    # 5. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"\nFinal feature matrix shape: {X_scaled.shape}")
    
    # 6. Split data for supervised learning
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state'],
        stratify=y_binary
    )
    
    print(f"\nData split for supervised learning:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # Create processed data directory
    os.makedirs('../data/processed', exist_ok=True)
    
    # Save data for supervised learning
    supervised_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'label_encoders': label_encoders,
        'scaler': scaler,
        'target_column': target_col
    }
    
    # Save as pickle
    import joblib
    supervised_pickle_path = '../data/processed/supervised_data.pkl'
    joblib.dump(supervised_data, supervised_pickle_path)
    
    # Save as CSV for inspection
    supervised_csv_path = '../data/processed/supervised_data.csv'
    full_supervised_data = X_scaled.copy()
    full_supervised_data[target_col] = y_binary
    full_supervised_data.to_csv(supervised_csv_path, index=False)
    
    print(f"\nSupervised learning data saved to:")
    print(f"  Pickle: {supervised_pickle_path}")
    print(f"  CSV: {supervised_csv_path}")
    
    # Save data for unsupervised learning (without target)
    if save_for_unsupervised:
        unsupervised_data = {
            'X': X_scaled,
            'feature_names': X.columns.tolist(),
            'scaler': scaler
        }
        
        unsupervised_pickle_path = '../data/processed/unsupervised_data.pkl'
        joblib.dump(unsupervised_data, unsupervised_pickle_path)
        
        unsupervised_csv_path = '../data/processed/unsupervised_data.csv'
        X_scaled.to_csv(unsupervised_csv_path, index=False)
        
        print(f"\nUnsupervised learning data saved to:")
        print(f"  Pickle: {unsupervised_pickle_path}")
        print(f"  CSV: {unsupervised_csv_path}")
    
    return supervised_data

if __name__ == "__main__":
    # Process data for both supervised and unsupervised learning
    data = load_and_preprocess_data(save_for_unsupervised=True)
    print("\nPreprocessing complete! Data is ready for both supervised and unsupervised learning.")