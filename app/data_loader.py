import pandas as pd

def load_data():
    df = pd.read_excel("data/raw/raw_mental_health_data.xlsx")
    df = df.where(pd.notnull(df), None)
    return df

