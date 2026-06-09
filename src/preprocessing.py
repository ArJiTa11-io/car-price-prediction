import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['Car_Age'] = 2026 - df['Year']
    df_cleaned = df.drop(columns=['Year', 'Car_Name'])
    
    # Note: added drop_first=True to avoid the dummy variable trap
    df_encoded = pd.get_dummies(df_cleaned, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)
    df_encoded = df_encoded.astype(int, errors='ignore')
    
    return df_encoded