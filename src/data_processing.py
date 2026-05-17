import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_dataframe(df):
    df = df.copy()
    # Feature engineering: gas ratios
    df['Ratio_CH4_H2'] = df['CH4'] / df['H2'].replace(0, 1e-6)
    df['Ratio_C2H2_C2H4'] = df['C2H2'] / df['C2H4'].replace(0, 1e-6)
    df['Ratio_C2H2_CH4'] = df['C2H2'] / df['CH4'].replace(0, 1e-6)
    df['Ratio_CO2_CO'] = df['CO2'] / df['CO'].replace(0, 1e-6)
    
    features = ['H2', 'CH4', 'C2H4', 'C2H2', 'CO', 'CO2', 
                'Ratio_CH4_H2', 'Ratio_C2H2_C2H4', 'Ratio_C2H2_CH4', 'Ratio_CO2_CO']
    
    X = df[features]
    y = df['Fault_Type']
    
    return X, y, features

def load_and_preprocess_data(filepath):
    # Maintained for backward compatibility if needed, but mostly use load_and_preprocess_dataframe
    df = pd.read_csv(filepath)
    return load_and_preprocess_dataframe(df)

def prepare_training_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return prepare_explicit_train_test_data(X_train, y_train, X_test, y_test)

def prepare_explicit_train_test_data(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE only on the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler
