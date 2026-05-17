import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_models(X_train, y_train):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    # XGBoost requires label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train_encoded)
    trained_models['XGBoost'] = (xgb, le)
    
    return trained_models

def evaluate_models(trained_models, X_test, y_test):
    results = {}
    for name, model_data in trained_models.items():
        if name == 'XGBoost':
            model, le = model_data
            y_test_encoded = le.transform(y_test)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test_encoded, preds)
            results[name] = acc
        else:
            preds = model_data.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc
            
    return results
