from sklearn.ensemble import IsolationForest

def train_isolation_forest(X_train):
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train)
    return iso_forest

def predict_anomalies(model, X):
    preds = model.predict(X)
    return preds
