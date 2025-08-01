import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import shap
from flask import Flask, request, jsonify

# Step 1: Data Collection (Assuming LendingClub dataset)
data = pd.read_csv("lendingclub_data.csv")

# Step 2: Data Preprocessing
def preprocess_data(df):
    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(columns=['loan_status']))
    df_scaled = pd.DataFrame(scaled_features, columns=df.drop(columns=['loan_status']).columns)
    df_scaled['loan_status'] = df['loan_status']
    
    return df_scaled, scaler

# Step 3: Feature Engineering
def engineer_features(df):
    df['debt_to_income'] = df['total_debt'] / df['annual_income']
    df['credit_utilization'] = df['credit_balance'] / df['credit_limit']
    return df

# Step 4: Model Development
def train_model(X_train, y_train):
    model = LGBMClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# Step 5: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return auc, cm

# Step 6: SHAP Explainer
def explain_model(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values

# Step 7: Flask Deployment
app = Flask(__name__)

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict_proba(input_scaled)[:, 1]
    shap_values = explainer.shap_values(input_scaled)
    return jsonify({'risk_score': float(prediction[0]), 'shap_values': shap_values.tolist()})

# Main Pipeline
if __name__ == "__main__":
    # Preprocess and engineer features
    data_scaled, scaler = preprocess_data(data)
    data_engineered = engineer_features(data_scaled)
    
    # Split data
    X = data_engineered.drop(columns=['loan_status'])
    y = data_engineered['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    auc, cm = evaluate_model(model, X_test, y_test)
    print(f"AUC: {auc:.4f}, Confusion Matrix: \n{cm}")
    
    # Explain
    explainer = shap.TreeExplainer(model)
    shap_values = explain_model(model, X_test)
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000)
