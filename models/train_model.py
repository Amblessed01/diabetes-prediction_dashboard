"""
Script to train machine learning models for diabetes prediction.
This can be run separately to train and save models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def prepare_data(thankgod_israel):
    """
    Prepare data for machine learning.
    """
    # Create a copy to avoid modifying original
    data = thankgod_israel.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['gender', 'smoking_history']
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Separate features and target
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }

def save_model(model, scaler, label_encoders, model_dir='models/saved_models'):
    """
    Save trained model and preprocessing objects.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'random_forest_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.joblib'))
    
    print(f"Model saved to {model_dir}")

if __name__ == "__main__":
    # Load data
    file_path = r"C:\Users\User\Desktop\OSIRI UNIVERSITY Files\diabetes_prediction_dashboard\diabetes_prediction_dataset.csv"
    thankgod_israel = pd.read_csv(file_path)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoders = prepare_data(thankgod_israel)
    
    # Train model
    print("Training Random Forest model...")
    model = train_random_forest(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(results['report']).transpose())
    
    # Save model
    save_model(model, scaler, label_encoders)
    print("Model training complete!")