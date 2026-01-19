import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import streamlit as st

class DiabetesRiskPredictor:
    """
    Diabetes risk predictor based on clinical parameters.
    """
    
    def __init__(self):
        self.risk_factors = {}
        self.risk_thresholds = {
            'age': 45,
            'bmi_high': 30,
            'bmi_moderate': 25,
            'hba1c_high': 6.5,
            'hba1c_moderate': 5.7,
            'glucose_high': 140,
            'glucose_moderate': 100
        }
    
    def calculate_risk_score(self, user_data: Dict) -> Tuple[float, List[str]]:
        """
        Calculate diabetes risk score based on user input.
        
        Args:
            user_data (dict): Dictionary containing user parameters
            
        Returns:
            tuple: (risk_score, risk_factors)
        """
        risk_score = 0
        risk_factors = []
        
        # Extract user data
        age = user_data.get('age', 45)
        bmi = user_data.get('bmi', 25)
        hba1c = user_data.get('hba1c', 5.5)
        glucose = user_data.get('glucose', 100)
        hypertension = user_data.get('hypertension', 'No')
        heart_disease = user_data.get('heart_disease', 'No')
        smoking = user_data.get('smoking', 'never')
        gender = user_data.get('gender', 'Female')
        
        # Age risk
        if age > self.risk_thresholds['age']:
            risk_score += 1
            risk_factors.append(f"Age > {self.risk_thresholds['age']} ({age})")
        
        # BMI risk
        if bmi > self.risk_thresholds['bmi_high']:
            risk_score += 1
            risk_factors.append(f"BMI > {self.risk_thresholds['bmi_high']} ({bmi:.1f})")
        elif bmi > self.risk_thresholds['bmi_moderate']:
            risk_score += 0.5
            risk_factors.append(f"BMI {self.risk_thresholds['bmi_moderate']}-{self.risk_thresholds['bmi_high']} ({bmi:.1f})")
        
        # HbA1c risk
        if hba1c > self.risk_thresholds['hba1c_high']:
            risk_score += 2
            risk_factors.append(f"HbA1c > {self.risk_thresholds['hba1c_high']} ({hba1c})")
        elif hba1c > self.risk_thresholds['hba1c_moderate']:
            risk_score += 1
            risk_factors.append(f"HbA1c {self.risk_thresholds['hba1c_moderate']}-{self.risk_thresholds['hba1c_high']} ({hba1c})")
        
        # Glucose risk
        if glucose > self.risk_thresholds['glucose_high']:
            risk_score += 2
            risk_factors.append(f"Glucose > {self.risk_thresholds['glucose_high']} ({glucose})")
        elif glucose > self.risk_thresholds['glucose_moderate']:
            risk_score += 1
            risk_factors.append(f"Glucose {self.risk_thresholds['glucose_moderate']}-{self.risk_thresholds['glucose_high']} ({glucose})")
        
        # Hypertension risk
        if hypertension == "Yes":
            risk_score += 1
            risk_factors.append("Hypertension")
        
        # Heart disease risk
        if heart_disease == "Yes":
            risk_score += 1
            risk_factors.append("Heart Disease")
        
        # Smoking risk
        if smoking in ["current", "former"]:
            risk_score += 0.5
            risk_factors.append(f"Smoking: {smoking}")
        
        # Gender risk (simplified - males have slightly higher risk)
        if gender == "Male":
            risk_score += 0.5
            risk_factors.append("Male gender")
        
        return risk_score, risk_factors
    
    def get_risk_level(self, risk_percentage: float) -> Tuple[str, str, str]:
        """
        Determine risk level based on risk percentage.
        
        Args:
            risk_percentage (float): Calculated risk percentage
            
        Returns:
            tuple: (risk_level, color, emoji)
        """
        if risk_percentage < 20:
            return "Low Risk", "#10B981", ""
        elif risk_percentage < 50:
            return "Moderate Risk", "#F59E0B", ""
        elif risk_percentage < 75:
            return "High Risk", "#EF4444", ""
        else:
            return "Very High Risk", "#DC2626", ""
    
    def get_recommendations(self, user_data: Dict) -> List[str]:
        """
        Generate personalized recommendations based on user data.
        
        Args:
            user_data (dict): Dictionary containing user parameters
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        bmi = user_data.get('bmi', 25)
        hba1c = user_data.get('hba1c', 5.5)
        glucose = user_data.get('glucose', 100)
        hypertension = user_data.get('hypertension', 'No')
        smoking = user_data.get('smoking', 'never')
        
        if bmi > 25:
            recommendations.append("Consider weight management through diet and exercise")
        
        if hba1c > 5.7:
            recommendations.append("Monitor HbA1c levels regularly with healthcare provider")
        
        if glucose > 100:
            recommendations.append("Regular blood glucose monitoring recommended")
        
        if hypertension == "Yes":
            recommendations.append("Manage hypertension with medical guidance and medication")
        
        if smoking in ["current", "former"]:
            recommendations.append("Consider smoking cessation programs and support")
        
        if not recommendations:
            recommendations.append("Maintain current healthy lifestyle habits")
        
        return recommendations