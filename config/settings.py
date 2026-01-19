"""
Configuration settings for the Diabetes Prediction Dashboard.
"""

# File paths
DATA_PATH = r"C:\Users\User\Desktop\OSIRI UNIVERSITY Files\diabetes_prediction_dashboard\diabetes_prediction_dataset.csv"
BACKUP_DATA_PATH = "data/diabetes_prediction_dataset.csv"

# App settings
APP_TITLE = "Diabetes Prediction Dashboard"
APP_ICON = ""
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Color scheme
COLORS = {
    'primary': '#3B82F6',
    'secondary': '#1E40AF',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'info': '#0EA5E9',
    'light': '#EFF6FF',
    'dark': '#1E3A8A'
}

# Risk thresholds (can be overridden by user)
DEFAULT_RISK_THRESHOLDS = {
    'age': 45,
    'bmi': 30,
    'hba1c': 6.5,
    'glucose': 140,
    'bmi_moderate': 25,
    'hba1c_moderate': 5.7,
    'glucose_moderate': 100
}

# Visualization settings
PLOT_CONFIG = {
    'template': 'plotly_white',
    'color_continuous_scale': 'Blues',
    'opacity': 0.7,
    'height': 500
}

# Data settings
SAMPLE_SIZE_FOR_PLOTS = 1000
MAX_RECORDS_DISPLAY = 10000

# Cache settings
CACHE_EXPIRE_TIME = 3600  # 1 hour in seconds