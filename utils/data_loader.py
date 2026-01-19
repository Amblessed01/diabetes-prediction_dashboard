import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
import streamlit as st

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the diabetes prediction dataset from the specified file path.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        thankgod_israel = pd.read_csv(file_path)
        st.success(f"Dataset loaded successfully: {len(thankgod_israel):,} records")
        return thankgod_israel
    except FileNotFoundError:
        st.error(f" File not found: {file_path}")
        return create_sample_data()

def create_sample_data(n_samples: int = 100000) -> pd.DataFrame:
    """
    Create sample dataset when the actual file is not found.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Generated sample dataset
    """
    np.random.seed(42)
    
    data = {
        'gender': np.random.choice(['Female', 'Male', 'Other'], n_samples, p=[0.585, 0.414, 0.001]),
        'age': np.random.uniform(0.08, 80, n_samples),
        'hypertension': np.random.choice([0, 1], n_samples, p=[0.925, 0.075]),
        'heart_disease': np.random.choice([0, 1], n_samples, p=[0.96, 0.04]),
        'smoking_history': np.random.choice(['never', 'No Info', 'former', 'current', 'not current', 'ever'], 
                                           n_samples, p=[0.35, 0.358, 0.094, 0.093, 0.064, 0.04]),
        'bmi': np.random.uniform(10.01, 95.69, n_samples),
        'HbA1c_level': np.random.uniform(3.5, 9.0, n_samples),
        'blood_glucose_level': np.random.uniform(80, 300, n_samples),
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.915, 0.085])
    }
    
    thankgod_israel = pd.DataFrame(data)
    st.warning(f" Using generated sample data with {n_samples:,} records")
    return thankgod_israel

def get_data_info(thankgod_israel: pd.DataFrame) -> dict:
    """
    Get comprehensive information about the dataset.
    
    Args:
        thankgod_israel (pd.DataFrame): The dataset
        
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'shape': thankgod_israel.shape,
        'columns': list(thankgod_israel.columns),
        'dtypes': thankgod_israel.dtypes.to_dict(),
        'missing_values': thankgod_israel.isnull().sum().sum(),
        'duplicates': thankgod_israel.duplicated().sum(),
        'diabetic_count': (thankgod_israel['diabetes'] == 1).sum(),
        'non_diabetic_count': (thankgod_israel['diabetes'] == 0).sum(),
        'diabetic_percentage': ((thankgod_israel['diabetes'] == 1).sum() / len(thankgod_israel)) * 100,
        'memory_usage_mb': thankgod_israel.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    # Add basic statistics for numeric columns
    numeric_cols = thankgod_israel.select_dtypes(include=[np.number]).columns
    info['numeric_stats'] = thankgod_israel[numeric_cols].describe().to_dict()
    
    # Add unique values for categorical columns
    categorical_cols = thankgod_israel.select_dtypes(include=['object']).columns
    info['categorical_unique'] = {col: thankgod_israel[col].nunique() for col in categorical_cols}
    
    return info

def filter_data(
    thankgod_israel: pd.DataFrame, 
    age_range: Tuple[float, float] = (0, 80),
    diabetes_filter: str = "All"
) -> pd.DataFrame:
    """
    Filter the dataset based on criteria.
    
    Args:
        thankgod_israel (pd.DataFrame): The dataset
        age_range (tuple): Min and max age for filtering
        diabetes_filter (str): "All", "Diabetic", or "Non-Diabetic"
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    filtered = thankgod_israel.copy()
    
    # Apply age filter
    filtered = filtered[
        (filtered['age'] >= age_range[0]) & 
        (filtered['age'] <= age_range[1])
    ]
    
    # Apply diabetes filter
    if diabetes_filter == "Diabetic":
        filtered = filtered[filtered['diabetes'] == 1]
    elif diabetes_filter == "Non-Diabetic":
        filtered = filtered[filtered['diabetes'] == 0]
    
    return filtered