import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from typing import List, Optional

def create_age_distribution_plot(thankgod_israel: pd.DataFrame, title: str = "Age Distribution") -> go.Figure:
    """
    Create an age distribution histogram by diabetes status.
    """
    fig = px.histogram(thankgod_israel, x='age', color='diabetes', 
                      nbins=50, barmode='overlay',
                      color_discrete_map={0: '#3B82F6', 1: '#EF4444'},
                      title=title,
                      opacity=0.7)
    fig.update_layout(
        showlegend=True, 
        legend_title_text='Diabetes',
        xaxis_title='Age', 
        yaxis_title='Count',
        hovermode='x unified'
    )
    return fig

def create_bmi_hba1c_scatter(thankgod_israel: pd.DataFrame, sample_size: int = 1000) -> go.Figure:
    """
    Create a scatter plot of BMI vs HbA1c level.
    """
    sample_data = thankgod_israel.sample(min(sample_size, len(thankgod_israel)))
    
    fig = px.scatter(sample_data, x='bmi', y='HbA1c_level', color='diabetes',
                    color_discrete_map={0: '#3B82F6', 1: '#EF4444'},
                    title='BMI vs HbA1c Level',
                    hover_data=['age', 'blood_glucose_level', 'gender'])
    
    fig.update_layout(
        xaxis_title='BMI',
        yaxis_title='HbA1c Level',
        showlegend=True
    )
    return fig

def create_correlation_heatmap(thankgod_israel: pd.DataFrame) -> go.Figure:
    """
    Create a correlation heatmap for numeric features.
    """
    numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 
                   'HbA1c_level', 'blood_glucose_level', 'diabetes']
    
    # Calculate correlation matrix
    corr_matrix = thankgod_israel[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Blues',
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=700
    )
    
    return fig

def create_risk_factor_bar_chart(risk_factors: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing risk factor prevalence.
    """
    fig = px.bar(risk_factors, x='Risk Factor', y='Percentage',
                title='Prevalence of Risk Factors',
                color='Percentage',
                color_continuous_scale='Blues',
                text='Percentage')
    
    fig.update_layout(
        xaxis_title='Risk Factor',
        yaxis_title='Percentage (%)',
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    return fig

def create_gender_diabetes_bar(thankgod_israel: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing diabetes distribution by gender.
    """
    gender_diabetes = thankgod_israel.groupby(['gender', 'diabetes']).size().reset_index(name='count')
    
    fig = px.bar(gender_diabetes, x='gender', y='count', color='diabetes',
                color_discrete_map={0: '#3B82F6', 1: '#EF4444'},
                title='Diabetes Cases by Gender',
                barmode='group',
                text='count')
    
    fig.update_layout(
        xaxis_title='Gender',
        yaxis_title='Count',
        showlegend=True,
        legend_title_text='Diabetes'
    )
    
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    
    return fig

def create_box_plot(thankgod_israel: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """
    Create a box plot for comparing distributions.
    """
    fig = px.box(thankgod_israel, x=x_col, y=y_col, color=x_col,
                color_discrete_map={0: '#3B82F6', 1: '#EF4444'},
                title=title,
                points=False)
    
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        showlegend=False
    )
    
    return fig