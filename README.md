\# Diabetes Prediction Dashboard (LightGBM + Streamlit)



\## Project Overview

This project presents an end-to-end machine learning pipeline for \*\*diabetes prediction\*\* using clinical, demographic, and lifestyle data. The workflow follows best practices across the full data science lifecycle: exploratory data analysis (EDA), feature engineering, model development, evaluation, and deployment via a \*\*Streamlit web application\*\*.



The final deployed model is a \*\*LightGBM (LGBMClassifier)\*\* trained on engineered features, with preprocessing handled using \*\*StandardScaler\*\* and \*\*SelectKBest (k = 15)\*\*. The system supports both \*\*single-patient prediction\*\* and \*\*batch prediction\*\* using engineered CSV files.



---



\## Objectives

\- Explore and understand diabetes-related clinical data

\- Engineer meaningful features to improve predictive performance

\- Train and evaluate a robust classification model

\- Select the best-performing model using appropriate metrics

\- Deploy the trained model using Streamlit for real-time inference



---



\## Project Structure

diabetes\_prediction\_dashboard/

│

├── notebooks/

│ ├── 01\_data\_exploration.ipynb

│ ├── 02\_feature\_engineering.ipynb

│ └── 03\_model\_development.ipynb

│

├── data/

│ ├── cleaned\_diabetes\_data.csv

│ ├── engineered\_train\_data.csv

│ ├── engineered\_val\_data.csv

│ └── engineered\_test\_data.csv

│

├── best\_model/

│ ├── best\_model.pkl

│ ├── metadata.json

│ └── preprocessing/

│ ├── scaler.pkl

│ └── feature\_selector.pkl

│

├── app.py

├── requirements.txt

└── README.md





---



\## Dataset Description

The dataset consists of patient-level medical and lifestyle information used to predict diabetes status.



\### Target Variable

\- \*\*diabetes\*\*  

&nbsp; - 0 = Non-diabetic  

&nbsp; - 1 = Diabetic  



\### Core Features

\- Demographic: `gender`, `age`

\- Clinical: `bmi`, `HbA1c\_level`, `blood\_glucose\_level`

\- Medical History: `hypertension`, `heart\_disease`

\- Lifestyle: `smoking\_history`



---



\## Notebook Workflow



\### 1. `01\_data\_exploration.ipynb`

This notebook focuses on understanding the dataset and identifying data quality issues.



Key steps:

\- Dataset loading and inspection

\- Missing value and duplicate checks

\- Class imbalance analysis

\- Distribution analysis of key medical variables

\- Preliminary relationship checks between features and diabetes status



Outcome:

\- Clear understanding of data structure and imbalance

\- Basis for informed feature engineering decisions



---



\### 2. `02\_feature\_engineering.ipynb`

This notebook transforms the cleaned dataset into a model-ready format.



Key steps:

\- Encoding categorical variables:

&nbsp; - Gender encoding (Female = 0, Male = 1, Other = 2)

&nbsp; - One-hot encoding of smoking history

\- Feature engineering:

&nbsp; - Interaction features (e.g., age × BMI, HbA1c × glucose)

&nbsp; - Polynomial features (squared terms)

&nbsp; - Composite medical risk score

\- Stratified train/validation/test split

\- Saving engineered datasets



Outcome:

\- `engineered\_train\_data.csv`

\- `engineered\_val\_data.csv`

\- `engineered\_test\_data.csv`



---



\### 3. `03\_model\_development.ipynb`

This notebook handles model training, evaluation, and artifact persistence.



Key steps:

\- Scaling features using `StandardScaler` (fit on training set only)

\- Feature selection using `SelectKBest (k = 15)`

\- Model training using \*\*LightGBM\*\*

\- Evaluation on validation and test sets using:

&nbsp; - Accuracy

&nbsp; - Precision

&nbsp; - Recall

&nbsp; - F1-score

&nbsp; - ROC-AUC

\- Saving deployment artifacts



Saved artifacts:

\- `best\_model.pkl` (trained LightGBM model)

\- `scaler.pkl`

\- `feature\_selector.pkl`

\- `metadata.json` (metrics and selected features)



---



\## Model Selection

After comparative evaluation, \*\*LightGBM\*\* was selected due to:

\- Strong performance on imbalanced classification problems

\- High recall and ROC-AUC for diabetic class detection

\- Efficient training and inference

\- Native support for feature importance and probability outputs



---



\## Deployment (Streamlit App)



\### Features

\- \*\*Single Prediction Mode\*\*

&nbsp; - User inputs patient details via form

&nbsp; - Real-time diabetes prediction with probability score

\- \*\*Batch Prediction Mode\*\*

&nbsp; - Upload engineered CSV file

&nbsp; - Download predictions as a new CSV



\### How It Works

The Streamlit app:

1\. Reconstructs engineered features from user input

2\. Applies saved scaler and feature selector

3\. Uses the trained LightGBM model for inference

4\. Displays predictions and probabilities



---



\## Running the Project



\### 1. Install Dependencies

```bash

pip install -r requirements.txt





\# These must be run first

* 01\_data\_exploration.ipynb
* 02\_feature\_engineering.ipynb
* 03\_model\_development.ipynb
  



\# Lauching Streamlit app

* python -m streamlit run app.py







\# Evaluation Metrics 

\## Metrics are saved in best\_model/metadata.json.



\### Typical performance indicators include:



* High recall for diabetic class (minimizing false negatives)



* Strong ROC-AUC, indicating good class separation



* Balanced precision–recall tradeoff







