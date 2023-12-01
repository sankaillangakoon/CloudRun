# Import necessary libraries
from google.cloud import bigquery
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint

# Initialize BigQuery client
client = bigquery.Client()

# Set BigQuery dataset parameters
PROJECT_ID = 'your-project-id'
DATASET_ID = 'Vertex'
TABLE_ID_VALID = 'EPCValid'
TABLE_ID_INVALID = 'EPCInvalid'

# Define SQL queries
sql_valid = f"""
SELECT 
* 
FROM 
`{PROJECT_ID}, {DATASET_ID}, {TABLE_ID_VALID}
"""
sql_invalid = f"""
SELECT 
* 
FROM 
`{PROJECT_ID}, {DATASET_ID}, {TABLE_ID_INVALID}
"""

# Load datasets from BigQuery
epc_valid = client.query(sql_valid).result().to_dataframe()
epc_invalid = client.query(sql_invalid).result().to_dataframe()

# Split datasets into training and validation datasets
X = epc_valid.drop(columns=['co2_emissions_current'])
y = epc_valid['co2_emissions_current']
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess the data
categorical_cols = ['property_type', 'built_form', 'main_fuel_rename', 'mainheat_env_eff', 'hot_water_env_eff', 'floor_env_eff', 'windows_env_eff', 'walls_env_eff', 'roof_env_eff', 'mainheatc_env_eff', 'lighting_env_eff']
numerical_cols = ['total_floor_area', 'construction_year']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_validation_transformed = preprocessor.transform(X_validation)

# Hyperparameter tuning
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

