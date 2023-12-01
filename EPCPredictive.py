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

