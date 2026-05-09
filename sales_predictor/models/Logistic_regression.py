import pandas as pd
import numpy as np

# Load the trained model
df = pd.read_csv('/data/generate.csv')
def Logistic_regression_model():
    X = df[['promo', 'price', 'marketing', 'month', 'sales']]