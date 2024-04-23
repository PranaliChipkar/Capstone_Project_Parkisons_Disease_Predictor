"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st


@st.cache_data()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('Parkinson.csv')

    # Rename the column names in the DataFrame.
    df.rename(columns = {"MDVP:Fo(Hz)": "AVFF",}, inplace = True)
    df.rename(columns = {"MDVP:Fhi(Hz)": "MAVFF",}, inplace = True)
    df.rename(columns = {"MDVP:Flo(Hz)": "MIVFF",}, inplace = True)
    

    # Perform feature and target split
    X = df[["AVFF", "MAVFF", "MIVFF","Jitter:DDP","MDVP:Jitter(%)","MDVP:RAP","MDVP:APQ","MDVP:PPQ","MDVP:Shimmer","Shimmer:DDA","Shimmer:APQ3","Shimmer:APQ5","NHR","HNR","RPDE","DFA","D2","PPE"]]
    y = df['status']

    return df, X, y

# @st.cache_data()
def train_model(X, y):
    """This function trains the model and return the model and model score"""
    # Create copies of input arrays
    X_copy = np.copy(X)
    y_copy = np.copy(y)

    # Create the model
    model = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='entropy',
            max_depth=4, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1, 
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=42, splitter='best'
        )
    # Fit the data on model
    model.fit(X_copy, y_copy)
    # Get the model score
    score = model.score(X_copy, y_copy)

    # Return the values
    return model, score

def predict(X, y, features):
    # Convert features to a NumPy array with the correct data type
    features_array = np.array(features, dtype=np.float64)
    # Create a copy of the array
    features_array_copy = np.copy(features_array)
    # Get model and model score
    model, score = train_model(X, y)
    # Predict the value
    prediction = model.predict(features_array_copy.reshape(1, -1))

    return prediction, score
