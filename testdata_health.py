import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def preprocess_and_generate_test_data_health():
    # Load and preprocess the Heart Diagnosis dataset
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, "Dataset/Heart Attack.csv")
    dataset = pd.read_csv(dataset_path)

    # Encode categorical variables using one-hot encoding
    categorical_cols = ["gender"]
    numerical_cols = ["age", "impluse", "pressurehight", "pressurelow", "glucose", "kcm", "troponin"]
    target_col = "class"

    # Separate features and target
    X = dataset[categorical_cols + numerical_cols]
    Y = dataset[target_col]

    # Apply one-hot encoding to categorical variables
    encoder = OneHotEncoder(drop="first", sparse=False)
    X_encoded = encoder.fit_transform(X[categorical_cols])

    # Normalize feature values
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_encoded)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y, test_size=0.2, random_state=0)

    # Fit a logistic regression model on the encoded data
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = logreg_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Get the original test data
    test_data = dataset.loc[y_test.index].copy()

    # Reset index for the test data
    test_data.reset_index(drop=True, inplace=True)

    return test_data, accuracy
