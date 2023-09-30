import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

def preprocess_and_generate_test_data_stocks():
    # Load and preprocess the stock market dataset
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, "Dataset/ADANIPORTS.csv")
    dataset = pd.read_csv(dataset_path)

    # Handle missing values
    dataset.fillna(0, inplace=True)  # Fill missing values with 0, you can choose other strategies if needed

    # Encode categorical variables using one-hot encoding
    categorical_cols = ["Symbol", "Series"]
    numerical_cols = ["Prev Close", "Open", "High", "Low", "Last", "VWAP", "Volume", "Turnover",
                      "Deliverable Volume", "%Deliverble"]
    target_col = "Close"

    # Separate features and target
    X = dataset[categorical_cols + numerical_cols]
    Y = dataset[target_col]

    # Apply one-hot encoding to categorical variables
    encoder = OneHotEncoder(drop="first", sparse=False)
    X_encoded = encoder.fit_transform(X[categorical_cols])
    X_encoded = np.concatenate((X_encoded, X[numerical_cols]), axis=1)

    # Normalize feature values
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_encoded)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y, test_size=0.2, random_state=0)

    # Fit a linear regression model on the encoded data
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = reg_model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Get the original test data
    test_data = dataset.loc[y_test.index].copy()

    # Reset index for the test data
    test_data.reset_index(drop=True, inplace=True)

    return test_data, mse
