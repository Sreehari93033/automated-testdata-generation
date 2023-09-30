import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

def preprocess_and_generate_test_data_ecomm():
    # Load and preprocess the E-commerce dataset
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, "Dataset/E-commerce Dataset.csv")

    # Convert mixed data types to strings while loading the dataset
    dtype_conversion = {
        "Gender": str,
        "Device_Type": str,
        "Customer_Login_type": str,
        "Product_Category": str,
        "Product": str,
        "Order_Priority": str,
        "Payment_method": str
    }

    dataset = pd.read_csv(dataset_path, dtype=dtype_conversion)

    # Handle missing values
    dataset.fillna(0, inplace=True)  # Fill missing values with 0, you can choose other strategies if needed

    # Encode categorical variables using one-hot encoding
    categorical_cols = ["Gender", "Device_Type", "Customer_Login_type", "Product_Category", "Product", "Order_Priority", "Payment_method"]
    numerical_cols = ["Aging", "Sales", "Quantity", "Discount", "Profit", "Shipping_Cost"]
    target_col = "Sales"

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

    # Get feature names after one-hot encoding
    encoded_feature_names = encoder.get_feature_names(categorical_cols)

    # Create a DataFrame for generated test data, actual values, and predicted values
    test_data = pd.DataFrame(X_test, columns=np.concatenate((encoded_feature_names, numerical_cols)))
    test_data[target_col] = y_test
    test_data['Predicted'] = y_pred

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    return test_data, mse
