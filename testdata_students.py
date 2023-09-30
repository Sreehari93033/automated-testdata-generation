import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error
import joblib
import os

def preprocess_and_generate_test_data_students():
    # Load and preprocess the student performance dataset
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, "Dataset/student_data.csv")
    dataset = pd.read_csv(dataset_path)

    # Encode categorical variables using one-hot encoding
    categorical_cols = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian"]
    numerical_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime",
                      "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]
    target_col = "G3"

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

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=10, whiten=False)
    X_pca = pca.fit_transform(X_normalized)

    # Train RandomForestRegressor on the preprocessed data
    rf = RandomForestRegressor()
    rf.fit(X_pca, Y)

    # Save the trained model and PCA instance
    rf_model_path = "student_rf_model.pkl"
    joblib.dump(rf, rf_model_path)

    pca_model_path = "student_pca_model.pkl"
    joblib.dump(pca, pca_model_path)

    # Generate test data using Kernel Density Estimation
    kde = KernelDensity(kernel="gaussian")
    kde.fit(X_pca)

    num_test_samples = len(dataset)
    new_test_data_pca = kde.sample(num_test_samples)
    X_new_test = pca.inverse_transform(new_test_data_pca)

    # Apply PCA transformation for test data
    X_new_test_pca = pca.transform(X_new_test)

    # Get the original test data
    test_data = pd.DataFrame(X_new_test_pca, columns=[f"PC{i}" for i in range(1, X_new_test_pca.shape[1] + 1)])

    # Predict using the trained model
    y_pred = rf.predict(X_new_test_pca)
    mse = mean_squared_error(Y, y_pred)

    return test_data, mse
