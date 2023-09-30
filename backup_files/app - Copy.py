from flask import Flask, render_template, request
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_test_data', methods=['POST'])
def generate_test_data():
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, "model/kde.pckl"), 'rb') as f:
        kde = pickle.load(f)

    # Load trained RandomForestRegressor model
    rf_model_path = os.path.join(script_dir, "model", "rf_model.pkl")
    rf = joblib.load(rf_model_path)

    # Create and fit a new PCA instance during test data generation
    pca = PCA(n_components=10, whiten=False)
    
    # Transform new test data using the new PCA instance
    new_test_data = np.abs(kde.sample(50))
    X_new_test = pca.fit_transform(new_test_data)  # Fit and transform new test data
    y_label = rf.predict(X_new_test)

    generated_data = []
    for i in range(len(new_test_data)):
        generated_data.append({
            "test_data": new_test_data[i],
            "original_delay": y_label[i]
        })

    return render_template('index.html', generated_data=generated_data)


if __name__ == '__main__':
    app.run(debug=True)
