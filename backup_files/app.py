from flask import Flask, render_template, request
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestRegressor
import joblib

# Import your separate use case files
import automated_testdata_generation
import automated_testdata_ecommerce
import automated_testdata_students
import automated_testdata_stockMarket
import automated_testdata_health

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_test_data', methods=['POST'])
def generate_test_data():
    use_case = request.form['use_case']  # Get the selected use case from the form

    if use_case == 'tcp_congestion':
        # Call the function from your TCP congestion use case file
        results = automated_testdata_generation.generate_tcp_congestion_test_data()

    elif use_case == 'ecommerce':
        # Call the function from your e-commerce use case file
        results = automated_testdata_ecommerce.generate_ecommerce_test_data()

    elif use_case == 'students':
        # Call the function from your students use case file
        results = automated_testdata_students.generate_students_test_data()

    elif use_case == 'stock':
        # Call the function from your stock market use case file
        results = automated_testdata_stockMarket.generate_stock_test_data()

    elif use_case == 'health':
        # Call the function from your health use case file
        results = automated_testdata_health.generate_health_test_data()

    return render_template('index.html', generated_data=results)

if __name__ == '__main__':
    app.run(debug=True)
