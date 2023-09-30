from flask import Flask, render_template, request, jsonify
from automated_testdata_generation import preprocess_data, train_model, train_kde, generate_test_data
import numpy as np
import pickle
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    use_case = request.form.get('use-case')
    
    if use_case == 'use-case-1':
        # Load the trained models
        with open('model/rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('model/kde.pckl', 'rb') as f:
            kde = pickle.load(f)
        
        # Generate test data
        generated_data = generate_test_data(rf_model, kde)
        
        return render_template('index.html', generated_data=generated_data)
    
    elif use_case == 'use-case-2':
        # Load the trained models for the second use case
        # You can add similar loading and generation logic here for other use cases
        
        # Generate test data for the second use case
        # generated_data = generate_test_data(rf_model2, kde2)
        
        return render_template('index.html', generated_data=generated_data)

if __name__ == '__main__':
    app.run(debug=True)
