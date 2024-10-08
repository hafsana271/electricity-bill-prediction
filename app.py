# app.py
import os
import pickle
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        consumption = float(request.form['consumption'])

        # Make prediction
        prediction = model.predict([[consumption]])[0]
        return render_template('index.html', prediction_text=f'Predicted Electricity Bill: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
