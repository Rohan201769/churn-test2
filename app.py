import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scaler
model = load_model('churn_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = np.array([[
        float(data['CreditScore']),
        float(data['Age']),
        float(data['Tenure']),
        float(data['Balance']),
        float(data['NumOfProducts']),
        float(data['HasCrCard']),
        float(data['IsActiveMember']),
        float(data['EstimatedSalary']),
        float(data['Geography_Germany']),
        float(data['Geography_Spain']),
        float(data['Gender_1'])
    ]])

    # Scale the features
    features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features)
    prediction = 'Churn' if prediction[0] > 0.5 else 'Not Churn'

    return render_template('index.html', prediction_text=f'Customer will {prediction}')

if __name__ == '__main__':
    app.run(debug=True)