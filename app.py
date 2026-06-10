from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({'predicted_price': round(float(prediction), 2)})
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'House Price Prediction API is running!',
        'usage': 'Send a POST request to /predict with house features'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)