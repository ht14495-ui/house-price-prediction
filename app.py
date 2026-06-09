from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

print("Starting app...")

app = Flask(__name__)

print("Loading model...")
model = joblib.load('model.pkl')
print("Model loaded!")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({'predicted_price': round(float(prediction), 2)})

print("Launching server on port 5000...")
app.run(host='0.0.0.0', port=5000, debug=True)