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

if __name__ == '__main__':
    from waitress import serve
<<<<<<< HEAD
    serve(app, host='0.0.0.0', port=8080)
=======
    serve(app, host='0.0.0.0', port=8080)
>>>>>>> 4b4ac610562285bbc35c35bce1e124cb5d444165
