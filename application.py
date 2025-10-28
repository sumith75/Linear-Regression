from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app= application

# import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

# @app.route('/')
# def index():
#     return render_template('index.html')
@app.route('/', methods=['GET' ,'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Helper to safely parse floats from the form (falls back to 0.0)
        def get_float(name):
            val = request.form.get(name)
            try:
                return float(val) if val is not None and val != '' else 0.0
            except (ValueError, TypeError):
                return 0.0

        Temperature = get_float('temperature')
        RH = get_float('rh')
        Ws = get_float('ws')
        Rain = get_float('rain')
        FFMC = get_float('ffmc')
        DMC = get_float('dmc')
        ISI = get_float('isi')
        classes = get_float('classes')
        region = get_float('region')

        # Build feature vector in the same order as the form/template
        features = [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, classes, region]

        # Scale and predict
        
        X_scaled = standard_scaler.transform([features])
        pred = ridge_model.predict(X_scaled)
        # prediction = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
       



        return render_template('home.html', result=pred[0])

    return render_template('home.html')

if __name__ == '__main__':
    # For local dev only: enables auto-reload and debug
    app.run(host='0.0.0.0', port=5000, debug=True)
