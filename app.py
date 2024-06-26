# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:24:42 2024

@author: Admin
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder = 'test_template')
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    inp_features  = [float(x) for x in request.form.values()]
    final_features = [np.array(inp_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('index.html',prediction_text=format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data= request.get_json(force=True)
    prediction =  model.predict([np.array(list(data.values()))])
    
    output = prediction[0]
    return jsonify(output)
    
if __name__ == "__main__":
    app.run(debug=True)