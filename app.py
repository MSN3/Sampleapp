from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('sampleapp')
cols = ['Dist', 'LdTime', 'TrlLng', 'Wgt', 'Equpt']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols) 
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    
    if int(final[0]) not in range(500,1801):
        return render_template('home.html', pred='WARNING!! Value of distance outside the suggested range. May result in abnormal values. Expected Cost Per Load will be {}'.format(prediction))
    elif int(final[3]) not in range(15000,40001):
        return render_template('home.html', pred='WARNING!! Value of weight outside the suggested range. May result in abnormal values. Expected Cost Per Load will be {}'.format(prediction))
    elif int(final[2]) != 53 and int(final[2]) != 48 : 
        return render_template('home.html', pred='Value Error. Please choose a trailer length value from the dropbox above the line.')
    elif final[4]!='DRY' and final[4]!='REF':
        return render_template('home.html', pred='ValueError. Please choose an equipment type from the dropbox above the line.')   
    else:
        return render_template('home.html',pred='Expected Cost Per Load will be {}'.format(prediction))
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
