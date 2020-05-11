# -*- coding: utf-8 -*-
"""
Created on Mon MAy 11 12:50:50 2020

@author: vbhoj
"""

#pip install flask
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('diabetes.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    
    # val = request.form['experience']
    # val = np.array(int(val))
    # final_features = val.reshape(1,1)
    # prediction = model.predict(final_features)
    # output = np.round(prediction[0],2)
    
    int_features = [int(x) for x in request.form.values()]
    int_features = int_features().reshape(1,1)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)


    if output == 1 :
        return render_template('diabetes.html', prediction_text = 'Your diabetes test is postivie {}'.format(output))
    else:
        return render_template('diabetes.html', prediction_text = 'Your diabetes test is negative {}'.format(output))

# app.run()

if __name__ == "__main__":
    app.run(debug = True)
       
    
    