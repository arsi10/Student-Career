import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import warnings
from joblib import Parallel, delayed
import joblib

app = Flask(__name__)

warnings.filterwarnings('ignore')
#Loading the model
model = joblib.load('prediction.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name'].title()
    email = request.form['email']
    cgpa = float(request.form['cgpa'])
    ss = int(request.form['ss'])
    ml = int(request.form['ml'])


    # Preprocess the data
    test_data = np.array([[cgpa, ss, ml]])

    #Using the loaded model to predict
    prediction_result = model.predict(test_data)

    if prediction_result == 0:
        prediction_result = 'Not Placed'
    elif prediction_result == 1:
        prediction_result = 'Placed'
    else:
        prediction_result = 'Cxotul'

    return render_template('result.html', name=name, email=email, prediction_result=prediction_result,cgpa=cgpa,ss=ss,ml=ml)

if __name__ == '__main__':
    app.run(debug=True)
