import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import warnings
from joblib import Parallel, delayed
import joblib
import os

app = Flask(__name__)

warnings.filterwarnings('ignore')

# Loading the model

model = joblib.load('prediction.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name'].title()
    cgpa = float(request.form['cgpa'])
    intern = int(request.form['intern'])
    projects = int(request.form['projects'])
    certifs = int(request.form['certifs'])
    apt = int(request.form['apt'])
    ssr = float(request.form['ssr'])
    hsc = int(request.form['hsc'])
    exa = int(request.form['exa'])
    train = int(request.form['train'])


    # Preprocess the data
    test_data = np.array([[cgpa, intern, projects, certifs, apt, ssr, exa, train, hsc]])

    # Using the loaded model to predict
    prediction_result = model.predict(test_data)
    prediction_proba = model.predict_proba(test_data)

    prediction_proba = round(prediction_proba[0][1],4) * 100

    if prediction_result == 0:
        prediction_result = 'Not Placed'
    elif prediction_result == 1:
        prediction_result = 'Placed'
    else:
        prediction_result = 'Cxotul'

    if (exa,train) == (0,0):
        exa,train = 'No','No'
    elif (exa,train) == (1,0):
        exa, train = 'Yes','No'
    elif (exa,train) == (0,1):
        exa, train = 'No', 'Yes'
    else:
        exa, train = 'Yes', 'Yes'

    return render_template('result.html', name=name, intern=intern, projects=projects,cgpa=cgpa, certifs=certifs, apt=apt, ssr=ssr, hsc=hsc, exa=exa, train=train, prediction_result=prediction_result, prediction_proba = prediction_proba)

@app.route('/save_to_csv', methods=['POST'])
def save_to_csv():
    # Collect data from the result page
    data = {
        'Name': request.form['name'],
        'CGPA': float(request.form['cgpa']),
        'Internships': int(request.form['intern']),
        'Projects': int(request.form['projects']),
        'Certificates': int(request.form['certifs']),
        'Aptitude Score': int(request.form['apt']),
        'Soft Skill Rating': float(request.form['ssr']),
        'HSC Percentage': int(request.form['hsc']),
        'Extra Curricular Activities': request.form['exa'],
        'Placement Training': request.form['train'],
        'Placement Result': request.form['prediction_result'],
        'Placement Probability': float(request.form['prediction_proba'])
    }

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame([data])

    # Append data to the CSV file or create a new CSV file if it doesn't exist
    try:
        df.to_csv('result_data.csv', mode='a', index=False, header=not os.path.exists('result_data.csv'))
        message = 'Data successfully saved to result_data.csv'
    except Exception as e:
        message = f'Error saving data: {str(e)}'

    return message


if __name__ == '__main__':
    app.run(debug=True)
