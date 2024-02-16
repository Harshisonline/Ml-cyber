from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import csv

def append_to_csv(file_name, data):
    with open(file_name, 'a', newline='') as csvfile:
        fieldnames = ['URL', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Check if the file is empty and write headers if needed
        if csvfile.tell() == 0:
            writer.writeheader()
        
        # Write data to the CSV file
        writer.writerow(data)

app = Flask(__name__)
loaded_model = pickle.load(open(r'model/phishing.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input from the form or JSON request
        url = request.form['url']

        result = loaded_model.predict([url])

        # Return the result
        return render_template('result.html', prediction=result)

@app.route('/report', methods=['POST'])
def report():
    if request.method == 'POST':
        # Get the input from the form or JSON request
        url = request.form['url']
        option = request.form['option']
        
        if option == 'yes':
            label = 'bad'
        else:
            label = 'good'
        
        file_name = r'dataset\report.csv'

        # Append data to the CSV file
        try:
            new_data = {'URL': url, 'Label': label}
            append_to_csv(file_name, new_data)
            feedback_message = "URL reported successfully."
        except Exception as e:
            feedback_message = f"An error occurred: {str(e)}"

        return render_template('index.html', feedback_message=feedback_message)

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/about')
def about():
     return render_template('about.html')