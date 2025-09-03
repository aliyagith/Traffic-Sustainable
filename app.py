from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = 'traffic_secret_key'


df = pd.read_csv('futuristic_city_traffic.csv')

# Load the save model and preprocessing components
model=joblib.load('traffic_model.pkl')
scala = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Home route
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict")
@login_required
def predict():
    return render_template('predict.html')

# About route
@app.route("/about")
def about():
    return render_template('About.html')

@app.route("/login")
def login():
    return render_template('login.html')


if __name__ == "__main__":
    app.run(debug=True)