from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from joblib import load

app = Flask(__name__)
app.secret_key = 'traffic_secret_key'

# Load models lazily to keep app startup simple
_density_model = None
_incident_model = None

def get_density_model():
    global _density_model
    if _density_model is None:
        model_path = os.path.join('Traffic_Model', 'traffic_density_xgb_pipeline.pkl')
        _density_model = load(model_path)
    return _density_model

def get_incident_model():
    global _incident_model
    if _incident_model is None:
        model_path = os.path.join('Traffic_Model', 'incident_xgb_pipeline.pkl')
        _incident_model = load(model_path)
    return _incident_model


def build_density_explanation(row: dict, pred: float) -> list:
    reasons = []
    # Peak hour
    if row.get('Is Peak Hour') in (1, '1', 'on', True):
        reasons.append('Peak hour typically increases congestion and density.')
    # Random event
    try:
        if int(row.get('Random Event Occurred', 0)) == 1:
            reasons.append('A random event was reported, which tends to spike density.')
    except Exception:
        pass
    # Weather
    weather = (row.get('Weather') or '').lower()
    if weather in ['rainy', 'snowy', 'electromagnetic storm', 'solar flare']:
        reasons.append(f"Weather condition '{row.get('Weather')}' generally slows traffic and raises density.")
    elif weather == 'clear':
        reasons.append("Clear weather is associated with lower density.")
    # Speed
    try:
        spd = float(row.get('Speed', 0))
        if spd <= 40:
            reasons.append('Lower observed speed suggests heavier congestion.')
        elif spd >= 90:
            reasons.append('Higher speed indicates freer flow, reducing density.')
    except Exception:
        pass
    # Hour bands
    try:
        hr = int(row.get('Hour Of Day', 0))
        if 7 <= hr <= 10 or 17 <= hr <= 20:
            reasons.append('Typical rush hours (commute times) elevate density.')
    except Exception:
        pass
    # City/Vehicle/Economy mention
    if row.get('Economic Condition') in ['Recession']:
        reasons.append('During recession, demand patterns vary; some corridors may see elevated density.')
    if not reasons:
        reasons.append('Conditions suggest moderate flow; model combined all inputs for this estimate.')
    reasons.insert(0, f"Estimated density: {pred:.4f} (lower means freer flow, higher means more congestion).")
    return reasons


def build_incident_explanation(row: dict, proba: float, pred: int) -> list:
    reasons = [
        f"Incident likelihood: {proba*100:.2f}% → {'Likely' if pred==1 else 'Unlikely'}"
    ]
    # Peak hour
    if row.get('Is Peak Hour') in (1, '1', 'on', True):
        reasons.append('Peak hour increases exposure and minor disruptions likelihood.')
    # Weather
    weather = (row.get('Weather') or '').lower()
    if weather in ['rainy', 'snowy', 'electromagnetic storm', 'solar flare']:
        reasons.append(f"Adverse weather ('{row.get('Weather')}') raises incident risk.")
    # Speed extremes
    try:
        spd = float(row.get('Speed', 0))
        if spd >= 100:
            reasons.append('Very high speeds can correlate with higher incident risk.')
        elif spd <= 25:
            reasons.append('Very low speeds may reflect disruptive conditions in progress.')
    except Exception:
        pass
    # Hour bands
    try:
        hr = int(row.get('Hour Of Day', 0))
        if 22 <= hr or hr <= 5:
            reasons.append('Late-night/early-morning hours can have elevated risk on some corridors.')
    except Exception:
        pass
    if not reasons:
        reasons.append('Model combined all inputs to estimate the probability.')
    return reasons

@app.route("/")
def index():
    return render_template('Index.html')


@app.route("/about")
def about():
    return render_template('About.html')

@app.route("/login")
def login():
    return render_template('login.html')


# -------- Density Prediction --------
@app.route('/predict/density', methods=['GET', 'POST'])
def predict_density():
    if request.method == 'POST':
        try:
            form = request.form
            # Build a single-row DataFrame matching training feature names
            row = {
                'City': form.get('City'),
                'Vehicle Type': form.get('Vehicle Type'),
                'Weather': form.get('Weather'),
                'Economic Condition': form.get('Economic Condition'),
                'Day Of Week': form.get('Day Of Week'),
                'Hour Of Day': int(form.get('Hour Of Day', 0)),
                'Speed': float(form.get('Speed', 0)),
                'Is Peak Hour': 1 if form.get('Is Peak Hour') == 'on' else 0,
                'Random Event Occurred': int(form.get('Random Event Occurred', 0)),
                'Energy Consumption': float(form.get('Energy Consumption', 0.0)),
            }
            df = pd.DataFrame([row])
            model = get_density_model()
            pred = float(model.predict(df)[0])
            explanation = build_density_explanation(row, pred)
            return render_template('density_predict.html', prediction=pred, form_data=row, explanation=explanation)
        except Exception as e:
            return render_template('density_predict.html', prediction=None, form_data=form.to_dict(), error=str(e))
    # GET
    return render_template('density_predict.html', prediction=None, form_data={})


# -------- Incident/Disruption Classification --------
@app.route('/predict/incident', methods=['GET', 'POST'])
def predict_incident():
    if request.method == 'POST':
        try:
            form = request.form
            # Features exclude target and leakage columns (Traffic Density, Energy Consumption)
            row = {
                'City': form.get('City'),
                'Vehicle Type': form.get('Vehicle Type'),
                'Weather': form.get('Weather'),
                'Economic Condition': form.get('Economic Condition'),
                'Day Of Week': form.get('Day Of Week'),
                'Hour Of Day': int(form.get('Hour Of Day', 0)),
                'Speed': float(form.get('Speed', 0)),
                'Is Peak Hour': 1 if form.get('Is Peak Hour') == 'on' else 0,
            }
            # Backward-compat: if current model pipeline expects leakage columns, provide neutral defaults
            row.setdefault('Energy Consumption', 0.0)
            row.setdefault('Traffic Density', 0.0)
            df = pd.DataFrame([row])
            model = get_incident_model()
            proba = float(model.predict_proba(df)[0, 1])
            pred = int(proba >= 0.5)
            explanation = build_incident_explanation(row, proba, pred)
            return render_template('incident_predict.html', prediction=pred, probability=proba, form_data=row, explanation=explanation)
        except Exception as e:
            return render_template('incident_predict.html', prediction=None, probability=None, form_data=form.to_dict(), error=str(e))
    # GET
    return render_template('incident_predict.html', prediction=None, probability=None, form_data={})


if __name__ == "__main__":
    app.run(debug=True)