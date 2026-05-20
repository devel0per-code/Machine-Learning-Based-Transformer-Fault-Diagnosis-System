from flask import Flask, request, jsonify, render_template_string
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Transformer Fault Diagnosis API</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; }
      .container { max-width: 650px; margin: auto; }
      label { display: block; margin-top: 1rem; }
      input { width: 100%; padding: 0.5rem; margin-top: 0.25rem; }
      button { margin-top: 1rem; padding: 0.75rem 1.25rem; }
      .result { margin-top: 1.5rem; padding: 1rem; border: 1px solid #ccc; background: #f9f9f9; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Transformer Fault Diagnosis</h1>
      <p>Use this API to predict transformer fault type from gas readings.</p>
      <form method="post" action="/predict">
        {% for gas, default in defaults.items() %}
          <label for="{{gas}}">{{gas}} (ppm)</label>
          <input type="number" step="any" name="{{gas}}" id="{{gas}}" value="{{default}}" required>
        {% endfor %}
        <button type="submit">Predict</button>
      </form>

      {% if prediction %}
        <div class="result">
          <h2>Prediction</h2>
          <p><strong>Random Forest:</strong> {{ prediction.random_forest }}</p>
        </div>
      {% endif %}
    </div>
  </body>
</html>
"""

DEFAULTS = {'H2': 100.0, 'CH4': 50.0, 'C2H4': 20.0, 'C2H2': 5.0, 'CO': 200.0, 'CO2': 1500.0}

FAULT_TYPES = [
    'Normal', 'PD', 'T1', 'T2', 'T3', 'D1', 'D2'
]

RANGES = {
    'Normal': {'H2': (10, 50), 'CH4': (10, 30), 'C2H4': (1, 10), 'C2H2': (0, 1), 'CO': (100, 300), 'CO2': (1000, 3000)},
    'PD': {'H2': (100, 500), 'CH4': (50, 150), 'C2H4': (1, 10), 'C2H2': (0, 1), 'CO': (100, 300), 'CO2': (1000, 3000)},
    'T1': {'H2': (10, 50), 'CH4': (50, 200), 'C2H4': (10, 50), 'C2H2': (0, 2), 'CO': (100, 500), 'CO2': (1000, 4000)},
    'T2': {'H2': (10, 50), 'CH4': (50, 200), 'C2H4': (50, 150), 'C2H2': (0, 2), 'CO': (100, 500), 'CO2': (1000, 4000)},
    'T3': {'H2': (10, 50), 'CH4': (50, 200), 'C2H4': (150, 500), 'C2H2': (0, 5), 'CO': (100, 500), 'CO2': (1000, 4000)},
    'D1': {'H2': (200, 800), 'CH4': (50, 200), 'C2H4': (10, 50), 'C2H2': (50, 200), 'CO': (100, 500), 'CO2': (1000, 4000)},
    'D2': {'H2': (200, 800), 'CH4': (50, 200), 'C2H4': (50, 200), 'C2H2': (100, 500), 'CO': (100, 500), 'CO2': (1000, 4000)}
}


def generate_synthetic_data(num_samples=2000):
    X, y = [], []
    counts = {fault: num_samples // len(FAULT_TYPES) for fault in FAULT_TYPES}
    counts['Normal'] += num_samples - sum(counts.values())

    for fault, n in counts.items():
        spec = RANGES[fault]
        for _ in range(n):
            sample = [
                max(0, np.random.uniform(*spec['H2']) + np.random.normal(0, 0.1 * spec['H2'][1])),
                max(0, np.random.uniform(*spec['CH4']) + np.random.normal(0, 0.1 * spec['CH4'][1])),
                max(0, np.random.uniform(*spec['C2H4']) + np.random.normal(0, 0.1 * spec['C2H4'][1])),
                max(0, np.random.uniform(*spec['C2H2']) + np.random.normal(0, 0.1 * max(spec['C2H2'][1], 1.0))),
                max(0, np.random.uniform(*spec['CO']) + np.random.normal(0, 0.1 * spec['CO'][1])),
                max(0, np.random.uniform(*spec['CO2']) + np.random.normal(0, 0.1 * spec['CO2'][1]))
            ]
            X.append(sample)
            y.append(fault)

    return np.array(X), np.array(y)


def extract_features(sample):
    h2, ch4, c2h4, c2h2, co, co2 = sample
    return np.array([
        h2, ch4, c2h4, c2h2, co, co2,
        ch4 / max(h2, 1e-6),
        c2h2 / max(c2h4, 1e-6),
        c2h2 / max(ch4, 1e-6),
        co2 / max(co, 1e-6)
    ])


def build_model():
    raw_X, raw_y = generate_synthetic_data(2000)
    X = np.vstack([extract_features(row) for row in raw_X])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, raw_y)
    return model, scaler

RF_MODEL, MODEL_SCALER = build_model()


def parse_inputs(source):
    try:
        return {gas: float(source.get(gas, DEFAULTS[gas])) for gas in DEFAULTS}
    except (TypeError, ValueError):
        return None


@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, defaults=DEFAULTS, prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    data = parse_inputs(request.form)
    if data is None:
        return "Invalid input values. Please enter valid numbers.", 400

    features = extract_features([data['H2'], data['CH4'], data['C2H4'], data['C2H2'], data['CO'], data['CO2']])
    input_scaled = MODEL_SCALER.transform([features])
    rf_pred = RF_MODEL.predict(input_scaled)[0]

    return render_template_string(HTML_TEMPLATE, defaults=data, prediction={'random_forest': rf_pred})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(force=True, silent=True) or request.form
    data = parse_inputs(payload)
    if data is None:
        return jsonify({'error': 'Invalid input values; please provide numeric gas readings.'}), 400

    features = extract_features([data['H2'], data['CH4'], data['C2H4'], data['C2H2'], data['CO'], data['CO2']])
    input_scaled = MODEL_SCALER.transform([features])
    rf_pred = RF_MODEL.predict(input_scaled)[0]

    return jsonify({'random_forest': rf_pred})
