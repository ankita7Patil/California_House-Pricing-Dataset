import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load trained California model and scaler
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# California Housing feature order (VERY IMPORTANT)
FEATURE_ORDER = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]

# Home page
@app.route('/')
def home():
    return render_template('Home.html')

# API prediction (Postman / JSON)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']

        input_data = np.array([[data[f] for f in FEATURE_ORDER]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

# HTML form prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form values in correct order
        input_data = np.array([
            float(request.form[f]) for f in FEATURE_ORDER
        ]).reshape(1, -1)

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        return render_template(
            'Home.html',
            prediction_text=f'Predicted House Price: ${prediction * 100000:,.2f}'
        )

    except Exception as e:
        return render_template(
            'Home.html',
            prediction_text=f'Error: {str(e)}'
        )

if __name__ == "__main__":
    app.run(debug=True)
