import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load trained California model and scaler
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# California Housing feature order (IMPORTANT)
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

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']

        # Ensure correct feature order
        input_data = np.array(
            [[data[feature] for feature in FEATURE_ORDER]]
        )

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
