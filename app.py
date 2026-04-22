from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), 'salary_predict_model.pkl')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Home route
@app.route("/")
def home():
    return "Sneha API is running 🚀"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [
        data['age'],
        data['gender'],
        data['country'],
        data['highest_edu'],
        data['coding_exp'],
        data['title'],
        data['company_size']
    ]

    prediction = model.predict([features])

    return jsonify({
        "predicted_salary": float(prediction[0])
    })

# Local run (Azure ignores this)
if __name__ == "__main__":
    app.run(debug=True)