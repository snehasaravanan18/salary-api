from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('salary_predict_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Sneha Saravanan"

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

if __name__ == "__main__":
    app.run(debug=True)