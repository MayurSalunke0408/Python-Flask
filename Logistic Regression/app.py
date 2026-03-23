from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "logistic.pkl")

# Load the model using the absolute file path
with open(file_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("Logistic.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get variables from the form
        variable1 = float(request.form["variable1"])

        # Predict using the loaded model
        prediction = model.predict([[variable1]])

        # Convert prediction to human-readable label
        prediction_label = "Yes" if prediction[0] == 1 else "No"

        return jsonify({"prediction": prediction_label})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
