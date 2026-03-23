from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "decision_re.pkl")

# Load the model using the absolute file path
with open(file_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('Decision_re.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input variables from the request
    variable1 = float(request.form['variable1'])
    variable2 = float(request.form['variable2'])
    variable3 = float(request.form['variable3'])
    variable4 = float(request.form['variable4'])

    # Perform prediction using the loaded model
    prediction = model.predict([[variable1, variable2, variable3, variable4]])

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=True for development purposes
