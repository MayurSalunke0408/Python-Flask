from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "decision_cl.pkl")

# Load the model using the absolute file path
with open(file_path, "rb") as f:
    model = pickle.load(f)

def convert_to_response(prediction):
    if prediction == 0:
        return "No"
    elif prediction == 1:
        return "Yes"
    else:
        return "Unknown"

@app.route('/')
def index():
    return render_template('Decision_cl.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input variables from the request
        variable1 = float(request.form['variable1'])
        variable2 = float(request.form['variable2'])
        variable3 = float(request.form['variable3'])
        variable4 = float(request.form['variable4'])

        # Perform prediction using the loaded model
        prediction = model.predict([[variable1, variable2, variable3, variable4]])

        # Convert the prediction into "No" or "Yes"
        response = convert_to_response(prediction[0])

        # Return the prediction as JSON response
        return jsonify({'prediction': response})
    except Exception as e:
        # Return error message if prediction fails
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=True for development purposes
