from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def index():
    message = []
    return render_template('index.html', message=message)

@app.route('/predict', methods=['POST'])
def predict():
    message = []
    try:
        # Get input from form
        feature = request.form['feature']
        
        # Convert input string to list of floats
        feature_list = [float(x) for x in feature.split(',')]
        
        # Reshape for model input
        data = np.array(feature_list).reshape(1, -1)
        
        # Predict
        prediction = model.predict(data)[0]

        # Convert prediction to readable message
        if prediction == 0:
            message = ['Not Cancerous']
        else:
            message = ['Cancerous']
            
    except Exception as e:
        message = [f"Error: {e}"]
        
    return render_template('index.html', message=message)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)