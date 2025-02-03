from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Train the model (this will run when the app starts)
def train_model():
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Load or train model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        
        # Get the flower name
        iris = load_iris()
        output = iris.target_names[prediction[0]]
        
        return render_template('index.html', 
                             prediction_text=f'Predicted Iris Species: {output}')
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)