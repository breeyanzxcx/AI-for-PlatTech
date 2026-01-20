from flask import Flask, render_template, jsonify
import pickle
import numpy as np
import random

app = Flask(__name__)

# Load trained SVM model
with open("iris_svm_model.pkl", "rb") as f:
    model = pickle.load(f)

iris_classes = ["Setosa", "Versicolor", "Virginica"]

# Home page route
@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict")
def predict_random():
    features = np.array([[ 
        random.uniform(4.0, 8.0),
        random.uniform(2.0, 4.5),
        random.uniform(1.0, 7.0),
        random.uniform(0.1, 2.5)
    ]])
    prediction = model.predict(features)
    result = iris_classes[prediction[0]]
    return jsonify({
        "features": features.tolist(),
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)
