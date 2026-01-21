from flask import Flask, render_template, jsonify
import pickle
import numpy as np
import random

app = Flask(__name__)

# Load trained SVM model and scaler
with open("iris_svm_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

iris_classes = ["Setosa", "Versicolor", "Virginica"]

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict using random feature values
@app.route("/predict")
def predict_random():
    features = np.array([[ 
        random.uniform(4.0, 8.0),   # Sepal Length
        random.uniform(2.0, 4.5),   # Sepal Width
        random.uniform(1.0, 7.0),   # Petal Length
        random.uniform(0.1, 2.5)    # Petal Width
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)
    result = iris_classes[prediction[0]]

    return jsonify({
        "features": {
            "sepal_length": round(features[0][0], 2),
            "sepal_width": round(features[0][1], 2),
            "petal_length": round(features[0][2], 2),
            "petal_width": round(features[0][3], 2),
        },
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)
