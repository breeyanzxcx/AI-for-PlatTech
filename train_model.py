# train_model.py
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save model
with open("iris_svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")
