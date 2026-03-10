from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# load dataset
iris = load_iris()

X = iris.data
y = iris.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# save model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")
