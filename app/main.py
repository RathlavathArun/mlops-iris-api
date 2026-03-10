from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# load model
model = joblib.load("app/model.pkl")

@app.get("/")
def home():
    return {"message": "Model API is running successfully"}

@app.post("/predict")
def predict(data: dict):
    
    features = np.array(data["features"]).reshape(1, -1)
    
    prediction = model.predict(features)
    
    return {"prediction": int(prediction[0])}
