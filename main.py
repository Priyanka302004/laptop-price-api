from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI(title="Laptop Price Prediction API")

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "API is running successfully 🚀"}

@app.post("/predict")
def predict(ram: int, storage: int, cpu: int):
    data = np.array([[ram, storage, cpu]])
    prediction = model.predict(data)
    return {"predicted_price": int(prediction[0])}
