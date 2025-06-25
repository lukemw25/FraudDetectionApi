from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import os

app = FastAPI()

# load the trained model
model_path = "models/fraud_model.pkl"
if not os.path.exists(model_path):
    raise RuntimeError(f"model not found at {model_path}")

model = joblib.load(model_path)

# define input schema
class InputModel(BaseModel):
    features: list[float] = Field(..., min_length=29, max_length=29)

    @field_validator("features")
    @classmethod
    def validate_length(cls, v):
        if len(v) != 29:
            raise ValueError("exactly 29 features are required")
        return v

@app.get("/")
def read_root():
    return {"message": "fraud detection api is running"}

@app.post("/predict")
def predict(input: InputModel):
    try:
        features_array = np.array(input.features).reshape(1, -1)
        prediction = model.predict(features_array)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
