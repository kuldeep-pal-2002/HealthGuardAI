from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI(
    title="HealthGuardAI API",
    version="1.0"
)

# -----------------------------
# Load AI Model
# -----------------------------

model = load_model("models/healthguard.keras")
scaler = joblib.load("models/scaler.pkl")


# -----------------------------
# Request Model
# -----------------------------

class HealthInput(BaseModel):
    PM25: float
    PM10: float
    SO2: float
    NO2: float
    CO: float
    O3: float
    TEMP: float
    PRES: float
    DEWP: float
    RAIN: float
    WSPM: float


@app.get("/")
def home():
    return {"message": "HealthGuardAI Backend Running 🚀"}


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/predict")
def predict(data: HealthInput):

    values = np.array([[
        data.PM25,
        data.PM10,
        data.SO2,
        data.NO2,
        data.CO,
        data.O3,
        data.TEMP,
        data.PRES,
        data.DEWP,
        data.RAIN,
        data.WSPM
    ]])

    values = scaler.transform(values)

    prediction = model.predict(values, verbose=0)[0][0]

    prediction = float(prediction)

    if prediction < 25:
        level = "Low"

    elif prediction < 50:
        level = "Moderate"

    elif prediction < 75:
        level = "High"

    else:
        level = "Critical"

    return {
        "health_risk": round(prediction, 2),
        "risk_level": level
    }