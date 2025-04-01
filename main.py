from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
import traceback

app = FastAPI()

# Load the trained Random Forest model
try:
    with open("rfc.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    print("✅ Random Forest Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}\n{traceback.format_exc()}")

# Load the label encoder
try:
    with open("label_encoder.pkl", "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    print("✅ Label Encoder Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Label Encoder: {e}\n{traceback.format_exc()}")

# Define Input Model for receiving average data
class AveragesData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    rainfall: float
    soilPH: float

@app.get("/")
def read_root():
    return {"message": "Crop Recommendation API is Running!"}

@app.post("/receive-averages/")
def receive_averages(data: AveragesData):
    """ Receive average data and predict crop recommendation """
    try:
        print("✅ Received Average Data:", data)

        input_data = np.array([[
            data.nitrogen, data.phosphorus, data.potassium,
            data.temperature, data.humidity, data.soilPH, data.rainfall
        ]])

        # Predict crop
        prediction = model.predict(input_data)[0]
        crop = label_encoder.inverse_transform([prediction])[0]

        return {"recommended_crop": crop}

    except Exception as e:
        print(f"❌ Error Receiving Averages: {e}\n{traceback.format_exc()}")
        return {"error": f"Failed to process averages: {e}"}
