from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
import traceback
import requests  # ✅ For sending data to Raspberry Pi

app = FastAPI()

# ✅ Load the trained Random Forest model
try:
    with open("rfc.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    print("✅ Random Forest Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}\n{traceback.format_exc()}")

# ✅ Load the label encoder
try:
    with open("label_encoder.pkl", "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    print("✅ Label Encoder Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Label Encoder: {e}\n{traceback.format_exc()}")

# ✅ Load the scaler
try:
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    print("✅ Scaler Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Scaler: {e}\n{traceback.format_exc()}")

# ✅ Define Input Model
class AveragesData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    rainfall: float
    soilPH: float

# ✅ Raspberry Pi Flask Server URL
RPI_URL = "http://192.168.1.101:5000/receive-data"

# ✅ Global variable to store the latest data and recommendation
latest_data = None
latest_recommendation = None

@app.get("/")
def read_root():
    """ Root endpoint to check API status """
    if latest_data and latest_recommendation:
        return {
            "message": "Crop Recommendation API is Running!",
            "latest_data": latest_data,
            "latest_recommendation": latest_recommendation,
        }
    else:
        return {"message": "Crop Recommendation API is Running! No data received yet."}

@app.post("/receive-averages/")
def receive_averages(data: AveragesData):
    """ Receive soil data, predict crop recommendation, and send it to Raspberry Pi """
    global latest_data, latest_recommendation

    try:
        print("✅ Received Average Data:", data)

        # Convert input to NumPy array
        input_data = np.array([[ 
            data.nitrogen, data.phosphorus, data.potassium,
            data.temperature, data.humidity, data.soilPH, data.rainfall
        ]])

        print(f"🟡 Raw Input Data: {input_data}")

        # ✅ Scale the input data
        scaled_input = scaler.transform(input_data)
        print(f"🟢 Scaled Input Data: {scaled_input}")

        # ✅ Predict crop
        prediction = model.predict(scaled_input)[0]  
        print(f"🎯 Predicted Label (Encoded): {prediction}")

        crop = label_encoder.inverse_transform([prediction])[0]  
        print(f"✅ Final Recommended Crop: {crop}")

        # ✅ Store the received data and recommendation
        latest_data = data.dict()
        latest_recommendation = crop

        # ✅ Send the recommendation to Raspberry Pi
        payload = {"recommended_crop": crop}
        response = requests.post(RPI_URL, json=payload)
        print(f"📡 Sent to Raspberry Pi: {response.text}")

        return {"recommended_crop": crop}

    except Exception as e:
        print(f"❌ Error Receiving Averages: {e}\n{traceback.format_exc()}")
        return {"error": f"Failed to process averages: {e}"}
