from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
import requests
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

# Raspberry Pi URL
raspberry_pi_url = "http://192.168.1.10:5000/receive-data"  # Modify this to match your Flask endpoint

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

        print(f"🟡 Input Data for Model: {input_data}")

        # Predict crop
        prediction = model.predict(input_data)[0]  # Check if this runs
        print(f"🎯 Predicted Label (Encoded): {prediction}")

        crop = label_encoder.inverse_transform([prediction])[0]  # Check if this runs
        print(f"✅ Final Recommended Crop: {crop}")

        # Send data to Raspberry Pi
        data_to_send = {
            "nitrogen": data.nitrogen,
            "phosphorus": data.phosphorus,
            "potassium": data.potassium,
            "temperature": data.temperature,
            "humidity": data.humidity,
            "rainfall": data.rainfall,
            "soilPH": data.soilPH,
            "recommended_crop": crop
        }

        # Sending data to Raspberry Pi
        try:
            response = requests.post(raspberry_pi_url, json=data_to_send)
            if response.status_code == 200:
                print("✅ Data sent successfully to Raspberry Pi!")
            else:
                print(f"❌ Failed to send data to Raspberry Pi. HTTP Status Code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error sending data to Raspberry Pi: {e}")

        return {"recommended_crop": crop}

    except Exception as e:
        print(f"❌ Error Receiving Averages: {e}\n{traceback.format_exc()}")
        return {"error": f"Failed to process averages: {e}"}
