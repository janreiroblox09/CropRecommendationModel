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

# Global variable to store the latest data and recommendation
latest_data = None
latest_recommendation = None

@app.get("/")
def read_root():
    # Print the latest data and recommendation
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
    """ Receive average data and predict crop recommendation """
    global latest_data, latest_recommendation  # Declare as global to update them

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

        # Store the received data and recommendation in global variables
        latest_data = data.dict()  # Convert to dictionary for easy display
        latest_recommendation = crop

        return {"recommended_crop": crop}

    except Exception as e:
        print(f"❌ Error Receiving Averages: {e}\n{traceback.format_exc()}")
        return {"error": f"Failed to process averages: {e}"}
