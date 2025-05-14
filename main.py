from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
import traceback
import requests
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# GitHub URL for the model
model_url = "https://github.com/janreiroblox09/CropRecommendationModel/releases/download/rfc/rfc.pkl"
model_file_path = "rfc.pkl"

# Function to download the model file from GitHub
def download_model():
    try:
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_file_path, "wb") as f:
                f.write(response.content)
            print("✅ Model file downloaded successfully.")
        else:
            print(f"❌ Error downloading model: {response.status_code}")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")

# ✅ Load the trained Random Forest model
def load_model():
    try:
        with open(model_file_path, "rb") as model_file:
            model = pickle.load(model_file)
        print("✅ Random Forest Model Loaded Successfully!")
        return model
    except Exception as e:
        print(f"❌ Error Loading Model: {e}\n{traceback.format_exc()}")
        return None

# Download and load the model when the server starts
download_model()
model = load_model()

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
    rainfall: float
    soilPH: float

# ✅ Global variables to store the latest data and recommendation
latest_data = None
latest_recommendation = None
top_recommended_crops = []

@app.get("/")
def read_root():
    """ Root endpoint to check API status and return latest prediction """
    if latest_data and latest_recommendation:
        return {
            "message": "Crop Recommendation API is Running!",
            "latest_data": latest_data,
            "latest_recommendation": latest_recommendation,
            "top_10_recommendations": top_recommended_crops
        }
    else:
        return {"message": "Crop Recommendation API is Running! No data received yet."}

@app.post("/receive-averages/")
def receive_averages(data: AveragesData):
    """ Receive average data and predict top 10 crop recommendations """
    global latest_data, latest_recommendation, top_recommended_crops

    try:
        print("✅ Received Average Data:", data)

        # Convert input to NumPy array
        input_data = np.array([[ 
            data.nitrogen, data.phosphorus, data.potassium,
            data.temperature,  data.soilPH, data.rainfall
        ]])

        print(f"🟡 Raw Input Data: {input_data}")

        # ✅ Scale the input data
        scaled_input = scaler.transform(input_data)
        print(f"🟢 Scaled Input Data: {scaled_input}")

        # ✅ Predict probabilities for each class (crop)
        probas = model.predict_proba(scaled_input)[0]
        print(f"🟠 Predicted Probabilities: {probas}")

        # Get the top 10 crop indices (highest probabilities)
        top_indices = np.argsort(probas)[-10:][::-1]
        top_crops = []

        # Map the indices to crop names and confidence scores
        for idx in top_indices:
            crop = label_encoder.inverse_transform([idx])[0]
            confidence = round(probas[idx] * 100, 2)
            top_crops.append({"crop": crop, "confidence": confidence})

        print(f"✅ Top 10 Recommended Crops: {top_crops}")

        # Store the received data and recommendation
        latest_data = data.dict()
        latest_recommendation = top_crops[0]["crop"]
        top_recommended_crops.clear()
        top_recommended_crops.extend(top_crops)

        return {"top_10_recommended_crops": top_crops}

    except Exception as e:
        print(f"❌ Error Receiving Averages: {e}\n{traceback.format_exc()}")
        return {"error": f"Failed to process averages: {e}"}

# ✅ CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
