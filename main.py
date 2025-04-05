import requests
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import traceback
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Download the trained Random Forest model from GitHub Releases
def download_model():
    url = 'https://github.com/janreiroblox09/CropRecommendationModel/releases/download/rfc/rfc.pkl'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open("rfc.pkl", "wb") as f:
                f.write(response.content)
            print("✅ Model file downloaded successfully.")
        else:
            print(f"❌ Error downloading model: {response.status_code}")
    except Exception as e:
        print(f"❌ Error downloading model: {e}\n{traceback.format_exc()}")

# ✅ Load the model, label encoder, and scaler after download
def load_model():
    try:
        with open("rfc.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        print("✅ Random Forest Model Loaded Successfully!")
        return model
    except Exception as e:
        print(f"❌ Error Loading Model: {e}\n{traceback.format_exc()}")
        return None

# ✅ Load the label encoder
def load_label_encoder():
    try:
        with open("label_encoder.pkl", "rb") as encoder_file:
            label_encoder = pickle.load(encoder_file)
        print("✅ Label Encoder Loaded Successfully!")
        return label_encoder
    except Exception as e:
        print(f"❌ Error Loading Label Encoder: {e}\n{traceback.format_exc()}")
        return None

# ✅ Load the scaler
def load_scaler():
    try:
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        print("✅ Scaler Loaded Successfully!")
        return scaler
    except Exception as e:
        print(f"❌ Error Loading Scaler: {e}\n{traceback.format_exc()}")
        return None

# ✅ Load everything at the start
download_model()  # Download the model
model = load_model()
label_encoder = load_label_encoder()
scaler = load_scaler()

# ✅ Define Input Model (humidity removed)
class AveragesData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    rainfall: float
    soilPH: float

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
    """ Receive average data and predict top 3 crop recommendations """
    global latest_data, latest_recommendation

    try:
        print("✅ Received Average Data:", data)

        # ✅ Convert input to NumPy array
        input_data = np.array([[ 
            data.nitrogen, data.phosphorus, data.potassium,
            data.temperature, data.soilPH, data.rainfall
        ]])

        print(f"🟡 Raw Input Data: {input_data}")

        # ✅ Scale the input data
        scaled_input = scaler.transform(input_data)
        print(f"🟢 Scaled Input Data: {scaled_input}")

        # ✅ Predict probabilities
        probas = model.predict_proba(scaled_input)[0]

        # ✅ Get top 3 crop indices
        top_indices = np.argsort(probas)[-3:][::-1]

        # ✅ Map to crop names and confidence
        top_crops = [
            {
                "crop": label_encoder.inverse_transform([idx])[0],
                "confidence": round(probas[idx] * 100, 2)
            }
            for idx in top_indices
        ]

        # ✅ Store for root route
        latest_data = data.dict()
        latest_recommendation = top_crops[0]["crop"]

        return {"top_3_recommended_crops": top_crops}

    except Exception as e:
        print(f"❌ Error Receiving Averages: {e}\n{traceback.format_exc()}")
        return {"error": f"Failed to process averages: {e}"}

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

