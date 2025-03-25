from fastapi import FastAPI
import pickle
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI()

# Load the trained Random Forest model
try:
    with open("rfc.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("❌ ERROR: 'rfc.pkl' not found. Make sure the model file is uploaded.")
    model = None

# Load the label encoder (if used)
try:
    with open("label_encoder.pkl", "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
except FileNotFoundError:
    print("⚠️ Warning: 'label_encoder.pkl' not found. Predictions may not work.")
    label_encoder = None

# Google Sheets API Authentication
try:
    if os.path.exists("google_credentials.json"):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("google_credentials.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("Soil_Data").sheet1  # Change this to your sheet name
    else:
        print("⚠️ Warning: 'google_credentials.json' not found. Google Sheets integration won't work.")
        client = None
        sheet = None
except Exception as e:
    print(f"❌ ERROR: Google Sheets authentication failed: {e}")
    client = None
    sheet = None

# Define input model
class SoilData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    pH: float
    rainfall: float

@app.get("/")
def read_root():
    return {"message": "Crop Recommendation API is Running!"}

@app.get("/fetch-data/")
def fetch_data():
    """ Fetch latest row from Google Sheets """
    if sheet:
        data = sheet.get_all_values()
        last_row = data[-1]  # Get the latest row
        return {"soil_data": last_row}
    return {"error": "Google Sheets connection is not available."}

@app.post("/predict/")
def predict_crop(soil: SoilData):
    """ Predict Crop Recommendation """
    if model is None or label_encoder is None:
        return {"error": "Model or label encoder is missing. Prediction not available."}

    input_data = np.array([[soil.N, soil.P, soil.K, soil.temperature, soil.humidity, soil.pH, soil.rainfall]])
    prediction = model.predict(input_data)[0]
    crop = label_encoder.inverse_transform([prediction])[0]  # Convert back to crop name
    return {"recommended_crop": crop}
