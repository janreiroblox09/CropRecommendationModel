from fastapi import FastAPI
import pickle
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load the trained Random Forest model
with open("rfc.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the label encoder (if used)
with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Google Sheets API Authentication
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("google_credentials.json", scope)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open("Soil_Data").sheet1  # Change this to your sheet name

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
    data = sheet.get_all_values()
    last_row = data[-1]  # Get the latest row
    return {"soil_data": last_row}

@app.post("/predict/")
def predict_crop(soil: SoilData):
    """ Predict Crop Recommendation """
    input_data = np.array([[soil.N, soil.P, soil.K, soil.temperature, soil.humidity, soil.pH, soil.rainfall]])
    prediction = model.predict(input_data)[0]
    crop = label_encoder.inverse_transform([prediction])[0]  # Convert back to crop name
    return {"recommended_crop": crop}
