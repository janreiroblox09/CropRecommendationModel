from fastapi import FastAPI
import pickle
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from pydantic import BaseModel
import os


app = FastAPI()

# ✅ Load the trained Random Forest model
try:
    with open("rfc.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    print("✅ Random Forest Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")

# ✅ Load the label encoder
try:
    with open("label_encoder.pkl", "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    print("✅ Label Encoder Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Label Encoder: {e}")

# ✅ Google Sheets API Authentication
GOOGLE_CREDENTIALS_PATH = "/etc/secrets/google_credentials.json"
print("✅ Checking for Google Credentials:", os.path.exists(GOOGLE_CREDENTIALS_PATH))

try:
    if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        raise FileNotFoundError("google_credentials.json not found in secrets!")

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIALS_PATH, scope)
    client = gspread.authorize(creds)
    sheet = client.open("Data").sheet1  # Ensure the sheet name is correct

    print("✅ Google Sheets Authentication Successful!")
except Exception as e:
    print(f"❌ ERROR: Google Sheets Authentication Failed: {e}")

# ✅ Define Input Model
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
    try:
        data = sheet.get_all_values()
        last_row = data[-1]  # Get the latest row
        return {"soil_data": last_row}
    except Exception as e:
        return {"error": f"Failed to fetch data: {e}"}

@app.post("/predict/")
def predict_crop(soil: SoilData):
    """ Predict Crop Recommendation """
    input_data = np.array([[soil.N, soil.P, soil.K, soil.temperature, soil.humidity, soil.pH, soil.rainfall]])
    prediction = model.predict(input_data)[0]
    crop = label_encoder.inverse_transform([prediction])[0]  # Convert back to crop name
    return {"recommended_crop": crop}
