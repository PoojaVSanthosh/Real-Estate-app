import streamlit as st 
import pandas as pd
import joblib
import numpy as np
import os
import requests

# ============================================================
# CONFIG
# ============================================================
MODEL_PKL = "xgboost_model.pkl"
VECTORIZER_PKL = "dict_vectorizer.pkl"

# Google Drive file ID for your dataset
GDRIVE_FILE_ID = "14oj5GU0ulIlE_A__4Uewi6urstiU8qAU"  # <- change if different
LOCAL_CSV_PATH = "india_housing_prices.csv"

CURRENT_YEAR = 2025
GROWTH_RATE = 0.08
YEARS_FWD = 5


# ============================================================
# GOOGLE DRIVE DOWNLOADER (WORKS FOR LARGE FILES)
# ============================================================
def download_from_gdrive(file_id: str, dest_path: str) -> str:
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)

    def get_confirm_token(resp):
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    token = get_confirm_token(response)

    if token is not None:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    return dest_path


# ============================================================
# LOAD DATASET FROM GOOGLE DRIVE (CACHED)
# ============================================================
@st.cache_data
def load_dataset_from_drive():
    if not os.path.exists(LOCAL_CSV_PATH):
        download_from_gdrive(GDRIVE_FILE_ID, LOCAL_CSV_PATH)

    df = pd.read_csv(LOCAL_CSV_PATH)

    # Derived feature
    if "Price_per_SqFt" not in df.columns:
        df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]

    return df


# ============================================================
# LOAD MODEL & VECTORIZER (PKL ONLY)
# ============================================================
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(MODEL_PKL)
    dv = joblib.load(VECTORIZER_PKL)
    return model, dv


model, dv = load_model_and_vectorizer()
df = load_dataset_from_drive()


# ============================================================
# BUILD STATE → CITY MAPPING FOR UI (TITLE CASE FOR DISPLAY)
# ============================================================
states = sorted(df["State"].dropna().str.title().unique())

state_city_map = (
    df.groupby(df["State"].str.title())["City"]
    .apply(lambda x: sorted(x.dropna().str.title().unique()))
    .to_dict()
)


# ============================================================
# MEDIAN CALCULATIONS
# ============================================================
global_median_pps = df["Price_per_SqFt"].median()

city_type_median_pps = df.groupby(
    ["City", "Property_Type"]
)["Price_per_SqFt"].median().reset_index()


# ============================================================
# UI LAYOUT
# ============================================================
st.set_page_config(page_title="Real Estate Advisor", layout="wide")
st.title("🏠 Real Estate Price Predictor & 5-Year Investment Advisor")
st.write("---")

col1, col2 = st.columns(2)

# ============================================================
# LOCATION & BASIC DETAILS PANEL
# ============================================================
with col1:
    st.subheader("📍 Location & Basic Details")

    # STATE
    State = st.selectbox("Select State", ["Select State"] + states)

    # CITY — DYNAMIC
    if State != "Select State":
        City = st.selectbox("Select City", state_city_map.get(State, []))
    else:
        City = st.selectbox("Select City", ["Select a State first"])

    Property_Type = st.selectbox(
        "Property Type", ["Apartment", "Independent House", "Villa"]
    )
    Availability_Status = st.selectbox(
        "Availability", ["Ready_to_Move", "Under_Construction"]
    )
    Owner_Type = st.selectbox("Owner Type", ["Owner", "Broker", "Builder"])
    BHK = st.number_input("BHK", 1, 10)
    Size_in_SqFt = st.number_input("Size (SqFt)", 300, 10000, 1200)

# ============================================================
# BUILDING DETAILS PANEL
# ============================================================
with col2:
    st.subheader("🏗️ Building Details")
    Year_Built = st.number_input("Year Built", 1970, CURRENT_YEAR, 2015)
    Floor_No = st.number_input("Floor Number", 0, 50, 1)
    Total_Floors = st.number_input("Total Floors", 1, 50, 10)
    Furnished_Status = st.selectbox(
        "Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"]
    )
    Facing = st.selectbox("Facing Direction", ["North", "South", "East", "West"])
    Nearby_Schools = st.number_input("Nearby Schools", 0, 20, 5)
    Nearby_Hospitals = st.number_input("Nearby Hospitals", 0, 20, 3)

# ============================================================
# AMENITIES & CONNECTIVITY
# ============================================================
st.subheader("🚍 Connectivity & Amenities")

Public_Transport_Accessibility = st.selectbox(
    "Transport Accessibility", ["Low", "Medium", "High"]
)
Parking_Space = st.selectbox("Parking Space", ["Yes", "No"])
Security = st.selectbox("Security", ["Yes", "No"])

Amenities = st.multiselect(
    "Amenities", ["clubhouse", "garden", "gym", "pool", "playground"]
)
Amenities_Count = len(Amenities)

st.write("---")
predict_btn = st.button(
    "🔍 Predict Price & 5-Year Projection", use_container_width=True
)

# ============================================================
# PREDICTION LOGIC USING PKL MODEL
# ============================================================
if predict_btn:

    if State == "Select State" or City.startswith("Select"):
        st.error("Please select a valid State and City.")
        st.stop()

    # Map transport to numeric (must match training)
    transport_map = {"Low": 1, "Medium": 2, "High": 3}

    # BUILD AMENITIES STRING EXACTLY LIKE TRAINING EXPECTS:
    # - lowercase
    # - sorted
    # - comma+space between
    if Amenities:
        amenities_str = ", ".join(sorted([a.lower() for a in Amenities]))
    else:
        amenities_str = "none"

    # Build input dictionary for DictVectorizer
    input_dict = {
        "State": State.lower(),          # dv feature: State=kerala
        "City": City.lower(),            # dv feature: City=kochi etc.
        "Property_Type": Property_Type,  # must match training values
        "BHK": BHK,
        "Size_in_SqFt": Size_in_SqFt,
        "Year_Built": Year_Built,
        "Age_of_Property": CURRENT_YEAR - Year_Built,
        "Furnished_Status": Furnished_Status,
        "Floor_No": Floor_No,
        "Total_Floors": Total_Floors,
        "Nearby_Schools": Nearby_Schools,
        "Nearby_Hospitals": Nearby_Hospitals,
        "Public_Transport_Accessibility": transport_map[
            Public_Transport_Accessibility
        ],
        "Parking_Space": Parking_Space,
        "Security": Security,
        "Amenities": amenities_str,
        "Facing": Facing,
        "Owner_Type": Owner_Type,
        "Availability_Status": Availability_Status,
        "Amenities_Count": Amenities_Count,
    }

    # Encode + Predict
    X_vec = dv.transform([input_dict])
    current_price = float(model.predict(X_vec)[0])

    # Derived metrics
    price_per_sqft = (current_price * 100000) / Size_in_SqFt
    future_price = current_price * (1 + GROWTH_RATE) ** YEARS_FWD

    # Market comparison
    row = city_type_median_pps[
        (city_type_median_pps["City"].str.lower() == City.lower())
        & (city_type_median_pps["Property_Type"] == Property_Type)
    ]
    median_pps = float(row["Price_per_SqFt"].iloc[0]) if len(row) else global_median_pps

    # Investment scoring
    investment_score = (
        (BHK >= 3)
        + (Availability_Status == "Ready_to_Move")
        + (Public_Transport_Accessibility in ["Medium", "High"])
        + (Amenities_Count >= 3)
    )

    price_ratio = price_per_sqft / median_pps if median_pps else 1.0

    if price_ratio <= 0.9 and investment_score >= 2:
        tag, desc = "🟢 Excellent Investment", "Below market value, strong ROI potential."
    elif price_ratio <= 1.1 and investment_score >= 1:
        tag, desc = "🟡 Moderate Investment", "Fair pricing and decent fundamentals."
    else:
        tag, desc = "🔴 Risky Investment", "Overpriced or weak fundamentals."

    # ============================================================
    # OUTPUT RESULTS
    # ============================================================
    st.success(f"💰 **Estimated Current Price:** ₹ {current_price:.2f} Lakhs")
    st.info(f"📈 **Projected Price in 5 Years:** ₹ {future_price:.2f} Lakhs")
    st.write(f"🏷️ **Price per SqFt:** ₹ {price_per_sqft:,.0f}")

    if median_pps:
        st.write(f"📌 **Market Median:** ₹ {median_pps:,.0f} per SqFt")

    st.subheader("📊 Investment Recommendation")
    st.markdown(f"### {tag}")
    st.write(desc)
    st.write(f"**Investment Score:** {investment_score} / 4")
