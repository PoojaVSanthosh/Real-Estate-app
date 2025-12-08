import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from xgboost import Booster
import numpy as np



# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = "D:\labmetrix\India_household_price\india_housing_prices.csv"
MODEL_PATH = "xgboost_model.json"
VECTORIZER_PATH = "dict_vectorizer.pkl"
CURRENT_YEAR = 2025
GROWTH_RATE = 0.08       # 8% annual growth
YEARS_FWD = 5

# ---------------------------
# LOAD MODEL, VECTORIZER, DATA
# ---------------------------
@st.cache_resource
def load_model_and_vectorizer():
     # Load XGBoost JSON model
    booster = Booster()
    booster.load_model(MODEL_PATH)

    dv = joblib.load(VECTORIZER_PATH)
    return model, dv

# LOAD DATASET
# ---------------------------
@st.cache_data
def load_dataset():
    path = Path(DATA_PATH)
    if path.exists():
        return pd.read_csv(path)
    else:
        st.warning("Dataset not found at: " + DATA_PATH)
        return None

model, dv = load_model_and_vectorizer()
df = load_dataset()

# Pre-compute medians for investment logic
if df is not None:
    global_median_pps = df["Price_per_SqFt"].median()
    city_type_median_pps = (
        df.groupby(["City", "Property_Type"])["Price_per_SqFt"]
        .median()
        .reset_index()
    )
else:
    global_median_pps = None
    city_type_median_pps = None

# ---------------------------
# UI HEADER
# ---------------------------
st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

st.title("🏠 Real Estate Price Predictor & 5-Year Investment Advisor")
st.write("Fill in the property details to get:")
st.markdown("""
- 📌 **Current price estimate** (in Lakhs)  
- 📈 **Projected price after 5 years** (assuming 8% annual growth)  
- 💰 **Price per SqFt** vs market median  
- ✅ **Investment recommendation** (Good / Moderate / Risky)
""")

st.write("---")

# ---------------------------
# USER INPUT LAYOUT
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location & Basic Details")
    State = st.text_input("State")
    City = st.text_input("City")
    Property_Type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
    Availability_Status = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
    Owner_Type = st.selectbox("Owner Type", ["Owner", "Broker", "Builder"])
    BHK = st.number_input("BHK", 1, 10)
    Size_in_SqFt = st.number_input("Size (Sq Ft)", 300, 10000, 1200)

with col2:
    st.subheader("🏗️ Building & Surroundings")
    Year_Built = st.number_input("Year Built", 1970, CURRENT_YEAR, 2015)
    Floor_No = st.number_input("Floor Number", 0, 50, 1)
    Total_Floors = st.number_input("Total Floors", 1, 50, 10)
    Furnished_Status = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
    Facing = st.selectbox("Facing Direction", ["North", "South", "East", "West"])
    Nearby_Schools = st.number_input("Nearby Schools (count)", 0, 20, 5)
    Nearby_Hospitals = st.number_input("Nearby Hospitals (count)", 0, 20, 3)

st.subheader("🚍 Connectivity, Security & Amenities")
Public_Transport_Accessibility = st.selectbox(
    "Public Transport Accessibility",
    ["Low", "Medium", "High"]
)
Parking_Space = st.selectbox("Parking Space", ["Yes", "No"])
Security = st.selectbox("Security", ["Yes", "No"])

Amenities = st.multiselect(
    "Amenities",
    ["Gym", "Pool", "Garden", "Clubhouse", "Playground"]
)
Amenities_Count = len(Amenities)

st.write("---")
center_col = st.columns(3)[1]
with center_col:
    do_predict = st.button("🔍 Predict Price & 5-Year Projection", use_container_width=True)

# ---------------------------
# PREDICTION + INVESTMENT LOGIC
# ---------------------------
if do_predict:

    transport_map = {"Low": 1, "Medium": 2, "High": 3}

    input_dict = {
        "State": State.lower(),
        "City": City.lower(),
        "Property_Type": Property_Type,
        "BHK": BHK,
        "Size_in_SqFt": Size_in_SqFt,
        "Year_Built": Year_Built,
        "Furnished_Status": Furnished_Status,
        "Floor_No": Floor_No,
        "Total_Floors": Total_Floors,
        "Age_of_Property": CURRENT_YEAR - Year_Built,
        "Nearby_Schools": Nearby_Schools,
        "Nearby_Hospitals": Nearby_Hospitals,
        "Public_Transport_Accessibility": transport_map[Public_Transport_Accessibility],
        "Parking_Space": Parking_Space,
        "Security": Security,
        "Amenities": ", ".join(Amenities) if Amenities else "None",
        "Facing": Facing,
        "Owner_Type": Owner_Type,
        "Availability_Status": Availability_Status,
        "Amenities_Count": Amenities_Count,
    }

    # Encode features
    X_vec = dv.transform([input_dict])

    # Predict current price (Lakhs)
    current_price = float(model.predict(X_vec)[0])

    # Price per SqFt (in ₹)
    price_per_sqft = (current_price * 100000) / Size_in_SqFt if Size_in_SqFt > 0 else None

    # 5-year future price using compound interest formula
    future_price = current_price * (1 + GROWTH_RATE) ** YEARS_FWD

    # ---------------------------
    # INVESTMENT CLASSIFICATION
    # ---------------------------
    # Default medians
    if df is not None and price_per_sqft is not None:
        # Try city + property-type median first
        row = city_type_median_pps[
            (city_type_median_pps["City"].str.lower() == City.lower()) &
            (city_type_median_pps["Property_Type"] == Property_Type)
        ]
        if len(row) > 0:
            median_pps = float(row["Price_per_SqFt"].iloc[0])
        else:
            median_pps = float(global_median_pps)
    else:
        median_pps = None

    # Multi-factor investment score
    investment_score = 0
    if BHK >= 3:
        investment_score += 1
    if Availability_Status == "Ready_to_Move":
        investment_score += 1
    if Public_Transport_Accessibility in ["Medium", "High"]:
        investment_score += 1
    if Amenities_Count >= 3:
        investment_score += 1

    # Compare with median price_per_sqft (if available)
    if median_pps is not None and price_per_sqft is not None:
        price_ratio = price_per_sqft / median_pps
    else:
        price_ratio = 1.0  # neutral

    # Decision logic
    if price_ratio <= 0.9 and investment_score >= 2:
        tag = "🟢 Excellent Investment"
        desc = "Below market rate per sq.ft with strong property features. High ROI potential."
        risk_label = "Good"
    elif price_ratio <= 1.1 and investment_score >= 1:
        tag = "🟡 Moderate Investment"
        desc = "Close to market price with decent features. Suitable for long-term holding."
        risk_label = "Moderate"
    else:
        tag = "🔴 Risky Investment"
        desc = "Above market price or weak fundamentals. Consider negotiating or exploring alternatives."
        risk_label = "Risky"

    # ---------------------------
    # DISPLAY RESULTS
    # ---------------------------
    st.write("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div style="background-color:#eef7ff;padding:18px;border-radius:12px;border-left:6px solid #1a73e8;">
            <h3>💰 Current Price Estimate</h3>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h2 style='color:#1a73e8;'>₹ {current_price:.2f} Lakhs</h2>",
            unsafe_allow_html=True
        )
        if price_per_sqft is not None:
            st.markdown(
                f"<p><b>Approx. ₹ {price_per_sqft:,.0f} per SqFt</b></p>",
                unsafe_allow_html=True
            )
        if median_pps is not None and price_per_sqft is not None:
            st.markdown(
                f"<p>Market median (city/type): ₹ {median_pps:,.0f} per SqFt</p>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            """
            <div style="background-color:#e9f9f1;padding:18px;border-radius:12px;border-left:6px solid #16a34a;">
            <h3>📈 Projected Price in 5 Years</h3>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h2 style='color:#16a34a;'>₹ {future_price:.2f} Lakhs</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p>Assuming <b>{GROWTH_RATE*100:.0f}%</b> annual growth over <b>{YEARS_FWD} years</b>.</p>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")
    st.subheader("📊 Investment Recommendation")
    st.markdown(
        f"""
        <div style="background-color:#fff7e6;padding:18px;border-radius:12px;border-left:6px solid #f59e0b;">
        <h3>{tag}</h3>
        <p style="font-size:16px;">{desc}</p>
        <p><b>Investment score:</b> {investment_score} / 4</p>
        </div>
        """,
        unsafe_allow_html=True
    )
