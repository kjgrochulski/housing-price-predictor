import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import shap 

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Melbourne House Price Predictor",
    page_icon="",
    layout="centered"
)

MELBOURNE_CBD = (-37.8136, 144.9631)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("melbourne_model.pkl")
    with open("model_features.json") as f:
        features = json.load(f)
    return model, features

model, feature_cols = load_model()

@st.cache_resource
def load_explainer(_model):
    regressor = _model.named_steps["regressor"]
    return shap.TreeExplainer(regressor)

explainer = load_explainer(model)

SUBURB_PROPERTYCOUNT = {
    "Abbotsford": 4019, "Aberfeldie": 1543, "Airport West": 3464, "Albanvale": 1899,
    "Albert Park": 3280, "Albion": 2185, "Alphington": 2211, "Altona": 5301,
    "Altona Meadows": 7630, "Altona North": 5132, "Ardeer": 1281, "Armadale": 4836,
    "Ascot Vale": 6567, "Ashburton": 3052, "Ashwood": 2894, "Aspendale": 2824,
    "Aspendale Gardens": 2243, "Attwood": 1130, "Avondale Heights": 4502,
    "Bacchus Marsh": 2871, "Balaclava": 2952, "Balwyn": 5682, "Balwyn North": 7809,
    "Bayswater": 5030, "Bayswater North": 3598, "Beaconsfield": 2332,
    "Beaumaris": 5366, "Bellfield": 790, "Bentleigh": 6795, "Bentleigh East": 10969,
    "Berwick": 17093, "Black Rock": 2866, "Blackburn": 5713, "Blackburn North": 2867,
    "Blackburn South": 4387, "Bonbeach": 2887, "Boronia": 9704, "Box Hill": 4605,
    "Braybrook": 3589, "Briar Hill": 1390, "Brighton": 10579, "Brighton East": 6938,
    "Broadmeadows": 4294, "Brooklyn": 962, "Brunswick": 11918, "Brunswick East": 5533,
    "Brunswick West": 7082, "Bulleen": 4480, "Bullengarook": 249, "Bundoora": 10175,
    "Burnley": 438, "Burnside": 1607, "Burnside Heights": 1686, "Burwood": 5678,
    "Burwood East": 4048, "Cairnlea": 2674, "Camberwell": 8920, "Campbellfield": 1889,
    "Canterbury": 3265, "Carlton": 6786, "Carlton North": 3106, "Carnegie": 7822,
    "Caroline Springs": 7719, "Carrum": 1989, "Carrum Downs": 8060, "Caulfield": 2379,
    "Caulfield East": 608, "Caulfield North": 6923, "Caulfield South": 5051,
    "Chadstone": 3582, "Chelsea": 3906, "Chelsea Heights": 2076, "Cheltenham": 9758,
    "Chirnside Park": 3789, "Clarinda": 2727, "Clayton": 5837, "Clayton South": 4734,
    "Clifton Hill": 2954, "Coburg": 11204, "Coburg North": 3445, "Collingwood": 4553,
    "Coolaroo": 1124, "Craigieburn": 15510, "Cranbourne": 7680, "Cranbourne North": 6464,
    "Cremorne": 1123, "Croydon": 11925, "Croydon Hills": 1705, "Croydon North": 2985,
    "Croydon South": 1863, "Dallas": 2246, "Dandenong": 10894, "Dandenong North": 8322,
    "Deepdene": 892, "Deer Park": 6388, "Delahey": 2898, "Derrimut": 2276,
    "Diamond Creek": 4258, "Diggers Rest": 1184, "Dingley Village": 3940,
    "Docklands": 4707, "Doncaster": 9028, "Doncaster East": 10999, "Donvale": 4790,
    "Doreen": 7254, "Doveton": 3533, "Eaglemont": 1651, "East Melbourne": 3040,
    "Edithvale": 2546, "Elsternwick": 4898, "Eltham": 6990, "Eltham North": 2346,
    "Elwood": 8989, "Endeavour Hills": 8443, "Epping": 10926, "Essendon": 9264,
    "Essendon North": 1308, "Essendon West": 588, "Fairfield": 2970, "Fawkner": 5070,
    "Ferntree Gully": 10788, "Fitzroy": 5825, "Fitzroy North": 6244, "Flemington": 3593,
    "Footscray": 7570, "Forest Hill": 4385, "Frankston": 17055, "Frankston North": 2500,
    "Frankston South": 7566, "Gardenvale": 534, "Gisborne": 3376, "Gladstone Park": 3285,
    "Glen Huntly": 2403, "Glen Iris": 10412, "Glen Waverley": 15321, "Glenroy": 8870,
    "Gowanbrae": 1071, "Greensborough": 8524, "Greenvale": 4864, "Hadfield": 2606,
    "Hallam": 3728, "Hampton": 5454, "Hampton East": 2356, "Hampton Park": 8256,
    "Hawthorn": 11308, "Hawthorn East": 6482, "Healesville": 3307, "Heathmont": 3794,
    "Heidelberg": 2890, "Heidelberg Heights": 2947, "Heidelberg West": 2674,
    "Highett": 4794, "Hillside": 5556, "Hoppers Crossing": 13830, "Hughesdale": 3145,
    "Huntingdale": 768, "Hurstbridge": 1345, "Ivanhoe": 5549, "Ivanhoe East": 1554,
    "Jacana": 851, "Kealba": 1202, "Keilor": 2339, "Keilor Downs": 3656,
    "Keilor East": 5629, "Keilor Lodge": 570, "Keilor Park": 1119, "Kensington": 5263,
    "Kew": 10331, "Kew East": 2671, "Keysborough": 8459, "Kilsyth": 4654,
    "Kings Park": 2878, "Kingsbury": 1414, "Kingsville": 1808, "Knoxfield": 2949,
    "Kooyong": 394, "Kurunjang": 3553, "Lalor": 8279, "Langwarrin": 8743,
    "Lower Plenty": 1624, "Maidstone": 3873, "Malvern": 4675, "Malvern East": 8801,
    "Maribyrnong": 4918, "McKinnon": 2397, "Meadow Heights": 4704, "Melbourne": 17496,
    "Melton": 3600, "Melton South": 4718, "Melton West": 6065, "Mentone": 6162,
    "Mernda": 5812, "Middle Park": 2019, "Mill Park": 10529, "Mitcham": 6871,
    "Monbulk": 1424, "Mont Albert": 2079, "Montmorency": 3891, "Montrose": 2493,
    "Moonee Ponds": 6232, "Moorabbin": 2555, "Mooroolbark": 8280, "Mordialloc": 3650,
    "Mount Evelyn": 3532, "Mount Waverley": 13366, "Mulgrave": 7113, "Murrumbeena": 4442,
    "Narre Warren": 9376, "New Gisborne": 849, "Newport": 5498, "Niddrie": 2291,
    "Noble Park": 11806, "North Melbourne": 6821, "North Warrandyte": 1058,
    "Northcote": 11364, "Notting Hill": 902, "Nunawading": 4973, "Oak Park": 2651,
    "Oakleigh": 3224, "Oakleigh East": 2547, "Oakleigh South": 3692, "Officer": 2768,
    "Ormond": 3578, "Pakenham": 17384, "Parkdale": 5087, "Parkville": 2309,
    "Pascoe Vale": 7485, "Point Cook": 15542, "Port Melbourne": 8648, "Prahran": 7717,
    "Preston": 14577, "Princes Hill": 1008, "Reservoir": 21650, "Richmond": 14949,
    "Riddells Creek": 1475, "Ringwood": 7785, "Ringwood East": 4407,
    "Ringwood North": 3619, "Ripponlea": 821, "Rosanna": 3540, "Rowville": 11667,
    "Roxburgh Park": 5833, "Sandhurst": 1721, "Sandringham": 4497, "Scoresby": 2206,
    "Seabrook": 1793, "Seaford": 8077, "Seaholme": 852, "Seddon": 2417, "Silvan": 457,
    "Skye": 2756, "South Kingsville": 984, "South Melbourne": 5943, "South Morang": 7969,
    "South Yarra": 14887, "Southbank": 8400, "Spotswood": 1223, "Springvale": 7412,
    "Springvale South": 4054, "St Albans": 14042, "St Helena": 915, "St Kilda": 13240,
    "Strathmore": 3284, "Strathmore Heights": 389, "Sunbury": 14092, "Sunshine": 3755,
    "Sunshine North": 4217, "Sunshine West": 6763, "Surrey Hills": 5457,
    "Sydenham": 3640, "Tarneit": 10160, "Taylors Hill": 4242, "Taylors Lakes": 5336,
    "Templestowe": 6202, "Templestowe Lower": 5420, "The Basin": 1690,
    "Thomastown": 7955, "Thornbury": 8870, "Toorak": 7217, "Travancore": 1052,
    "Truganina": 5811, "Tullamarine": 3296, "Vermont": 4181, "Vermont South": 4280,
    "Viewbank": 2698, "Wallan": 3988, "Wantirna": 5424, "Wantirna South": 7082,
    "Warrandyte": 2003, "Waterways": 709, "Watsonia": 2329, "Watsonia North": 1442,
    "Werribee": 16166, "West Footscray": 5058, "West Melbourne": 2230,
    "Westmeadows": 2474, "Wheelers Hill": 7392, "Whittlesea": 2170,
    "Williams Landing": 1999, "Williamstown": 6380, "Williamstown North": 802,
    "Windsor": 4380, "Wollert": 2940, "Wyndham Vale": 5262, "Yallambie": 1369,
    "Yarra Glen": 1160, "Yarraville": 6543,
}
PROPERTYCOUNT_DEFAULT = 6555  # fallback to median

KNOWN_COUNCILS = {
    "Banyule", "Bayside", "Boroondara", "Brimbank", "Cardinia", "Casey",
    "Darebin", "Frankston", "Glen Eira", "Greater Dandenong", "Hobsons Bay",
    "Hume", "Kingston", "Knox", "Macedon Ranges", "Manningham", "Maribyrnong",
    "Maroondah", "Melbourne", "Melton", "Monash", "Moonee Valley", "Moorabool",
    "Moreland", "Nillumbik", "Port Phillip", "Stonnington", "Whitehorse",
    "Whittlesea", "Wyndham", "Yarra", "Yarra Ranges",
}
 
def normalise_council(raw_council: str) -> str:
    import re
    stripped = re.sub(
        r"^(City of|Shire of|Borough of|Rural City of|Municipal Council of|Council of)\s*",
        "", raw_council, flags=re.IGNORECASE
    ).strip()
    return stripped if stripped in KNOWN_COUNCILS else raw_council


# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        }
        .main { max-width: 400px; margin: auto; }
            
        button.step-up { display: none !important; }
        button.step-down { display: none !important; }
            
        div[data-baseweb="input"] { border-radius: 6px !important; }
    
        .result-box {
            background: #f0fdf4;
            border: 2px solid #22c55e;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            margin-top: 16px;
        }
        .result-price {
            font-size: 2.4rem;
            font-weight: 700;
            color: #15803d;
        }
        .warning-box {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #78350f;
        }
    </style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.title("Melbourne House Price Predictor")
st.caption(
    "Enter a Melbourne address and property details to get an estimated sale price.  \n"
    "[Source Code and Model Training Pipeline on GitHub](https://github.com/kjgrochulski/Housing-Price-Predictor)"
)
st.divider()


# ── Address Input ─────────────────────────────────────────────────────────────

address = st.text_input(
    "Property Address",
    placeholder="e.g. 45 Collins St, Melbourne VIC 3000"
)

lat, lon, suburb, postcode, council, distance = None, None, "", "", "", None

if address:
    with st.spinner("Looking up address..."):
        try:
            geolocator = Nominatim(user_agent="melb_house_predictor_v1")
            time.sleep(1)                                                   # Nominatim rate limit
            location = geolocator.geocode(
                address + ", Victoria, Australia",
                addressdetails=True
            )

            if location:
                raw = location.raw.get("address", {})
                lat = location.latitude
                lon = location.longitude
                suburb = (
                    raw.get("suburb")
                    or raw.get("city_district")
                    or raw.get("town")
                    or raw.get("village")
                    or ""
                )
                postcode = str(raw.get("postcode", ""))
                council = normalise_council(raw.get("county", raw.get("state_district", "")))
                distance = round(geodesic(MELBOURNE_CBD, (lat, lon)).km, 2)

                st.success(f"Found: **{suburb}**, {postcode}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Distance from CBD", f"{distance} km")
                c2.metric("Latitude", round(lat, 4))
                c3.metric("Longitude", round(lon, 4))
                c4.metric("Postcode", postcode)
            else:
                st.error("Address not found. Try adding suburb and state, e.g. 'Richmond VIC'.")
        except Exception as e:
            st.error(f"Geocoding error: {e}")

# all we need from above geolocation is lat, lon, postcode, distance (from CBD)
# additionally we need user input for internal area, year built, plot size, rooms 

st.divider()


# ── Property Details ──────────────────────────────────────────────────────────

st.subheader("Property Details")
st.caption("Fill in the property details below. Location fields are auto-filled from the address.")

col1, col2 = st.columns(2)

with col1:
    prop_type = st.selectbox(
        "Property Type",
        options=["House", "Unit/Apartment", "Townhouse"],
        # format_func=lambda x: {"h": "House", "t": "Townhouse", "u": "Unit"}[x]
    )
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)

with col2:
    car_spaces = st.number_input(
    "Car Spaces", 
    min_value=0, 
    max_value=20, 
    value=None,  # This allows the field to be empty
    placeholder="Leave blank if unknown",
)
    landsize = st.number_input("Plot Size (m²)", min_value=0, max_value=100000, value=300) 

    suburb_input = st.text_input(
        "Suburb (auto-filled)",
        value=suburb,
        help="Auto-filled from address. Edit if incorrect."
    )   
    propertycount = SUBURB_PROPERTYCOUNT.get(suburb_input or suburb, PROPERTYCOUNT_DEFAULT)


# ── Predict ───────────────────────────────────────────────────────────────────

st.divider()

predict_ready = lat is not None or (suburb_input and postcode)

if st.button("Predict Price", type="primary", use_container_width=True, disabled=not predict_ready):
    if lat is None:
        st.warning("Please enter a valid address first.")
    else:
        try:
            input_df = pd.DataFrame([{
                "Suburb": suburb_input or suburb,
                "Type": prop_type,
                "Distance": distance,
                "Postcode": postcode,
                "Landsize": landsize,
                "CouncilArea": council, # or formerly council_input,
                "Lattitude": lat,
                "Longtitude": lon,
                "Propertycount": propertycount,
                "Bedroom2": bedrooms,
                "Bathroom": bathrooms,
                "Car": np.nan if car_spaces is None else car_spaces,
            }])

            prediction = np.expm1(model.predict(input_df)[0])

            # Transform input using preprocessor
            X_processed = model.named_steps["preprocessor"].transform(input_df)

            # Get feature names AFTER preprocessing
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()

            # Compute SHAP values
            shap_values = explainer.shap_values(X_processed)

           # Convert SHAP values from log space → price space
            prediction_log = model.predict(input_df)[0]
            prediction_price = np.expm1(prediction_log)

            shap_impacts = shap_values[0]

            # convert each feature impact to price contribution
            price_impacts = np.expm1(prediction_log + shap_impacts) - prediction_price

            shap_df = pd.DataFrame({
                "feature": feature_names,
                "impact": price_impacts
            })

            # Sort by importance
            shap_df["abs_impact"] = shap_df["impact"].abs()
            shap_df = shap_df.sort_values(by="abs_impact", ascending=False).head(5)

            def clean_feature_name(name):
                return name.replace("num__", "").replace("cat__", "").replace("_", " ")

            shap_df["feature"] = shap_df["feature"].apply(clean_feature_name)

            st.markdown(f"""
                <div class="result-box">
                    <div style="font-size:1rem; color:#166534; margin-bottom:6px;">Estimated Sale Price</div>
                    <div class="result-price">A${prediction:,.0f}</div>
                    <div style="font-size:0.85rem; color:#6b7280; margin-top:8px;">
                        Based on LightGBM model trained on Melbourne Housing Snapshot dataset
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### Why this price?")

            for _, row in shap_df.iterrows():
                sign = "+" if row["impact"] > 0 else "-"
                color = "#16a34a" if row["impact"] > 0 else "#dc2626"

                st.markdown(f"""
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        padding:8px 12px;
                        margin-bottom:6px;
                        border-radius:6px;
                    ">
                        <span>{sign} {row['feature']}</span>
                        <span style="color:{color}; font-weight:600;">
                            {sign}${abs(row['impact']):,.0f}
                        </span>
                    </div>
                """, unsafe_allow_html=True)


        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("This may happen if the suburb wasn't seen during training. Try editing the suburb field.")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("Built with LightGBM · scikit-learn · Streamlit · OpenStreetMap Nominatim")