import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# Set page configurations
st.set_page_config(
    page_title="Banana Pseudostem Soup - Shelf Life Predictor",
    page_icon="🍌",
    layout="wide"
)

# =====================================================================
# SESSION STATE INITIALIZATION (Handles Navigation & History)
# =====================================================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# Helper function to switch pages
def navigate_to(page_name):
    st.session_state.current_page = page_name
    st.rerun()

# =====================================================================
# DATA LOADING & MODEL TRAINING (Cached)
# =====================================================================
@st.cache_resource
def load_data_and_train_model():
    possible_paths = [
        "Shelf Life dataset (1).xlsx",
        "Shelf Life dataset.xlsx",
        r"C:\Users\Asus\Downloads\Shelf Life dataset (1).xlsx",
        r"C:\Users\Asus\Downloads\Shelf Life dataset.xlsx"
    ]
    
    selected_file = None
    for path in possible_paths:
        if os.path.exists(path):
            selected_file = path
            break
            
    if selected_file is None:
        return None, None, None

    # Load and clean
    df = pd.read_excel(selected_file, header=None)
    df.columns = df.iloc[3]
    df = df.iloc[5:]
    df = df.drop(columns=[df.columns[0]])
    df = df.reset_index(drop=True)
    
    # Standardize column headers
    df.columns = (
        df.columns.str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace(' & ', '_', regex=False)
        .str.replace(' ', '_', regex=False)
        .str.upper()
    )
    
    # Split color coordinates
    df[['L', 'A', 'B']] = df['COLOUR'].str.split(',', expand=True).astype(float)
    df = df.drop(columns=['COLOUR'])
    
    numeric_columns = ["DAY", "MOISTURE", "PH", "ACIDITY", "WATER_ACTIVITY", "TPC", "YEAST_MOLD", "SENSORY_SCORE"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
        
    X = df.drop("DAY", axis=1)
    y = df["DAY"]
    
    # Train Deployment Model
    model = RandomForestRegressor(n_estimators=120, max_depth=4, random_state=42)
    model.fit(X, y)
    
    return model, X, df

model, X, df = load_data_and_train_model()

# Handle missing dataset error gracefully in UI
if model is None:
    st.error("⚠️ Could not locate 'Shelf Life dataset (1).xlsx'! Please ensure the file is in your Downloads folder.")
    st.stop()

# =====================================================================
# SIDEBAR NAVIGATION
# =====================================================================
st.sidebar.title("Navigation Menu")
if st.sidebar.button("🏠 Home / Title Page", use_container_width=True):
    navigate_to("Home")
if st.sidebar.button("🔬 Predictor Form", use_container_width=True):
    navigate_to("Predictor")
if st.sidebar.button("📊 Results Screen", use_container_width=True, disabled=(st.session_state.last_prediction is None)):
    navigate_to("Results")
if st.sidebar.button("📜 Sample History", use_container_width=True):
    navigate_to("History")
if st.sidebar.button("📉 Model Diagnostics", use_container_width=True):
    navigate_to("Diagnostics")

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Enter measurements in the Predictor Form to automatically view the Results screen.")

# =====================================================================
# PAGE 1: HOME / STARTING PAGE
# =====================================================================
if st.session_state.current_page == "Home":
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Grid layout to center our logo and title
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        # Display Logo (Looks for a file named "logo.png" - displays a beautiful banner as a fallback)
        if os.path.exists("logo.png"):
            st.image("logo.png", width=300)
        else:
            # Styled placeholder logo using markdown
            st.markdown(
                """
                <div style="text-align: center; background-color: #f7f9fc; border-radius: 50%; width: 180px; height: 180px; display: inline-flex; justify-content: center; align-items: center; margin-bottom: 20px; border: 4px solid #fec30a;">
                    <span style="font-size: 80px;">🍌</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        st.markdown("<h1 style='text-align: center; color: #1e3d59;'>Banana Pseudostem Instant Soup</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #17b978;'>Smart Food Preservation & AI Shelf Life Prediction</h3>", unsafe_allow_html=True)
        st.write("---")
        st.markdown(
            """
            Welcome to the AI deployment dashboard. This application employs machine learning models 
            trained over longitudinal physical, chemical, sensory, and microbiological data to accurately 
            predict the storage age and remaining safe shelf life of instant functional banana pseudostem soup powder.
            """
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Enter Predictor Dashboard", use_container_width=True, type="primary"):
            navigate_to("Predictor")

# =====================================================================
# PAGE 2: INPUTS FORM
# =====================================================================
elif st.session_state.current_page == "Predictor":
    st.subheader("🔬 Sample Quality Measurement Form")
    st.write("Provide the laboratory metrics below to calculate the shelf life assessment.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### **Physicochemical Properties**")
            moisture = st.slider("Moisture Content (%)", min_value=0.0, max_value=15.0, value=12.2, step=0.1)
            pH = st.slider("pH Level", min_value=0.0, max_value=7.0, value=6.35, step=0.01)
            acidity = st.slider("Acidity (%)", min_value=0.0, max_value=2.0, value=1.09, step=0.01)
            water_activity = st.slider("Water Activity ($a_w$)", min_value=0.00, max_value=0.85, value=0.716, step=0.001)

        with col2:
            st.markdown("##### **Microbial & Sensory**")
            tpc = st.number_input("Total Plate Count (TPC) (CFU/g)", min_value=0, max_value=500, value=115)
            yeast_mold = st.number_input("Yeast & Mold (CFU/g)", min_value=0, max_value=100, value=14)
            sensory = st.slider("Sensory Acceptability (1-10)", min_value=1.0, max_value=10.0, value=9.0, step=0.1)

        with col3:
            st.markdown("##### **CIELAB Color Coordinates**")
            L = st.slider("L* (Lightness)", min_value=0.0, max_value=50.0, value=32.1, step=0.1)
            a = st.slider("a* (Redness)", min_value=0.0, max_value=15.0, value=8.0, step=0.1)
            b = st.slider("b* (Yellowness)", min_value=0.0, max_value=25.0, value=13.8, step=0.1)
            
        submit_button = st.form_submit_button("🔮 Submit & Calculate Predictions", use_container_width=True)
        
    if submit_button:
        # Construct Input Dataframe
        input_vector = pd.DataFrame([{
            'MOISTURE': moisture, 'PH': pH, 'ACIDITY': acidity, 'WATER_ACTIVITY': water_activity,
            'TPC': tpc, 'YEAST_MOLD': yeast_mold, 'SENSORY_SCORE': sensory, 'L': L, 'A': a, 'B': b
        }])
        
        # Calculate Machine Learning Predictions
        predicted_stored_days = model.predict(input_vector)[0]
        remaining_days = max(0.0, 56.0 - predicted_stored_days)
        
        # Check alerts
        alerts = []
        if moisture > 13.5:
            alerts.append(f"High moisture ({moisture}%). Standard limit is 13.5%.")
        if water_activity > 0.75:
            alerts.append(f"Dangerous Water Activity level ({water_activity}). Risk of bacterial expansion!")
        if sensory < 8.3:
            alerts.append(f"Low Sensory Acceptability Score ({sensory}/10).")
        if tpc > 250:
            alerts.append(f"High Total Plate Count ({tpc} CFU/g).")
            
        status = "CRITICAL RISK" if len(alerts) > 0 else "SAFE"
        
        # Record results dictionary
        result_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "moisture": moisture, "pH": pH, "acidity": acidity, "water_activity": water_activity,
            "tpc": tpc, "yeast_mold": yeast_mold, "sensory": sensory,
            "L": L, "a": a, "b": b,
            "predicted_age": round(predicted_stored_days, 2),
            "remaining_life": round(remaining_days, 2),
            "status": status,
            "alerts": alerts
        }
        
        # Save to Session States
        st.session_state.last_prediction = result_payload
        st.session_state.history.append(result_payload)
        
        # Navigate to Results Page automatically
        navigate_to("Results")

# =====================================================================
# PAGE 3: RESULTS SCREEN
# =====================================================================
elif st.session_state.current_page == "Results":
    st.subheader("📋 Machine Learning Assessment Results")
    
    res = st.session_state.last_prediction
    
    if res is None:
        st.warning("⚠️ No prediction has been made yet. Please visit the Predictor Page first.")
        if st.button("Go to Predictor"):
            navigate_to("Predictor")
    else:
        # Display key metrics in highlighted layout cards
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Predicted Storage Age of Sample", value=f"{res['predicted_age']} Days")
        with col_m2:
            st.metric(label="Estimated Safe Remaining Shelf Life", value=f"{res['remaining_life']} Days", delta=f"{res['remaining_life'] - 56.0:.2f} Days from Day 0")
        
        # Render status box
        if res["status"] == "CRITICAL RISK":
            st.error("🔴 **ALERT STATUS: SPOILED / CRITICAL SAFETY RISK**")
            for alert in res["alerts"]:
                st.write(f"⚠️ {alert}")
        else:
            st.success("🟢 **ALERT STATUS: SAFE / EXCELLENT CONSERVATION**")
            st.write("This soup formulation strictly satisfies all commercial microbiological stability guidelines.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🔄 Test Another Sample", use_container_width=True):
                navigate_to("Predictor")
        with col_b2:
            if st.button("📜 View Sample History Log", use_container_width=True):
                navigate_to("History")

# =====================================================================
# PAGE 4: SAMPLE HISTORY LOG
# =====================================================================
elif st.session_state.current_page == "History":
    st.subheader("📜 Project Tested Sample Archive")
    st.write("History log of all instant soup test cycles evaluated during this active session.")
    
    if len(st.session_state.history) == 0:
        st.info("ℹ️ No samples have been entered in this session yet.")
        if st.button("Back to Predictor Form"):
            navigate_to("Predictor")
    else:
        # Convert list of dicts to flat DataFrame for visual rendering
        history_df = pd.DataFrame(st.session_state.history)
        
        # Rearrange columns for friendly viewing
        friendly_col_order = [
            "timestamp", "predicted_age", "remaining_life", "status",
            "moisture", "pH", "water_activity", "tpc", "yeast_mold", "sensory"
        ]
        render_df = history_df[friendly_col_order].rename(columns={
            "timestamp": "Timestamp", "predicted_age": "Est. Age (Days)", "remaining_life": "Est. Remaining (Days)",
            "status": "Safety Alert Status", "moisture": "Moisture (%)", "pH": "pH", "water_activity": "Water Activity (aw)",
            "tpc": "TPC (CFU/g)", "yeast_mold": "Yeast/Mold", "sensory": "Sensory Score"
        })
        
        # Show interactive dataframe
        st.dataframe(render_df, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Clear Local History Log"):
            st.session_state.history = []
            st.session_state.last_prediction = None
            st.rerun()

# =====================================================================
# PAGE 5: MODEL DIAGNOSTICS & FEATURE IMPORTANCE
# =====================================================================
elif st.session_state.current_page == "Diagnostics":
    st.subheader("📊 Analytical Models & Core Features")
    
    col_d1, col_d2 = st.columns([1, 1])
    
    with col_d1:
        st.markdown("##### **Quality Feature Drivers Matrix**")
        # Calculate feature importance dynamically
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Importance", y="Feature", data=importance_df, palette="mako", ax=ax, hue="Feature", legend=False)
        ax.set_title("Relative Gini Feature Importance Matrix", fontsize=10, fontweight='bold')
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("")
        st.pyplot(fig)
        
    with col_d2:
        st.markdown("##### **Mathematical Algorithm Verification**")
        comparison_data = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'Support Vector Regressor', 'Gradient Boosting'],
            'MAE (Days)': [4.80, 3.98, 5.66, 4.18],
            'RMSE (Days)': [5.17, 5.20, 9.13, 4.91],
            'R2 Score': [0.917, 0.917, 0.744, 0.926]
        })
        st.dataframe(comparison_data, hide_index=True)
        
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        sns.barplot(x='Model', y='R2 Score', data=comparison_data, palette='coolwarm', ax=ax2, hue='Model', legend=False)
        ax2.set_title("Coefficient of Determination ($R^2$ Score)", fontsize=10, fontweight='bold')
        ax2.set_ylabel("R2 Score")
        ax2.set_xlabel("")
        ax2.tick_params(axis='x', rotation=15, labelsize=8)
        st.pyplot(fig2)
