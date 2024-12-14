import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Initialize XGBoost model with parameters
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    max_depth=12,
    subsample=0.8,
    learning_rate=0.09,
    n_estimators=75,
    reg_lambda=1.2
)

# Load the pre-trained model using pickle instead of .ubj
with open('based.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(
   page_title="BASED",
   page_icon="🌊",
   layout="wide",
   initial_sidebar_state="expanded",
)

def predict(slope, discharge, width):
    """
    Function for making predictions using the XGBoost model.
    """
    # Prepare the input features for prediction
    input_data = pd.DataFrame({'width': width, 'slope': slope, 'discharge': discharge}, index=[0], dtype=float)

    # Make the prediction using the XGBoost model
    prediction = model.predict(input_data)

    # Return the prediction
    return {"depth": float(prediction[0])}

def main():
    st.title("🌊 Boost-Assisted Stream Estimator for Depth (BASED)")

    st.sidebar.title("Input Values")
    slope = st.sidebar.text_input("Slope [m/m]:", "0.0001")
    discharge = st.sidebar.text_input("Discharge [m³/s]:", "400")
    width = st.sidebar.text_input("Width [m]:", "250")

    if slope and discharge and width:
        try:
            slope = float(slope)
            discharge = float(discharge)
            width = float(width)

            prediction = predict(slope, discharge, width)
            depth = prediction["depth"]

            st.metric("Predicted Depth", f"{depth:.2f} m")
            st.metric("Width/Depth Ratio", f"{width/depth:.2f}")

        except ValueError:
            st.error("Please enter valid numeric values for all inputs.")

    st.subheader("📊 Model Performance")
    st.markdown("- **MAE:** 33 cm\n- **RMSE:** 102 cm\n- **R²:** 0.89\n- **MAPE:** 20%")
    st.image("img/BASED_validation.png", caption="BASED Validation Results", use_container_width=True)
    st.caption("Image source: Gearon, J.H. et al. Rules of river avulsion change downstream. Nature 634, 91–95 (2024). https://doi.org/10.1038/s41586-024-07964-2")
    
    st.subheader("ℹ️ Model Information")
    st.markdown("""
    **Version:** 1.0.0
    
    **Citation:**
    ```
    Gearon, James H. (2024). Boost-Assisted Stream Estimator for Depth (BASED) [Computer software]. Version 1.0.0. https://github.com/JakeGearon/based-api
    ```
    
    **Description:** BASED is an XGBoost regressor designed for predicting channel depth using channel width, slope, and discharge.
    """)

    st.markdown("---")
    st.subheader("⚠️ Disclaimer")
    st.markdown("""
    BASED is provided for informational and research purposes only. Do not use these predictions as the sole basis for critical decisions. 
    Always consult qualified professionals and conduct proper field measurements. The creators are not responsible for consequences arising 
    from the use or misuse of this tool.
    """)

if __name__ == "__main__":
    main()
