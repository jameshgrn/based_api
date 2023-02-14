import streamlit as st
import pandas as pd
import xgboost as xgb

# Load the pre-trained XGBoost model from the .pkl file
model = xgb.XGBRegressor()
model.load_model("based.pkl")

def check_password():
    """Returns `True` if the user had the correct password."""
    st.title("Boost-Assissted Stream Estimator for Depth (BASED) Prediction App")
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True



def predict(slope, discharge, width):
    """
    Function for making predictions using the XGBoost model.
    """
    # Prepare the input features for prediction
    input_data = pd.DataFrame({'width': width, 'slope': slope, 'discharge': discharge}, index = [0], dtype=float)

    # Make the prediction using the XGBoost model
    prediction = model.predict(input_data)

    # Return the prediction
    return {"depth": float(prediction[0])}

def main():
    if check_password():
        st.header("Error Benchmarks")
        st.markdown("MAE = 0.24 cm, "
                    "RMSE = 0.55, "
                    "R^2=0.97")
        # st.image("BASED_validation.png", width=500)
        slope = .0001
        discharge = 100
        width = 150

        st.sidebar.title("Input Values")
        slope_input = st.sidebar.text_input("Slope [m/m]:", str(slope))
        discharge_input = st.sidebar.text_input("Discharge [m^3/s]:", str(discharge))
        width_input = st.sidebar.text_input("Width [m]:", str(width))

        if slope_input:
            slope = float(slope_input)
        if discharge_input:
            discharge = float(discharge_input)
        if width_input:
            width = float(width_input)

        prediction = predict(slope, discharge, width)
        st.metric(label = "Depth ", value = str(round(prediction["depth"], 2))+' m')

if __name__ == "__main__":
    main()
