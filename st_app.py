import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

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
    input_data = pd.DataFrame({'slope': slope, 'width': width, 'discharge': discharge}, index = [0], dtype=float)

    # Make the prediction using the XGBoost model
    prediction = model.predict(input_data)

    # Return the prediction and the explanation
    return {"depth": float(prediction[0])}

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def explain_model_prediction(data):
    # Calculate Shap values
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(data)
    p = shap.force_plot(explainer.expected_value[1], shap_values[1], data)
    return p, shap_values

def main():
    if check_password():
        st.text("up to date error benchmarks: MAE= 0.24 cm, RMSE=0.55, R^2=0.97")
        #st.image("BASED_validation.png", width=500)
        slope = .0001
        discharge = 100
        width = 150

        # Add text boxes to input values
        slope_input = st.sidebar.text_input("Enter slope [m/m]:", str(slope))
        discharge_input = st.sidebar.text_input("Enter discharge [m^3/s]:", str(discharge))
        width_input = st.sidebar.text_input("Enter width [m]:", str(width))

        if slope_input:
            slope = float(slope_input)
        if discharge_input:
            discharge = float(discharge_input)
        if width_input:
            width = float(width_input)
        prediction = predict(slope, discharge, width)
        results = pd.DataFrame({'slope': slope, 'width': width, 'discharge': discharge}, index = [0], dtype = float)
        p, shap_values = explain_model_prediction(results)
        st.subheader('Model Prediction Interpretation Plot')
        st_shap(p)

        st.subheader('Summary Plot 1')
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        shap.summary_plot(shap_values[1], results)
        st.pyplot(fig)

        st.subheader('Summary Plot 2')
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        shap.summary_plot(shap_values[1], results, plot_type = 'bar')
        st.pyplot(fig)
        st.metric(label = "Depth ", value = str(round(prediction["depth"], 2))+' m')


if __name__ == "__main__":
    main()
