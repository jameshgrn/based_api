import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Must be the first Streamlit command
st.set_page_config(
   page_title="BASED",
   page_icon="🌊",
   layout="wide",
   initial_sidebar_state="expanded",
)

try:
    model = xgb.Booster()
    model.load_model("based_model_v2.ubj")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()


def predict(slope, discharge, width):
    input_data = pd.DataFrame({
        'log_Q': [np.log10(discharge)],
        'log_w': [np.log10(width)],
        'log_S': [np.log10(slope)],
    }, dtype=float)
    dmat = xgb.DMatrix(input_data)
    log_pred = model.predict(dmat)
    return {"depth": float(10 ** log_pred[0])}


def main():
    st.title("🌊 Boost-Assisted Stream Estimator for Depth (BASED)")

    st.sidebar.title("Input Values")

    slope_help = """Channel slope [m/m]. Valid range: 1e-5 to 0.2
    • Typical lowland rivers: 0.0001-0.001
    • Mountain streams: 0.001-0.02
    """
    discharge_help = """Water discharge [m³/s]. Valid range: 0.01 to 160,000
    • Small streams: 0.1-10
    • Medium rivers: 10-1,000
    • Large rivers: 1,000-100,000
    """
    width_help = """Channel width [m]. Valid range: 0.5 to 3,400
    • Small streams: 1-10
    • Medium rivers: 10-100
    • Large rivers: 100-1,000+
    """

    slope = st.sidebar.text_input(
        "Slope [m/m]:", "0.0001",
        help=slope_help
    )
    width = st.sidebar.text_input(
        "Width [m]:", "250",
        help=width_help
    )
    discharge = st.sidebar.text_input(
        "Discharge [m³/s]:", "400",
        help=discharge_help
    )

    with st.sidebar.expander("ℹ️ About Input Ranges"):
        st.markdown("""
        **Typical Ranges by River Type:**

        *Mountain Streams*
        - Slope: 0.001-0.02
        - Width: 1-20m
        - Discharge: 0.1-50 m³/s

        *Meandering Rivers*
        - Slope: 0.0001-0.001
        - Width: 20-200m
        - Discharge: 50-1000 m³/s

        *Large Rivers*
        - Slope: 0.00001-0.0001
        - Width: 200-2000m
        - Discharge: 1000-100000 m³/s
        """)

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
    st.markdown(
        "- **MAE:** 26.4 cm\n- **RMSE:** 65.2 cm\n- **R²:** 0.91\n- **MAPE:** 17.3%"
    )
    st.image(
        "img/BASED_v2_postmortem.png",
        caption="BASED v2 Validation & Post-Mortem",
        use_container_width=True,
    )
    st.caption(
        "Image source: Gearon, J.H. et al. Rules of river avulsion change "
        "downstream. Nature 634, 91–95 (2024). "
        "https://doi.org/10.1038/s41586-024-07964-2"
    )

    st.subheader("ℹ️ Model Information")
    st.markdown("""
    **Version:** 2.0.0

    **Training data:** 4,315 bankfull hydraulic geometry observations from
    Deal (compilation), Dunne & Jerolmack (global dataset), and
    Bieger et al. 2015 (1,310 field-surveyed US sites).

    **Citation:**
    ```
    Gearon, James H. (2024). Boost-Assisted Stream Estimator for Depth
    (BASED) [Computer software]. Version 2.0.0.
    https://github.com/JakeGearon/based-api
    ```

    **Description:** BASED is an XGBoost regressor that predicts bankfull
    channel depth from channel width, slope, and discharge using
    log-transformed features.
    """)

    st.subheader("🔌 REST API")
    st.markdown("""
    A REST API is also available. Run `uvicorn api:app` and query:

    ```
    GET /predict?slope=0.0001&discharge=400&width=250
    ```

    Returns `{"depth_m": 1.23, "width_depth_ratio": 203.25}`.
    """)

    st.markdown("---")
    st.subheader("⚠️ Disclaimer")
    st.markdown("""
    BASED is provided for informational and research purposes only.
    Do not use these predictions as the sole basis for critical decisions.
    Always consult qualified professionals and conduct proper field
    measurements. The creators are not responsible for consequences
    arising from the use or misuse of this tool.
    """)


if __name__ == "__main__":
    main()
