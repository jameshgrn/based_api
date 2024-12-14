import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches
import numpy as np

# Must be the first Streamlit command
st.set_page_config(
   page_title="BASED",
   page_icon="🌊",
   layout="wide",
   initial_sidebar_state="expanded",
   menu_items={
       'Get Help': 'https://github.com/jameshgrn/based_api/issues',
       'Report a bug': 'https://github.com/jameshgrn/based_api/issues/new',
       'About': "BASED: The Boost-Assisted Stream Estimator for Depth."
   }
)

try:
    # Load model directly with XGBoost
    model = xgb.Booster()
    model.load_model("based_us_sans_trampush_early_stopping_combat_overfitting.ubj")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

def predict(slope, discharge, width):
    """
    Function for making predictions using the XGBoost model.
    """
    # Prepare the input features for prediction
    input_data = pd.DataFrame({
        'width': [width], 
        'slope': [slope], 
        'discharge': [discharge]
    }, dtype=float)
    
    # Convert to DMatrix for prediction
    dmat = xgb.DMatrix(input_data)
    
    # Make prediction
    prediction = model.predict(dmat)
    
    return {"depth": float(prediction[0])}

def main():
    st.title("🌊 BASED: River Depth Predictor")
    st.markdown("""
        <style>
            .reportview-container {
                background: linear-gradient(to right, #f0f2f6, #ffffff)
            }
            .sidebar .sidebar-content {
                background: linear-gradient(to bottom, #f0f2f6, #ffffff)
            }
        </style>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        **Welcome to BASED** - The Boost-Assisted Stream Estimator for Depth
        
        Enter your river parameters in the sidebar to get instant depth predictions.
        This model is trained on thousands of river measurements and published in Nature.
        """)
    with col2:
        st.markdown("""
        [![GitHub](https://img.shields.io/github/stars/jameshgrn/based_api?style=social)](https://github.com/jameshgrn/based_api)
        [![Paper](https://img.shields.io/badge/Read-Paper-blue)](https://www.nature.com/articles/s41586-024-07964-2)
        """)

    st.sidebar.title("Input Values")
    
    # Add help text for each input
    slope_help = """Channel slope [m/m]. Valid range: 1e-6 to 0.1
    • Typical lowland rivers: 0.0001-0.001
    • Mountain streams: 0.001-0.02
    """
    discharge_help = """Water discharge [m³/s]. Valid range: 0.1 to 100,000
    • Small streams: 0.1-10
    • Medium rivers: 10-1,000
    • Large rivers: 1,000-100,000
    """
    width_help = """Channel width [m]. Valid range: 1 to 10,000
    • Small streams: 1-10
    • Medium rivers: 10-100
    • Large rivers: 100-1,000+
    """
    
    slope = st.sidebar.text_input(
        "Slope [m/m]:", "0.0001",
        help=slope_help
    )
    discharge = st.sidebar.text_input(
        "Discharge [m³/s]:", "400",
        help=discharge_help
    )
    width = st.sidebar.text_input(
        "Width [m]:", "250",
        help=width_help
    )

    # Add typical ranges info box
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

            # Clear any existing plots
            plt.close('all')

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Depth", f"{depth:.2f} m")
                st.metric("Width/Depth Ratio", f"{width/depth:.2f}")
            with col2:
                # Create a simple cartoon cross-section
                fig, ax = plt.subplots(figsize=(12, 6))
                fig.patch.set_facecolor('#0E1117')
                ax.set_facecolor('#0E1117')
                
                # Calculate vertical exaggeration
                ve = 20  # Fixed vertical exaggeration for consistency
                
                # Set view limits relative to river dimensions
                margin = max(width * 0.2, 10)  # minimum 10m margin
                ax.set_xlim(-margin, width + margin)
                ax.set_ylim(-depth * ve * 1.2, depth * ve * 0.4)
                
                # Force aspect ratio
                ax.set_aspect(1/ve)
                
                # Simple rectangular channel with rounded corners
                rect = patches.Rectangle((-margin, -depth * ve), 
                                      width + 2*margin, 
                                      depth * ve * 1.4,
                                      facecolor='#654321',  # Brown color for ground
                                      alpha=0.7)
                ax.add_patch(rect)
                
                # Water body (simple rectangle with slight transparency)
                water = patches.Rectangle((0, -depth * ve),
                                       width,
                                       depth * ve,
                                       facecolor='#4BA3F9',  # Light blue
                                       alpha=0.8)
                ax.add_patch(water)
                
                # Simple wave pattern on top
                x = np.linspace(0, width, 100)
                wave_height = depth * ve * 0.02
                wave = wave_height * np.sin(x * np.pi * 8 / width)
                ax.plot(x, wave, color='white', alpha=0.8, linewidth=2)
                
                # Add measurement arrows with correct style
                arrow_props = dict(arrowstyle='<->', color='white', linewidth=2,
                                 mutation_scale=15)  # mutation_scale controls the arrow size
                
                # Width arrow
                width_y = -depth * ve * 0.8
                ax.annotate('', xy=(0, width_y), xytext=(width, width_y),
                           arrowprops=arrow_props)
                ax.text(width/2, width_y * 1.1, f'Width: {width:.1f} m',
                       color='white', ha='center', va='bottom',
                       bbox=dict(facecolor='#0E1117', edgecolor='none', alpha=0.7),
                       fontsize=12)
                
                # Depth arrow
                depth_x = width * 0.85
                ax.annotate('', xy=(depth_x, 0), xytext=(depth_x, -depth * ve),
                           arrowprops=arrow_props)
                ax.text(depth_x * 1.05, -depth * ve/2, f'Depth:\n{depth:.1f} m',
                       color='white', ha='left', va='center',
                       bbox=dict(facecolor='#0E1117', edgecolor='none', alpha=0.7),
                       fontsize=12)
                
                # Add VE annotation
                ax.text(0.02, 0.98, f'VE: {ve}×',
                       transform=ax.transAxes,
                       color='white', ha='left', va='top',
                       bbox=dict(facecolor='#0E1117', edgecolor='none', alpha=0.7),
                       fontsize=10)
                
                # Style improvements
                ax.grid(False)
                ax.set_xlabel('Width (m)', color='#FAFAFA', fontsize=11)
                ax.set_ylabel('Elevation (m)', color='#FAFAFA', fontsize=11)
                ax.tick_params(colors='#FAFAFA', labelsize=10)
                
                # Remove spines
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                # Adjust layout
                plt.tight_layout()
                st.pyplot(fig)

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
