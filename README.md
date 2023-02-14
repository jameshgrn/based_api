# based-api

based-api is a package that serves predictions from the Boost-Assissted Stream Estimator for Depth (BASED).
BASED is an XGBoost regressor deployed using Streamlit. This package is designed for making channel depth predictions using channel width, slope, and discharge.

Requirements
- pandas
- scikit-learn
- xgboost
### File Descriptions

`based_trainer.py`: contains the script for training the XGBoost regressor on the input dataset and generating model reports and results.
app.py: contains the Streamlit app script for serving the pre-trained XGBoost regressor. It also includes password validation and a function for making predictions.

### Usage

1. Clone the repository:
```git clone https://github.com/<username>/based-api.git```

2. Install the required packages: ```pip install -r requirements.txt```

3. Train the XGBoost regressor by running `based_trainer.py`. It will generate an XGBoost model file named based.pkl.

4. Run the Streamlit app by running `st_app.py`.

5. Enter the password to access the app.
6. Enter the values for slope, discharge, and width to make a prediction for the channel depth.
The app will output the predicted channel depth.

### License

This project is licensed under the terms of the MIT license.