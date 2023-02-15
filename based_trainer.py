import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils import resample
from data_format import generate_data

# Set Seaborn style and font scale
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.2)
generate_data()
df = pd.read_csv('data/based_input_data.csv')



X = df.filter(['width', 'slope', 'discharge'], axis=1)
y = df['depth']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)

# Train the XGBoost model
# Set up the parameter grid for the randomized search
param_dist = {
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'max_depth': range(3, 10),
    'n_estimators': range(50, 150, 10),
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1, 10],
    'reg_lambda': [0, 0.1, 0.5, 1, 10]
}

# Train and tune the XGBoost model using randomized search with cross-validation
xg_reg = xgb.XGBRegressor(objective="reg:squarederror")
random_search = RandomizedSearchCV(estimator=xg_reg, param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
params = random_search.best_params_

xg_reg = xgb.XGBRegressor(**params)
xg_reg.fit(X_train, y_train, eval_metric='mae')

# Predict on test set using bootstrapping and calculate mean and confidence interval
n_iterations = 100
predictions = np.zeros((n_iterations, len(y_test)))
for i in range(n_iterations):
    X_resample, y_resample = resample(X_train, y_train)
    xg_reg.fit(X_resample, y_resample)
    predictions[i] = xg_reg.predict(X_test)
mean_predictions = predictions.mean(axis=0)
std_predictions = predictions.std(axis=0)
confidence_interval = [mean_predictions - 1.96 * std_predictions, mean_predictions + 1.96 * std_predictions]

# Print model report and plot results
print("\nModel Report")
print("MAE (Train): %f" % metrics.mean_absolute_error(y_test, mean_predictions))
print("RMSE (Train): %f" % np.sqrt(metrics.mean_squared_error(y_test, mean_predictions)))
print("R2: %f" % metrics.r2_score(y_test, mean_predictions))


plt.figure(figsize=(10,10))
sns.set_context('talk')
sns.set(font_scale = 2.2)
p1 = max(max(mean_predictions), max(y_test))
p2 = min(min(mean_predictions), min(y_test))

sns.scatterplot(x=y_test, y=mean_predictions, color= '#FFCCBC', edgecolor='k', s=100)
plt.plot([p1, p2], [p1, p2], 'k--', label='1:1 line', lw=3.5)

plt.yscale('log')
plt.xscale('log')

plt.xlabel('Measured Channel Depth (m)', )
plt.ylabel('Predicted Channel Depth (m)', )
plt.title('BASED Validation | n = {}'.format(len(y_test)), )
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.savefig("img/BASED_validation.png", dpi=250)
xg_reg.save_model("based.pkl")


