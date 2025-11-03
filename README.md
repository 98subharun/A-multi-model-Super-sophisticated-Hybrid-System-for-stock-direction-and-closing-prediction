# Description
This model is a combination of a DNN and a ensemble based architecture to forcast both next day closing prices and direction of financial assests

# Core pipeline
It loads and merges hundreds of stock CSVs(here used S&P 500 dataset)
performs time based sorting and ticket tagging

# Feature Engineering

To obtain the desired temporal features, the following quantitative variables are created:

Returns: The change in price from yesterday to today relative to yesterday's closing price (percentage)
Log Returns: A logarithmized form to stabilize variance
Moving Averages (MA7, MA30, MA21): The average price for the past week, month and 21 days to show intermediate to long patterns
Volatility (σ7, σ21): The rolling standard deviation of returns is a measure of uncertainty over the market's history.
Momentum: Change compared to MA of momentum based on shorter averaged time - tendency to persist in the same direction.

Everything was scaled using ```StandardScaler()```

# Model Architecture

A. Base Models (Level 1)

Random Forest Regressor - captures non-linear dependencies and feature interactions through an ensemble of decision trees.

XGBoost Regressor - a highly regularized, tree-based, gradient-boosted model tailored for structured tabular data.

Deep Neural Network (DNN) - A feedforward structure for generalization with dense layers, batch normalization, and dropout.

Each of these models predicts the next-day closing price based on the engineered features independently.

B. Meta-Learner (Level 2 Stacking)

Predictions from RF, XGB, and DNN are treated as meta-features.

A RidgeCV regressor (for regression) or Logistic Regression (for classification) learns the appropriate weights for these predictions combined.

This meta-ensemble minimizes bias and variance from different independent models.

📈 Two Predictions at Once

The SSPA performs two predictions simultaneously:

Next-Day Closing Price (Regression)
Predicts the value for the next trading day.

Next-Day Trend Direction (Classification)
Predicts whether the value will increase (↑) or decrease (↓) compared to the current day by leveraging:

Direction= ![Direction Equation](https://latex.codecogs.com/png.latex?%5Ctext%7BDirection%7D%3D%5Cbegin%7Bcases%7D1%2C%26%5Ctext%7Bif+%7DClose_%7Bt%2B1%7D%3EClose_t%5C%5C0%2C%26%5Ctext%7Botherwise%7D%5Cend%7Bcases%7D)

Two models will be trained for regression and classification; however, both follow the same ensemble structure as a basis.

🧮 Evaluation Metrics

For Regression:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² (Coefficient of Determination)

For Classification:

Accuracy

ROC-AUC

Confusion Matrix

Visualizations will show actual vs predicted closing prices and the trend prediction results.
