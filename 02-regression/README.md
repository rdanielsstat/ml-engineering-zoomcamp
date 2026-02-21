![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-FA0F00?logo=jupyter&logoColor=white)

# Car Fuel Efficiency Prediction — Linear Regression

This project demonstrates a linear regression model to predict car fuel efficiency (MPG) using Python.  It covers:

- Data preparation and handling missing values
- Exploratory data analysis (EDA)
- Linear regression from scratch
- Regularization
- Model evaluation with RMSE
- Model stability analysis

## Dataset

**Source:** [Car Fuel Efficiency Dataset](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv)

**Columns used:**

- `engine_displacement`
- `horsepower` (contains missing values)
- `vehicle_weight`
- `model_year`
- `fuel_efficiency_mpg` (target variable)

Download dataset:

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
```

## Data Preparation & Validation Framework

To ensure accurate predictions, the dataset is prepared carefully:
- Shuffle the dataset to avoid ordering bias
- Split into train/validation/test sets (60% / 20% / 20%)
- Reset indices after splitting to avoid indexing issues

This workflow ensures that the validation and test sets simulate unseen data for fair evaluation.

## Exploratory Data Analysis (EDA)

EDA provides insight into feature quality and target distribution:
- `fuel_efficiency_mpg` is approximately normally distributed (skew ≈ -0.01)
- Only `horsepower` contains missing values (708 missing)
- Median horsepower ≈ 149
- Histogram confirms no long tail in the target distribution

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("car_fuel_efficiency.csv")

# Visualize distribution of fuel efficiency
sns.histplot(df.fuel_efficiency_mpg)
plt.title("Fuel Efficiency (MPG)")
plt.show()
```

## Handling Missing Values

`horsepower` contains missing values, which can be handled by:
- Filling with 0
- Filling with the median (or mean) of the training set

These strategies affect model performance and are compared using RMSE.

```python
def prepare_X(df, fill_method = 'zeros'):
    df = df.copy()
    features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
    df_num = df[features]
    
    if fill_method == 'zeros':
        df_num = df_num.fillna(0)
    elif fill_method == 'mean':
        df_num = df_num.fillna(df_num.mean())
    return df_num.values
```

## Linear Regression Model

Linear regression predicts a continuous target as a linear combination of features:
- Vector form: $y^ = X \cdot w$
- Normal Equation: $w = (XᵀX)⁻¹ Xᵀy$
- Efficient computation with vectorized operations

```python
import numpy as np

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]

def rmse(y, y_pred):
    return np.sqrt(((y - y_pred) ** 2).mean())
```

## Model Evaluation
- Compare missing value strategies (0 vs mean) using RMSE
- Tune regularization (r) to prevent overfitting
- Analyze model stability across different random seeds
- Compute final RMSE on the test set

```python
# Example: fill NAs with mean and train
X_train = prepare_X(df_train, fill_method = 'mean')
y_train = df_train.fuel_efficiency_mpg.values
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val, fill_method = 'mean')
y_val = df_val.fuel_efficiency_mpg.values
y_pred = w0 + X_val.dot(w)

score = round(rmse(y_val, y_pred), 2)
print("Validation RMSE:", score)
```

## Regularization & Tuning

Regularization improves model stability by penalizing large weights:
- L2 (Ridge) regularization adds $r·||w||²$ to the loss
- Evaluate r on the validation set
- Choose r that minimizes RMSE without overfitting

In this dataset, a small r or no regularization may already yield good performance.

## Final Model & Stability

- Combine train + validation sets for final training
- Evaluate on the test set to estimate generalization performance
- RMSE standard deviation across multiple seeds (~0.007) indicates model is stable

## Summary of Results

- Filling missing values with mean improves predictions
- RMSE provides an intuitive measure of error in MPG
- Regularization may not be necessary for this dataset
- Model is stable and generalizes well to unseen data