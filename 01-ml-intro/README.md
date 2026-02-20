![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Jupyter](https://img.shields.io/badge/Jupyter-FA0F00?logo=jupyter&logoColor=white)
![CRISP-DM](https://img.shields.io/badge/Framework-CRISP--DM-informational)
![Supervised Learning](https://img.shields.io/badge/ML-Supervised%20Learning-brightgreen)

# Machine Learning Introduction

*Foundations of supervised learning, model selection, and the ML workflow*

## Project Overview

This module establishes the conceptual and mathematical foundations of machine learning engineering.

Using the Car Fuel Efficiency dataset, I explored:
- The difference between machine learning and rule-based systems
- Core supervised learning concepts
- The CRISP-DM framework for structuring ML projects
- Model validation and selection strategies
- Linear algebra foundations for regression
- Practical environment setup for reproducible ML work

This project demonstrates not just how to train models, but how to think about ML problems systematically and rigorously.

## What Is Machine Learning?

Machine learning is the process of learning patterns from data in order to make predictions on unseen examples.

Each dataset consists of:
- **Features (X)**: Descriptive attributes of objects
- **Target (y)**: The value we want to predict

The goal of training is to learn a function:

$$g(X) = y$$

Where the model approximates the mapping from features to target.

In this module, we focused on **supervised learning**, where labeled examples guide model training.

### Types of Supervised Learning
- Regression → Predicting numeric values
- Classification → Predicting categories
- Ranking → Ordering items by relevance

The Car Fuel Efficiency dataset is primarily a regression problem, since fuel efficiency is a numeric target.

## ML vs Rule-Based Systems

Traditional rule-based systems rely on manually defined logic. For example, spam detection based on keywords.

These systems:
- Require constant maintenance
- Become complex and brittle over time
- Do not generalize well

Machine learning systems instead:
1. Learn from historical data
2. Convert inputs into numerical features
3. Produce probabilistic predictions
4. Use thresholds for decision-making

This shift from explicit rules to learned patterns is foundational in modern ML systems.

## Project Framework: CRISP-DM

I structured the project using the CRISP-DM methodology, the industry standard lifecycle for ML projects.

1. Business Understanding

    Define the prediction goal clearly and ensure it is measurable.

2. Data Understanding

    Explore distributions, identify missing values, understand feature structure.

3. Data Preparation

    Clean data, handle missing values, transform features into tabular form.

4. Modeling

    Train candidate models and evaluate performance.

5. Evaluation

    Assess whether the model solves the business objective.

6. Deployment

    Prepare the solution for production use.

Machine learning is inherently iterative. Each stage informs the next.

## Model Selection and Validation Strategy

A critical part of ML engineering is avoiding overfitting and selection bias.

Dataset split strategy:
- Training: 60%
- Validation: 20%
- Test: 20%

Workflow:
1. Train models on training data
2. Evaluate on validation data
3. Select the best candidate
4. Retrain on combined train + validation
5. Evaluate once on the test set

This guards against the **Multiple Comparisons Problem**, where one model may appear best purely by chance.

## Environment and Reproducibility

This project was implemented using:
- Python 3.11
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook

Pandas version used:

```python
import pandas as pd

pd.__version__
```

### Installed Version: 2.x (installed in the ml-zoomcamp conda environment)

A dedicated conda environment ensures reproducibility:

```bash
conda create -n ml-zoomcamp python = 3.11
conda activate ml-zoomcamp
conda install numpy pandas scikit-learn seaborn jupyter
```

## Dataset Exploration: Car Fuel Efficiency

Dataset: `car_fuel_efficiency.csv`

### Dataset Size

Total records: **8704**

### Fuel Types

The dataset contains **4 distinct fuel types**.

### Missing Values

There are **3 columns with missing values**, including horsepower.

## Handling Missing Data

The horsepower column contained missing values.

Steps taken:
1. Compute median horsepower
2. Compute most frequent value (mode)
3. Fill missing values using the mode
4. Recalculate median

Result: **The median horsepower did not change after imputation.**

This illustrates how mode imputation can preserve distribution shape when missingness is limited.

## Linear Algebra Foundations

To reinforce understanding of regression mechanics, I implemented matrix operations manually.

### Dot Product

$$ u \cdot v = \sum_{i = 1}^{n} u_i v_i $$

### Matrix Multiplication

Matrix multiplication forms the backbone of linear regression computation.

### Linear Regression via Normal Equation

$$ w = (X^T X)^{-1} X^T y $$

Using:
- Vehicle weight
- Model year
- Selected subset of Asian cars

After computing:
- $X^T X$
- Its inverse
- Final weight vector $w$

**Sum of regression coefficients:** 51

This exercise connects linear algebra directly to model training mechanics.

## Key Technical Skills Demonstrated

- Data loading and exploration with Pandas
- Handling missing values using statistical imputation
- Vectorized numerical computation with NumPy
- Manual implementation of matrix multiplication
- Linear regression via the normal equation
- Proper dataset splitting and validation methodology
- Reproducible Python environment setup

## Core Takeaways

1. Machine learning is structured pattern extraction, not magic.
2. Validation strategy is as important as modeling.
3. Linear algebra underpins regression models.
4. Clean environments enable reproducibility.
5. ML engineering begins long before deployment.

This module establishes the conceptual rigor needed for scaling from notebook experiments to production ML systems.