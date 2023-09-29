# Paris_Housing_using_LinearRegression
---

# Paris Housing Price Prediction

## Overview

This repository contains code for a machine learning project that predicts housing prices in Paris based on various features of the properties. The project uses a Linear Regression model to make predictions.

## Dataset

We used the "ParisHousing.csv" dataset, which contains information about housing properties in Paris. The dataset includes features such as square meters, number of rooms, presence of a yard or pool, number of floors, city code, and more. The target variable is the price of the properties.

### Data Exploration

- `dataset.info()`: Provides information about the dataset, including data types and missing values.
- `dataset.head(10)`: Displays the first 10 rows of the dataset.
- `dataset.isnull().sum()`: Counts missing values in the dataset.
- `dataset.describe()`: Computes summary statistics of the dataset.
- `dataset.shape`: Shows the number of rows and columns.
- `dataset.columns`: Lists the column names.

## Data Preprocessing

We split the dataset into training and testing sets using `train_test_split` from `sklearn.model_selection`. The features are stored in `X_train` and `X_test`, while the target variable (price) is in `y_train` and `y_test`.

## Model Training

We used a Linear Regression model from `sklearn.linear_model` to train the model. The model was trained on the training data with the `model.fit(X_train, y_train)` method.

## Model Evaluation

To evaluate the model, we defined a function `evaluate` that calculates the Mean Absolute Percentage Error (MAPE) and accuracy of the model's predictions. We applied this function to both the training and testing sets.

- `evaluate(model, X_train, y_train)`: Evaluates the model on the training data.
- `evaluate(model, X_test, y_test)`: Evaluates the model on the testing data.

## Performance Metrics

- **Mean Absolute Percentage Error (MAPE)**: This metric measures the average percentage difference between the predicted prices and the actual prices.

- **Accuracy**: The accuracy of the model is also provided, which is calculated as 100% minus the MAPE.

## Usage

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/paris-housing-prediction.git
   ```

2. Install the required libraries if not already installed:

   ```
   pip install pandas numpy matplotlib scikit-learn
   ```

3. Run the Jupyter Notebook or Python script to train and evaluate the model:

   ```
   python housing_prediction.py
   ```

4. Examine the model's performance and explore the results.

