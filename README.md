# Data Preprocessing for Heart Disease Prediction

A hands-on machine learning lab that takes a messy patient dataset and turns it into model-ready features, then measures how each preprocessing step affects a KNN classifier.

## What's inside

- **Dataset:** 200 synthetic patient records (age, blood pressure, cholesterol, BMI, gender, city, smoker) with a `heart_disease` label and deliberately injected missing values
- **Missing data:** analysis of missing percentages, then mean vs. median imputation with `SimpleImputer`
- **Encoding:** `LabelEncoder` for binary columns and one-hot encoding for city
- **Scaling:** `StandardScaler` vs. `MinMaxScaler`, with before/after box plots
- **Feature selection:** correlation-based selection at several thresholds and its effect on KNN accuracy
- **Full workflow:** an end-to-end scikit-learn `Pipeline` from raw data to evaluation

## Key takeaway

Scaling made the biggest difference. Without it, large-range features such as cholesterol dominated KNN's distance calculations.

## Tech

Python · pandas · NumPy · scikit-learn · Matplotlib · Jupyter

## Run it

Open `lab3.ipynb` in Jupyter or Google Colab and run all cells.
