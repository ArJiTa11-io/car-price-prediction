# 🚗 Used Car Price Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/pandas-data--analysis-blue)

A machine learning project that predicts used car prices based on features like
year, fuel type, and brand. Built with a Random Forest Regressor, achieving an
R² score of 0.95 on the test set.

## 📁 Project Structure

| File/Folder | What it does |
| :--- | :--- |
| `Data/car data.csv` | Raw dataset of used car listings |
| `src/preprocessing.py` | Cleans data and creates features (e.g. car age, encoding categories) |
| `src/train.py` | Splits data, trains the model, and saves it |
| `main.py` | Runs the full pipeline end-to-end |

## 🛠️ What I did

**1. Feature engineering**
- Created a `Car_Age` feature (`current year − manufacturing year`) instead of using
  raw year, since age is more directly related to price
- Converted categorical columns (fuel type, transmission, etc.) into numeric format
  using one-hot encoding, dropping the first category to avoid redundant columns

**2. Model training**
- Split data 80/20 into train and test sets
- Trained a Random Forest Regressor, which handles non-linear relationships between
  features better than plain linear regression (e.g. how fuel type interacts with
  brand to affect price)

## 📊 Results

| Model | MAE | R² Score |
| :--- | :--- | :--- |
| Linear Regression (baseline) | — | — |
| **Random Forest** | **0.72 (Lakhs)** | **0.95** |

An R² of 0.95 means the model explains 95% of the variation in car prices in this
dataset.

## ▶️ How to run

```bash
python main.py
```

This will load and clean the data, train the model, and print the results.
