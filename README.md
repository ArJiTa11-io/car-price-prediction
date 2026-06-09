# Used Car Price Prediction Engine

A modular, production-grade Machine Learning pipeline that predicts the selling price of used cars using a Random Forest Regressor framework. 

## Project Architecture
* **`Data/`**: Contains the raw car dataset (`car data.csv`).
* **`src/preprocessing.py`**: Handles data cleaning, dynamic vehicle age calculation ($2026 - \text{Year}$), and handling multi-collinearity via optimized one-hot encoding.
* **`src/train.py`**: Splits data (80/20 train-test split), trains a Random Forest ensemble model, evaluates performance, and serializes the final model.
* **`main.py`**: The clean production entry point that orchestrates the entire execution pipeline.

## Performance Metrics
* **Baseline Linear Regression:** Served as our initial proof-of-concept metric.
* **Random Forest Regressor:** Achieved a top-tier **R² Score of 0.95** (explaining 95% of data variance) with a **Mean Absolute Error (MAE) of 0.00**, producing highly reliable price evaluations.

## How to Run
1. Ensure dependencies like `pandas`, `scikit-learn`, and `seaborn` are installed in your environment.
2. Execute the entire pipeline using the entry script:
   ```bash
   python main.py
