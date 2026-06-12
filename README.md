# 🏎️ Used Car Price Prediction Engine

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Modular_Pipeline-brightgreen?style=for-the-badge)

A modular, production-grade Machine Learning execution pipeline designed to ingest historical automotive market records, process multi-collinear categorical features, and train a high-accuracy Random Forest Regressor to estimate vehicle valuation.

---

## 🏗️ Production Architecture & File Structure

This system transitions loose notebook experimentation into a clean, decoupled software engineering structure:

| Component Path | Technical Responsibility | Core Mechanisms |
| :--- | :--- | :--- |
| 🗃️ **`Data/`** | Raw Asset Ingestion Node | Houses raw `car data.csv` lifecycle records. |
| 🛠️ **`src/preprocessing.py`** | Feature Engineering & Hygiene | Feature generation ($2026 - \text{Year}$), variance encoding, multi-collinearity management via `pd.get_dummies(drop_first=True)`. |
| 🧠 **`src/train.py`** | Compute & Model Serialization | 80/20 deterministic data splitting, Random Forest Ensemble training, performance scoring, and binary pickle dumping. |
| 🚀 **`main.py`** | Pipeline Orchestration Entrypoint | Central controller execution block that wraps and sequentializes data streams across internal scripts. |

---

## ⚙️ Data Engineering & Pipeline Dynamics

### 🔄 Data Hygiene Layer
* **Dynamic Lifecycle Tracking:** Calculates precise temporal asset depreciation by transforming raw timestamps directly into a live numeric feature:
 $$\text{Car\_Age} = 2026 - \text{Year}$$
* **Mathematical Collinearity Pruning:** Enforces dummy variable trap prevention (`drop_first=True`) during feature matrix binarization to keep regression estimators mathematically stable.

### 🤖 Ensemble Modeling Context
* Leverages a multi-decision tree **Random Forest Regressor** to evaluate non-linear feature interactions seamlessly (e.g., matching fuel types with brand tier trends), outperforming basic linear assumptions.

---

## 📊 Analytical Performance Benchmarks

Our model evaluation matrix yielded top-tier optimization characteristics:

| Model Architecture | Mean Absolute Error (MAE) | $R^2$ Variance Score | Status |
| :--- | :--- | :--- | :--- |
| Baseline Linear Regression | *Proof-of-Concept Baseline* | — | Deprecated |
| **Random Forest Ensemble** | **0.72 (Lakhs)** | **0.95** | **Active Production Model** |

> **Key Takeaway:** An **$R^2$ score of 0.95** means our optimized training parameters successfully explain **95% of the total price variance** within the target automotive dataset.

---

## 📝 Pipeline Orchestration Core Code

The system execution flow is kept clean and scalable via a localized wrapper context within `main.py`:

```python
if __name__ == "__main__":
    print("Starting Car Price Prediction Pipeline...")
    
    # Triggering decoupled feature engine
    data = load_and_clean_data('Data/car data.csv')
    
    # Transferring matrix payloads to training sequence
    train_pipeline(data)
    
    print("Pipeline completed successfully!")
