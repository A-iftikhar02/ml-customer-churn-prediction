# Customer Churn Prediction

## Overview
This project predicts whether a telecom customer is likely to churn (leave the service) based on various features like tenure, service subscriptions, and monthly charges. It uses a **RandomForestClassifier** trained on the Telco Customer Churn dataset, with a **Streamlit web app** for user interaction.

---

## Features
- **Interactive Web App**: Built with Streamlit for easy predictions.
- **Machine Learning Model**: RandomForestClassifier trained on processed customer data.
- **Data Preprocessing**: Includes handling of categorical features, encoding, and scaling.
- **Saved Model Pipeline**: Preprocessing steps and model are stored together for consistent inference.
- **Probability-Based Output**: The app shows if a customer is *Likely to Churn* or *Not Likely to Churn* based on probability thresholds.

---

## Development Process & File Purpose

We have **two main files** for model training:

### 1. `01_exploration.ipynb`
- **Purpose**: Used for *Exploratory Data Analysis (EDA)* and initial model experimentation.
- **Why a Notebook?**
  - Allows quick visualizations and trial of different preprocessing steps.
  - Easier to debug and adjust model parameters interactively.
- **What it contains?**
  - Data exploration (missing values, distributions, correlations).
  - Preprocessing trials (encoding, scaling, feature selection).
  - Model selection and evaluation experiments.

### 2. `train_model.py`
- **Purpose**: Final *production-ready* script for training the model.
- **Why a Script?**
  - Automates model training without manual intervention.
  - Ensures consistent preprocessing and feature engineering steps every time.
  - Saves the trained model (`.pkl` file) and preprocessing pipeline.
- **Workflow**:
  1. Reads raw dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
  2. Preprocesses data based on the pipeline finalized in `01_exploration.ipynb`.
  3. Trains the RandomForest model with optimal hyperparameters.
  4. Saves the trained pipeline using `joblib` for use in the Streamlit app.

**Reason for Both Files**:  
Keeping EDA and production code separate is an **MLOps best practice**. The notebook is for research and experimentation, while the script is for reproducible, automated training.

---

## Challenges & Solutions

### 1. **Model Version Compatibility Issue**
- **Problem**: Old model pickle caused errors due to updated scikit-learn internal structure.
- **Solution**: Retrained the model using the current scikit-learn version and saved with `joblib`.

### 2. **Mismatched Features at Prediction**
- **Problem**: Model expected 30 features, but the app provided fewer due to missing preprocessing pipeline.
- **Solution**: Saved preprocessing steps together with the model to ensure consistent feature transformation.

### 3. **Data Path Errors**
- **Problem**: `train_model.py` couldn’t find the dataset.
- **Solution**: Used relative paths carefully and ensured dataset is stored in `data/` folder.

### 4. **Threshold for Churn Prediction**
- **Logic**: If churn probability ≥ 0.5 → "Likely to Churn", else → "Not Likely to Churn".

---

## Project Structure
```
customer_churn_prediction/
│
├── app/
│   └── app.py                # Streamlit app for predictions
│
├── scripts/
│   └── train_model.py        # Model training script
│
├── notebooks/
│   └── 01_exploration.ipynb  # EDA and experimentation
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl       # Trained model + preprocessing pipeline
│
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/A-iftikhar02/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (optional if model.pkl is already provided)
```bash
python scripts/train_model.py
```

### 4. Run the Streamlit app
```bash
streamlit run app/app.py
```

---

## Links
- **GitHub**: [A-iftikhar02](https://github.com/A-iftikhar02)
- **LinkedIn**: [Abdullah Iftikhar](https://www.linkedin.com/in/abdullah-iftikhar-a67986322/)

---

## Author
**Abdullah Iftikhar** – Data Scientist | Machine Learning Engineer
