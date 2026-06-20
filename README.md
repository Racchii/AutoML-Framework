intern id- CITS2218

# AutoML Framework (AutoML Nexus)

An intelligent, full-stack Automated Machine Learning (AutoML) framework designed to simplify the process of applying machine learning to tabular data. AutoML Nexus automates repetitive tasks such as data preprocessing, model selection, and hyperparameter tuning, allowing you to focus on the results.

## Features

- **Automated Data Preprocessing:** Automatically handles missing values and encodes categorical features using `scikit-learn` pipelines.
- **Model Training:** Supports both Classification and Regression tasks using industry-standard models including Random Forest, Logistic/Linear Regression, and XGBoost.
- **Hyperparameter Tuning:** Integrates **Optuna** for Bayesian hyperparameter optimization to find the best configuration for your models.
- **Feature Importance:** Automatically extracts and visualizes the top features driving your model's predictions.
- **Modern UI:** A sleek, premium, dark-themed React frontend built with Vite and vanilla CSS glassmorphism.

## Tech Stack

- **Backend:** Python, FastAPI, Scikit-learn, XGBoost, Optuna, Pandas
- **Frontend:** React, Vite, Vanilla CSS

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+

### 1. Start the Backend

Open a terminal and navigate to the `backend` directory:

```bash
cd backend
# Create a virtual environment if you haven't already
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn scikit-learn xgboost pandas numpy python-multipart pydantic optuna

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```
The backend API will be available at `http://localhost:8000`. You can view the interactive API docs at `http://localhost:8000/docs`.

### 2. Start the Frontend

Open a new terminal window and navigate to the `frontend` directory:

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
The frontend will be available at `http://localhost:5173` (or `http://localhost:5174` if the port is busy).

## Usage

1. Open the frontend in your browser.
2. Upload a CSV dataset (e.g., the included `sample_classification.csv` or `sample_regression.csv`).
3. Select the **Target Column** you want to predict.
4. Select the **Task Type** (Classification or Regression).
5. (Optional) Enable **Hyperparameter Tuning** and set the number of optimization trials.
6. Click **Start AutoML Pipeline** and wait for the results and feature importances to be displayed!
