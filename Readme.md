# 🏠 Morocco House Price Prediction

An end-to-end MLOps project that predicts housing prices in Morocco using a **ZenML + MLflow** training and deployment pipeline.

The project ingests a Moroccan real-estate dataset, runs full EDA, handles missing values and outliers, applies feature engineering, trains a Linear Regression model inside a scikit-learn pipeline, tracks experiments with MLflow, and finally deploys the model as a local prediction service.

---

## 📁 Project Structure

```
.
├── analysis/                  # Exploratory Data Analysis
│   ├── EDA.ipynb              # Main EDA notebook
│   └── analyze_src/           # Reusable analysis strategy classes
│       ├── basic_data_inspection.py
│       ├── missing_values_analysis.py
│       ├── univariate_analysis.py
│       ├── bivariate_analysis.py
│       └── multivariate_analysis.py
│
├── data/                      # Raw data
│   └── housing_data.zip
├── extracted_data/            # Auto-extracted from the zip
│   └── housing_data.csv
├── cleaned_data/              # Cleaned dataset used by the pipeline
│   └── cleaned_housing_data.csv
├── figures/                   # EDA plots (distributions, heatmaps, etc.)
│
├── src/                       # Core ML logic (Strategy pattern based)
│   ├── ingest_data.py
│   ├── handle_missing_values.py
│   ├── feature_engineering.py
│   ├── outlier_detection.py
│   ├── data_splitter.py
│   ├── model_building.py
│   └── model_evaluator.py
│
├── steps/                     # ZenML pipeline steps (wrappers around src/)
│   ├── data_ingestion_step.py
│   ├── handle_missing_values_step.py
│   ├── feature_engineering_step.py
│   ├── outlier_detection_step.py
│   ├── data_splitter_step.py
│   ├── model_building_step.py
│   ├── model_evaluator_step.py
│   ├── dynamic_importer.py
│   ├── model_loader.py
│   ├── prediction_service_loader.py
│   └── predictor.py
│
├── pipelines/                 # ZenML pipeline definitions
│   ├── training_pipeline.py
│   └── deployment_pipeline.py
│
├── tests/
│   └── predict.py             # Sample predictions on test houses
│
├── config.yaml                # ZenML model config
├── requirements.txt
├── run_pipeline.py            # Runs the training pipeline
└── run_deployment.py          # Runs the deployment + inference pipeline
```

---

## 🧠 Pipeline Overview

The training pipeline (`pipelines/training_pipeline.py`) chains the following ZenML steps:

1. **Data Ingestion** — loads `cleaned_housing_data.csv`.
2. **Missing Value Handling** — fills/drops missing values.
3. **Feature Engineering** — applies a `log` transform to skewed features (`surface_m2`, `price`).
4. **Outlier Detection** — removes outliers from the (log-transformed) target.
5. **Train/Test Split** — splits with `price` as the target.
6. **Model Building** — fits a `StandardScaler → LinearRegression` scikit-learn pipeline.
7. **Model Evaluation** — computes metrics (MSE etc.) and logs them via MLflow.

The deployment pipeline (`pipelines/deployment_pipeline.py`) re-trains the model and deploys it through MLflow's local model deployer, then runs an inference pipeline against the live service.

The codebase uses the **Strategy design pattern** throughout (`src/`), so each step (missing values, feature engineering, outliers, model building, evaluation) can be swapped with a different concrete strategy without touching the pipeline.

---

## 📊 Dataset

The dataset contains Moroccan housing listings with the following features:

| Feature         | Description                              |
|-----------------|------------------------------------------|
| `bedrooms`      | Number of bedrooms                       |
| `bathrooms`     | Number of bathrooms                      |
| `surface_m2`    | Surface area in m² (log-transformed)     |
| `floor`         | Floor number                             |
| `elevator`      | Has elevator (bool)                      |
| `terrace`       | Has terrace (bool)                       |
| `parking`       | Has parking (bool)                       |
| `property_type` | Apartment, Villa, Studio, etc.           |
| `city`          | Casablanca, Fès, Marrakech, …            |
| `price`         | **Target** — price in thousands of MAD   |

EDA outputs are saved under `figures/` (numeric and categorical distributions, correlation heatmaps, price drivers, price-per-m² by city, pairplot, etc.).

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone git@github.com:AhmedBassoul1/Morocco-House-Price-Prediction.git
cd Morocco_House_Price_Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Main libraries used: `zenml==0.64.0`, `mlflow==2.15.1`, `scikit_learn==1.3.2`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`.

### 4. Initialize ZenML and the MLflow integration

```bash
zenml init
zenml integration install mlflow -y

# Register the MLflow experiment tracker and model deployer
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer    register mlflow         --flavor=mlflow

# Register and activate a stack that uses them
zenml stack register local_mlflow_stack \
    -a default \
    -o default \
    -d mlflow \
    -e mlflow_tracker \
    --set
```

> ⚠️ **Heads-up:** `pipelines/training_pipeline.py` uses an absolute path to `cleaned_housing_data.csv`. Update the `file_path=` argument in `data_ingestion_step(...)` to match the location of `cleaned_data/cleaned_housing_data.csv` on your machine before running the pipeline.

---

## 🚀 Usage

### Run the training pipeline

```bash
python run_pipeline.py
```

After the run completes, inspect experiments in the MLflow UI:

```bash
mlflow ui --backend-store-uri "<tracking-uri-printed-by-the-script>"
```

### Run the continuous deployment + inference pipeline

```bash
python run_deployment.py
```

This trains the model, deploys it as a local MLflow service, and runs a batch inference pipeline against it. The script prints the prediction endpoint URL at the end.

To stop the running prediction service:

```bash
python run_deployment.py --stop-service
```

### Make sample predictions

`tests/predict.py` loads the latest registered ZenML model and predicts prices on three sample houses (a Casablanca apartment, a Fès villa, a Marrakech studio):

```bash
python tests/predict.py
```

It handles the same preprocessing the pipeline used during training (column ordering, bool→int, `log1p` on `surface_m2`, `expm1` on the prediction).

---

## 🧪 Exploratory Data Analysis

Open the EDA notebook to walk through the dataset:

```bash
jupyter notebook analysis/EDA.ipynb
```

The notebook uses the strategy classes in `analysis/analyze_src/` for basic inspection, missing-value analysis, and univariate / bivariate / multivariate analysis. Generated figures are saved to `figures/`.

---

## 🛠 Tech Stack

- **ZenML** — pipeline orchestration & artifact tracking
- **MLflow** — experiment tracking and model deployment
- **scikit-learn** — preprocessing pipeline + Linear Regression model
- **pandas / numpy** — data wrangling
- **matplotlib / seaborn / statsmodels** — EDA and visualizations

---

## 📌 Notes

- Prices in the dataset are stored in **thousands of MAD** (so `100.0` → `100,000 MAD`).
- The target (`price`) and `surface_m2` are log-transformed during training; predictions are inverse-transformed back to the original scale before being returned.
- The default model is a simple Linear Regression — thanks to the Strategy pattern in `src/model_building.py`, you can plug in any other regressor (Random Forest, Gradient Boosting, etc.) by adding a new `ModelBuildingStrategy`.

---

## 📄 License

Apache 2.0 (per `config.yaml`).