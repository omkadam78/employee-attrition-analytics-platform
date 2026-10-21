# Assignment 5441 Employee Attrition Analytics Platform

Subject: 24UADPC401 Data Science Advanced Programming IBM

This repository contains a working local employee attrition analytics platform. It compares a neural network with classical, Bayesian, Linear Discriminant Analysis, Quadratic Discriminant Analysis, and ensemble classifiers. The saved pipeline powers a real FastAPI service, while Streamlit presents HR metrics and model-risk views.

The application currently runs locally. It has **not** been deployed to IBM Cloud, and no live endpoint is claimed. `docs/IBM_DEPLOYMENT.md` explains the pending Watson Studio and Watson Machine Learning procedure.

## Verified result

| Accuracy | Precision | Recall | F1 | ROC-AUC | Mean CV ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 0.772 | 0.381 | 0.681 | 0.489 | 0.812 | 0.831 |

## Repository contents

| Path | Purpose |
|---|---|
| `data/employee_attrition.csv` | Bundled public IBM sample dataset |
| `src/train.py` | Advanced comparison, evaluation, SHAP output, and model persistence |
| `notebooks/advanced_analytics.ipynb` | Executed analytics notebook |
| `models/attrition_platform_model.joblib` | Actual trained preprocessing and Logistic Regression pipeline |
| `api/app.py` | FastAPI backend that loads and scores with the saved model |
| `dashboard/app.py` | Streamlit HR analytics dashboard |
| `workflow/retrain.py` | Reproducible local retraining entry point |
| `reports/Technical_Report.docx` | Technical report |
| `docs/API.md` | API request documentation |
| `docs/IBM_DEPLOYMENT.md` | Pending IBM deployment procedure |
| `docs/system_architecture.md` | Architecture diagram |
| `docs/Executive_Presentation.pptx` | Executive presentation |

## Installation

Python 3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Reproduce training and evaluation

```powershell
python src/train.py
```

This refreshes metrics, visualizations, scored test records, compatible SHAP output, and `models/attrition_platform_model.joblib`.

The retraining wrapper invokes the same workflow:

```powershell
python workflow/retrain.py
```

## Run the notebook

```powershell
jupyter lab notebooks/advanced_analytics.ipynb
```

Choose **Kernel > Restart Kernel and Run All Cells**.

## Run and test the FastAPI backend

`api/app.py` loads `models/attrition_platform_model.joblib` at startup. The `/predict` route orders the supplied fields to match training and calls the model's real `predict_proba` method.

```powershell
uvicorn api.app:app --reload
```

- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

The complete request schema and a JSON example are in `docs/API.md`. A verified example response format is:

```json
{
  "attrition_probability": 0.872656,
  "predicted_attrition": true,
  "risk_category": "High"
}
```

The probability is model-generated and may change after retraining. Run automated checks with:

```powershell
pytest -q
```

## Run the Streamlit dashboard

```powershell
streamlit run dashboard/app.py
```

Open the address printed by Streamlit, normally `http://localhost:8501`. The dashboard separates observed outcomes from model-generated risk categories.

## IBM deployment

Deployment remains pending. It requires an IBM Cloud account, Watson Studio and Watson Machine Learning services, a deployment space, protected credentials, a compatible software specification, and a verified scoring response. Follow `docs/IBM_DEPLOYMENT.md`. Never commit API keys or `.env` files.

## Dataset, safety, and limitations

The repository contains IBM's fictional public teaching dataset, generated model artifacts, reports, and source code. No known private employee records or credentials are included.

- The sample is fictional, small, and cross-sectional.
- The local API lacks production authentication, encryption, auditing, and rate limiting.
- Risk categories support review; they do not determine employment actions.
- Production use requires privacy, fairness, calibration, drift, and human-oversight controls.
- No IBM deployment, production latency, ROI, or financial savings is claimed.

Upload the contents of this folder to its own GitHub repository.
