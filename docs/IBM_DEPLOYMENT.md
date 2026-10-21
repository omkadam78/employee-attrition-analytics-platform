# IBM Watson Deployment Procedure

Cloud deployment was not completed because IBM Cloud credentials and a deployment space were not supplied.

1. Create Watson Studio and Watson Machine Learning services and a deployment space.
2. Set `IBM_CLOUD_API_KEY`, `WML_SPACE_ID`, and `WML_REGION` as protected environment variables. Never commit them.
3. Upload `models/attrition_platform_model.joblib` with a compatible scikit-learn software specification.
4. Create an online deployment and test its scoring endpoint with a record matching `docs/API.md`.
5. Create a Watson Studio Job that runs `workflow/retrain.py` on an approved schedule.
6. Add evaluation gates for ROC-AUC, recall, calibration, drift, and subgroup review before promoting a new model.
7. Record the deployment ID and verified scoring response only after the endpoint succeeds.
