# System Architecture

```mermaid
flowchart LR
  A[IBM HR CSV] --> B[Preprocessing pipeline]
  B --> C[Model training and comparison]
  C --> D[Saved joblib pipeline]
  D --> E[FastAPI scoring service]
  D --> F[Streamlit HR dashboard]
  G[Scheduled retraining job] --> C
  E --> H[Validated prediction response]
  F --> I[Observed KPIs and model risk views]
```

The local implementation separates training, model persistence, scoring, and presentation. The IBM guide maps the same flow to Watson Studio Jobs and Watson Machine Learning after credentials become available.
