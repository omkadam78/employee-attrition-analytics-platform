from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "employee_attrition.csv"
OUT = ROOT / "reports" / "results"
VIS = ROOT / "visualizations"
MODELS = ROOT / "models"
for folder in (OUT, VIS, MODELS): folder.mkdir(parents=True, exist_ok=True)


def preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(exclude="object").columns.tolist()
    categorical = frame.select_dtypes(include="object").columns.tolist()
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])


def metrics(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred, prob = model.predict(x), model.predict_proba(x)[:, 1]
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0), "ROC-AUC": roc_auc_score(y, prob)}


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(DATA)
    x = df.drop(columns=["Attrition", "EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"])
    y = df["Attrition"].map({"No": 0, "Yes": 1})
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y, random_state=SEED)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", C=.1, random_state=SEED),
        "Gaussian Naive Bayes": GaussianNB(),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(reg_param=.2),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=600, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1),
    }
    rows, fitted = {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, estimator in models.items():
        pipe = Pipeline([("pre", preprocessor(x)), ("model", estimator)])
        pipe.fit(x_train, y_train)
        fitted[name] = pipe
        row = metrics(pipe, x_test, y_test)
        row["CV ROC-AUC"] = float(cross_val_score(pipe, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
        rows[name] = row
    table = pd.DataFrame(rows).T.sort_values("ROC-AUC", ascending=False)
    table.to_csv(OUT / "advanced_model_metrics.csv")
    best_name = str(table.index[0]); best = fitted[best_name]
    joblib.dump(best, MODELS / "attrition_platform_model.joblib")
    sample = x_test.copy(); sample["ObservedAttrition"] = y_test.values; sample["PredictedRisk"] = best.predict_proba(x_test)[:, 1]
    sample["RiskCategory"] = pd.cut(sample["PredictedRisk"], bins=[-1, .33, .66, 1], labels=["Low", "Medium", "High"])
    sample.to_csv(OUT / "scored_test_employees.csv", index=False)
    summary = {"selected_model": best_name, "rows": len(df), "attrition_rate": float(y.mean()), "test_metrics": table.loc[best_name].to_dict(), "selection_criterion": "highest held-out ROC-AUC with five-fold CV reported", "ibm_deployment_status": "not deployed because credentials were not supplied"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for name, model in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1]); ax.plot(fpr, tpr, label=f"{name} ({table.loc[name, 'ROC-AUC']:.3f})")
    ax.plot([0,1],[0,1],"--",color="#777777"); ax.set(title="Advanced model comparison", xlabel="False positive rate", ylabel="True positive rate"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(VIS / "advanced_roc_curves.png", dpi=180); plt.close(fig)
    if best_name == "Logistic Regression":
        transformed = best.named_steps["pre"].transform(x_test)
        explainer = shap.LinearExplainer(best.named_steps["model"], transformed)
        values = explainer(transformed[:200])
        shap.summary_plot(values, transformed[:200], feature_names=best.named_steps["pre"].get_feature_names_out(), show=False, max_display=12)
        plt.tight_layout(); plt.savefig(VIS / "shap_summary.png", dpi=180, bbox_inches="tight"); plt.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
