from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, roc_auc_score, classification_report)

project_folder = Path(__file__).resolve().parents[1]
output_folder = project_folder / "artifacts"
train = pd.read_csv(output_folder / "splits/train.csv")
test = pd.read_csv(output_folder / "splits/test.csv")
target = "Engine Condition"
X_train = train.drop(columns=target)
y_train = train[target]
X_test = test.drop(columns=target)
y_test = test[target]
data_hash = hashlib.sha256((project_folder / "data/engine_data.csv").read_bytes()).hexdigest()

def model_performance(model, predictors, actual):
    predicted = model.predict(predictors)
    probability = model.predict_proba(predictors)[:, 1]
    return pd.DataFrame({
        "Accuracy": [accuracy_score(actual, predicted)],
        "Precision (class 1)": [precision_score(actual, predicted, zero_division=0)],
        "Recall (class 1)": [recall_score(actual, predicted, zero_division=0)],
        "Recall (class 0)": [recall_score(actual, predicted, pos_label=0, zero_division=0)],
        "F1 (class 1)": [f1_score(actual, predicted, zero_division=0)],
        "Macro F1": [f1_score(actual, predicted, average="macro", zero_division=0)],
        "Balanced accuracy": [balanced_accuracy_score(actual, predicted)],
        "ROC-AUC": [roc_auc_score(actual, probability)]
    })

baseline = DummyClassifier(strategy="most_frequent", random_state=42)
baseline.fit(X_train, y_train)
model_performance(baseline, X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"roc_auc": "roc_auc", "f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy"}

dt_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("model", DecisionTreeClassifier(random_state=42))])
dt_parameters = {"model__max_depth": [3, 5, 8],
                 "model__min_samples_leaf": [10, 50],
                 "model__class_weight": [None, "balanced"]}
dt_search = GridSearchCV(dt_pipeline, dt_parameters, scoring=scoring, refit="roc_auc",
                         cv=cv, n_jobs=2, error_score="raise", verbose=1)
dt_search.fit(X_train, y_train)

print("Best parameters:", dt_search.best_params_)
print("Best CV ROC-AUC:", round(dt_search.best_score_, 2))

rf_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("model", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1))])
rf_parameters = {"model__max_depth": [6, 12],
                 "model__min_samples_leaf": [5, 20],
                 "model__class_weight": [None, "balanced"]}
rf_search = GridSearchCV(rf_pipeline, rf_parameters, scoring=scoring, refit="roc_auc",
                         cv=cv, n_jobs=2, error_score="raise", verbose=1)
rf_search.fit(X_train, y_train)

print("Best parameters:", rf_search.best_params_)
print("Best CV ROC-AUC:", round(rf_search.best_score_, 2))

gb_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("model", GradientBoostingClassifier(random_state=42))])
gb_parameters = {"model__n_estimators": [80, 160],
                 "model__learning_rate": [0.05, 0.10],
                 "model__max_depth": [1, 2]}
gb_search = GridSearchCV(gb_pipeline, gb_parameters, scoring=scoring, refit="roc_auc",
                         cv=cv, n_jobs=2, error_score="raise", verbose=1)
gb_search.fit(X_train, y_train)

print("Best parameters:", gb_search.best_params_)
print("Best CV ROC-AUC:", round(gb_search.best_score_, 2))

searches = {"Decision Tree": dt_search, "Random Forest": rf_search, "Gradient Boosting": gb_search}
comparison = pd.DataFrame({
    "Model": list(searches),
    "CV ROC-AUC": [search.best_score_ for search in searches.values()],
    "Best parameters": [search.best_params_ for search in searches.values()]
}).sort_values("CV ROC-AUC", ascending=False)
comparison

best_model_name = comparison.iloc[0]["Model"]
best_model = searches[best_model_name].best_estimator_
print("Selected model:", best_model_name)

output_folder.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("sqlite:///" + str((output_folder / "mlflow.db").resolve()))
client = mlflow.tracking.MlflowClient()
if client.get_experiment_by_name("capstone-predictive-maintenance") is None:
    client.create_experiment("capstone-predictive-maintenance", artifact_location=(output_folder / "mlflow_artifacts").resolve().as_uri())
mlflow.set_experiment("capstone-predictive-maintenance")

# Log every tried combination, not just the winning parameters
experiment_results = []
with mlflow.start_run(run_name="Model comparison") as parent_run:
    mlflow.log_params({"random_state": 42, "cv_folds": 5, "source_sha256": data_hash})
    for model_name, search in searches.items():
        for i, parameters in enumerate(search.cv_results_["params"]):
            candidate_model = clone(search.estimator).set_params(**parameters)
            score = float(search.cv_results_["mean_test_roc_auc"][i])
            with mlflow.start_run(run_name=f"{model_name} candidate {i + 1}", nested=True):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("imputer_strategy", "median")
                mlflow.log_params(candidate_model.named_steps["model"].get_params())
                mlflow.log_metrics({"cv_roc_auc": score,
                                    "cv_auc_std": float(search.cv_results_["std_test_roc_auc"][i]),
                                    "cv_f1_macro": float(search.cv_results_["mean_test_f1_macro"][i]),
                                    "cv_balanced_accuracy": float(search.cv_results_["mean_test_balanced_accuracy"][i])})
            experiment_results.append({"Model": model_name, "Parameters": parameters,
                                       "CV ROC-AUC": score,
                                       "CV AUC SD": float(search.cv_results_["std_test_roc_auc"][i]),
                                       "CV macro F1": float(search.cv_results_["mean_test_f1_macro"][i]),
                                       "CV balanced accuracy": float(search.cv_results_["mean_test_balanced_accuracy"][i])})
print(f"Logged {len(experiment_results)} candidate runs with parameters and validation metrics.")

experiment_results = pd.DataFrame(experiment_results)
experiment_results.to_csv(output_folder / "experiment_results.csv", index=False)
experiment_results
print(experiment_results)

experiment = client.get_experiment_by_name("capstone-predictive-maintenance")
runs = client.search_runs([experiment.experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'")
print("MLflow candidate runs:", len(runs))
assert len(runs) == len(experiment_results) == 28

baseline_test = model_performance(baseline, X_test, y_test)
selected_train = model_performance(best_model, X_train, y_train)
selected_test = model_performance(best_model, X_test, y_test)
performance_comparison = pd.concat([baseline_test, selected_train, selected_test])
performance_comparison.index = ["Baseline - Test", "Selected Model - Train", "Selected Model - Test"]
performance_comparison
performance_comparison.to_csv(output_folder / "performance_comparison.csv")
print(performance_comparison)

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, digits=2))
y_pred = best_model.predict(X_test)

model_folder = output_folder / "models"
model_folder.mkdir(parents=True, exist_ok=True)
model_path = model_folder / "best_model.joblib"
joblib.dump(best_model, model_path)
print("Saved:", model_path.name)

# Check that loading the saved model does not change its predictions
loaded_model = joblib.load(model_path)
np.testing.assert_array_equal(best_model.predict(X_test), loaded_model.predict(X_test))
print("Saved model predictions verified.")

model_details = {"selected_model": best_model_name,
                 "parameters": searches[best_model_name].best_params_,
                 "feature_columns": X_train.columns.tolist(),
                 "source_sha256": data_hash,
                 "class_meanings": "Not confirmed"}
(model_folder / "model_details.json").write_text(json.dumps(model_details, indent=2))
print("Model details saved.")
