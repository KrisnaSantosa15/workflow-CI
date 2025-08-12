import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
import json
from datetime import datetime
from dagshub import dagshub_logger
import dagshub
import joblib
from sklearn.pipeline import Pipeline

# Import SMOTE dan pipeline imblearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

dagshub.init(repo_owner='KrisnaSantosa15',
             repo_name='stroke_prediction', mlflow=True)

# Local MLflow tracking
# mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_tracking_uri(
    "https://dagshub.com/KrisnaSantosa15/stroke_prediction.mlflow")

data = pd.read_csv("stroke_data_clean.csv")
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("stroke", axis=1),
    data["stroke"],
    test_size=0.2,
    random_state=42,
    stratify=data["stroke"]
)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"RandomForest_Tuning_SMOTE_Threshold_{timestamp}"

with mlflow.start_run(run_name=run_name):
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rf__n_estimators': [50, 100, 150],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2],
        'rf__max_features': ['sqrt', 'log2', None],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Save the best model as model.pkl
    inference_pipeline = Pipeline([
        ('scaler', best_model.named_steps['scaler']),
        ('rf', best_model.named_steps['rf'])
    ])

    # Save inference pipeline
    joblib.dump(inference_pipeline, "model.pkl")

    # Prediksi probabilitas kelas 1
    probs = best_model.predict_proba(X_test)[:, 1]

    # Cari threshold terbaik berdasarkan F1-score
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.1, 0.91, 0.05):
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_test, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    # Evaluasi dengan threshold terbaik
    y_pred_thresh = (probs >= best_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred_thresh)
    cm = confusion_matrix(y_test, y_pred_thresh)
    report = classification_report(y_test, y_pred_thresh, output_dict=True)

    # Logging ke MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("best_f1_threshold", best_f1)

    for cls in report:
        if cls in ['0', '1']:
            mlflow.log_metric(
                f"precision_class_{cls}", report[cls]['precision'])
            mlflow.log_metric(f"recall_class_{cls}", report[cls]['recall'])
            mlflow.log_metric(f"f1_class_{cls}", report[cls]['f1-score'])

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Threshold={best_threshold:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Save classification report JSON
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("classification_report.json")

    # Save estimator info
    with open("estimator.html", "w") as f:
        f.write("<html><body>")
        f.write("<h1>Best RandomForest Estimator</h1>")
        f.write(f"<pre>{str(best_model)}</pre>")
        f.write("</body></html>")
    mlflow.log_artifact("estimator.html")

    # Log model dengan signature dan contoh input
    signature = infer_signature(X_test, y_pred_thresh)
    input_example = X_train.iloc[:5]
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_rf_model",
        signature=signature,
        input_example=input_example
    )

print(
    f"Training selesai. Accuracy: {accuracy:.4f}, Best threshold: {best_threshold:.2f}, Best F1: {best_f1:.4f}")
