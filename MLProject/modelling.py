import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from dagshub import dagshub_logger
import dagshub

# dagshub.init(repo_owner='KrisnaSantosa15',
#              repo_name='stroke_prediction', mlflow=True)

# mlflow.set_tracking_uri(
#     "https://dagshub.com/KrisnaSantosa15/stroke_prediction.mlflow")

data = pd.read_csv("stroke_data_clean.csv")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("stroke", axis=1),
    data["stroke"],
    test_size=0.2,
    random_state=42,
    stratify=data["stroke"]
)

mlflow.autolog(disable=True)

input_example = X_train[0:5]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"KNN_Modelling_{timestamp}"

with mlflow.start_run(run_name=run_name):
    n_estimators = 100
    random_state = 42

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    input_example = X_train.iloc[:5]
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

print("\nTraining selesai. Metric accuracy:", accuracy)
