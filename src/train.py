import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# File paths
DATA_PATH = Path(
    "data/processed/cleaned_telco_churn.csv"
)

MODEL_PATH = Path(
    "models/churn_model.joblib"
)


def train_model():

    print("Loading processed dataset...")

    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "number"]
    ).columns.tolist()

    print("\nCategorical features:")
    print(categorical_features)

    print("\nNumerical features:")
    print(numeric_features)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ]
    )

    # Machine learning model
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    # Complete pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining model...")

    # Set MLflow experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco-Churn-Prediction")

    # Start MLflow run
    with mlflow.start_run():

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Log parameters
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("class_weight", "balanced")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Save model using MLflow
        mlflow.sklearn.log_model(
            model,
            "model"
        )

        # Print results
        print("\nModel Evaluation Results")
        print("------------------------")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Print MLflow run ID
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    # Save model locally
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()