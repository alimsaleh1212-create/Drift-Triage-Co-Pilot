import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.ml.data import clean_data, get_feature_columns, load_raw_data, split_data
from src.ml.pipeline import build_preprocessor


SEED = 42
MIN_RECALL = 0.75

DATA_PATH = Path("data/raw/bank-additional-full.csv")

MODEL_OUTPUT_PATH = Path("artifacts/models/bank_marketing_model.joblib")
REPORT_OUTPUT_PATH = Path("artifacts/reports/training_report.json")
THRESHOLD_OUTPUT_PATH = Path("artifacts/reports/operating_threshold.json")
MODEL_CARD_PATH = Path("artifacts/reports/model_card.md")
ENV_OUTPUT_PATH = Path("artifacts/reports/environment.json")

REGISTRY_DIR = Path("mlruns_registry")
MODEL_NAME = "bank-marketing-classifier"
EXPERIMENT_NAME = "week5-bank-marketing"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def to_jsonable(obj: Any) -> Any:
    """
    Convert numpy/pandas objects to normal Python types so json.dump works safely.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    return obj


def sha256_of(path: Path) -> str:
    """
    Create a SHA-256 fingerprint for the saved model artifact.
    This proves exactly which model file was registered.
    """
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def build_model_pipeline(
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Build a full sklearn pipeline:
    preprocessing + classifier.

    All models share the same preprocessing so the comparison is fair.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
            solver="lbfgs",
        )

    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )

    elif model_name == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def find_operating_threshold(
    y_true,
    y_proba,
    min_recall: float = MIN_RECALL,
) -> dict:
    """
    Choose the highest threshold where validation recall >= min_recall.

    This follows the project rule:
    do not blindly use 0.5; choose an operating threshold on validation data.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # precision_recall_curve returns one more precision/recall value than thresholds
    precisions_for_thresholds = precisions[:-1]
    recalls_for_thresholds = recalls[:-1]

    f1_scores = (
        2 * precisions_for_thresholds * recalls_for_thresholds
    ) / np.where(
        (precisions_for_thresholds + recalls_for_thresholds) > 0,
        precisions_for_thresholds + recalls_for_thresholds,
        1.0,
    )

    best_f1_index = int(np.argmax(f1_scores))
    f1_optimal_threshold = float(thresholds[best_f1_index])

    recall_mask = recalls_for_thresholds >= min_recall

    if recall_mask.any():
        valid_thresholds = thresholds[recall_mask]
        operating_threshold = float(valid_thresholds.max())
    else:
        operating_threshold = f1_optimal_threshold

    return {
        "operating_threshold": operating_threshold,
        "f1_optimal_threshold": f1_optimal_threshold,
        "best_validation_f1": float(f1_scores[best_f1_index]),
        "rule": f"highest threshold where validation recall >= {min_recall}",
    }


def evaluate_with_threshold(y_true, y_proba, threshold: float) -> dict:
    """
    Evaluate model probabilities using a chosen decision threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["no", "yes"],
            output_dict=True,
            zero_division=0,
        ),
    }


def train_and_compare_models(
    X_train,
    y_train,
    X_val,
    y_val,
    numeric_features,
    categorical_features,
) -> tuple[str, dict]:
    """
    Train candidate models on train set.
    Tune threshold on validation set.
    Select best model by validation F1 at the operating threshold.
    """
    candidate_names = [
        "logistic_regression",
        "random_forest",
        "extra_trees",
    ]

    results = {}

    for model_name in candidate_names:
        print(f"\nTraining {model_name}...")

        pipeline = build_model_pipeline(
            model_name=model_name,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        pipeline.fit(X_train, y_train)

        train_proba = pipeline.predict_proba(X_train)[:, 1]
        val_proba = pipeline.predict_proba(X_val)[:, 1]

        threshold_info = find_operating_threshold(
            y_true=y_val,
            y_proba=val_proba,
            min_recall=MIN_RECALL,
        )

        threshold = threshold_info["operating_threshold"]

        train_metrics = evaluate_with_threshold(
            y_true=y_train,
            y_proba=train_proba,
            threshold=threshold,
        )

        val_metrics = evaluate_with_threshold(
            y_true=y_val,
            y_proba=val_proba,
            threshold=threshold,
        )

        train_val_auc_gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]

        results[model_name] = {
            "pipeline": pipeline,
            "threshold_info": threshold_info,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "train_val_auc_gap": train_val_auc_gap,
        }

        print(
            f"{model_name} | "
            f"threshold={threshold:.6f} | "
            f"val_auc={val_metrics['roc_auc']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"gap={train_val_auc_gap:.4f}"
        )

    best_model_name = max(
        results,
        key=lambda name: results[name]["validation_metrics"]["f1"],
    )

    return best_model_name, results


def setup_mlflow() -> None:
    """
    Set up local MLflow tracking using SQLite metadata and local artifact storage.
    """
    REGISTRY_DIR.mkdir(exist_ok=True)
    (REGISTRY_DIR / "artifacts").mkdir(exist_ok=True)

    tracking_uri = f"sqlite:///{(REGISTRY_DIR / 'mlflow.db').as_posix()}"
    artifact_root = (REGISTRY_DIR / "artifacts").resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    existing_experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if existing_experiment is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=artifact_root,
        )

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nMLflow tracking URI:", mlflow.get_tracking_uri())
    print("MLflow experiment:", EXPERIMENT_NAME)


def create_environment_fingerprint() -> dict:
    """
    Capture versions of the environment used to train/register this model.
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "mlflow": mlflow.__version__,
        "captured_at": utc_now(),
    }


def write_model_card(
    *,
    model_name: str,
    selected_model: str,
    artifact_hash: str,
    env_meta: dict,
    split_info: dict,
    threshold: float,
    test_metrics: dict,
    train_val_auc_gap: float,
) -> None:
    """
    Write a simple model card as markdown.
    """
    env_json = json.dumps(env_meta, indent=2)

    card = f"""
# Model Card — {model_name}

**Status:** Staging candidate  
**Created at:** {utc_now()}  
**Selected model:** `{selected_model}`  
**Artifact hash:** `{artifact_hash}`

## Intended use

Predict whether a client will subscribe to a term deposit using the UCI Bank Marketing dataset.

The model outputs a probability for the positive class:

- `yes` = client subscribes
- `no` = client does not subscribe

## Training data

Dataset: UCI Bank Marketing, `bank-additional-full.csv`

Split:

| Split | Rows |
|---|---:|
| Train | {split_info["train_rows"]} |
| Validation | {split_info["validation_rows"]} |
| Test | {split_info["test_rows"]} |

The split is stratified 60/20/20 with `random_state=42`.

## Important data rules

- `duration` is dropped because it leaks the target.
- `pdays == 999` is treated as a sentinel, with an extra `pdays_was_999` flag.
- `"unknown"` values are kept as real categories, not deleted.
- Categorical features are encoded using `OneHotEncoder(handle_unknown="ignore")`.

## Architecture

sklearn `Pipeline`:

1. `ColumnTransformer`
   - Numeric: median imputation + scaling
   - Categorical: most-frequent imputation + one-hot encoding
2. Classifier: `{selected_model}`

## Operating threshold

Chosen threshold: `{threshold}`

Rule: highest threshold where validation recall >= {MIN_RECALL}

## Final test metrics

| Metric | Value |
|---|---:|
| AUC | {test_metrics["roc_auc"]:.4f} |
| F1 | {test_metrics["f1"]:.4f} |
| Precision | {test_metrics["precision"]:.4f} |
| Recall | {test_metrics["recall"]:.4f} |
| Accuracy | {test_metrics["accuracy"]:.4f} |

## Generalization check

Train-Val AUC gap for selected model: `{train_val_auc_gap:.4f}`

A large gap would suggest overfitting.

## Known limitations

- The positive class is rare, so precision is expected to be low when recall is forced near 0.75.
- The model is trained on historical campaign data and may drift if economic indicators or customer behavior shift.
- The threshold is a deployment policy and should be logged with every prediction.

## Environment fingerprint

{env_json}
"""

    MODEL_CARD_PATH.write_text(card.strip(), encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    raw_df = load_raw_data(DATA_PATH)
    clean_df = clean_data(raw_df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean_df)

    numeric_features, categorical_features = get_feature_columns(X_train)

    print("\nSplit sizes:")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    print("\nFeatures:")
    print(f"Numeric: {len(numeric_features)}")
    print(f"Categorical: {len(categorical_features)}")

    best_model_name, results = train_and_compare_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    best_pipeline = results[best_model_name]["pipeline"]
    best_threshold = results[best_model_name]["threshold_info"]["operating_threshold"]

    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_with_threshold(
        y_true=y_test,
        y_proba=test_proba,
        threshold=best_threshold,
    )

    model_artifact = {
        "registered_model_name": MODEL_NAME,
        "selected_model": best_model_name,
        "pipeline": best_pipeline,
        "threshold": best_threshold,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_mapping": {
            "no": 0,
            "yes": 1,
        },
        "created_at": utc_now(),
    }

    joblib.dump(model_artifact, MODEL_OUTPUT_PATH)

    artifact_hash = sha256_of(MODEL_OUTPUT_PATH)
    env_meta = create_environment_fingerprint()

    split_info = {
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
    }

    threshold_artifact = {
        "registered_model_name": MODEL_NAME,
        "selected_model": best_model_name,
        "operating_threshold": best_threshold,
        "rule": f"highest threshold where validation recall >= {MIN_RECALL}",
        "created_at": utc_now(),
    }

    with THRESHOLD_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(threshold_artifact), f, indent=2)

    with ENV_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(env_meta), f, indent=2)

    report = {
        "created_at": utc_now(),
        "data_path": str(DATA_PATH),
        "random_state": SEED,
        "split": split_info,
        "target_distribution": {
            "train": y_train.value_counts(normalize=True).to_dict(),
            "validation": y_val.value_counts(normalize=True).to_dict(),
            "test": y_test.value_counts(normalize=True).to_dict(),
        },
        "selected_model": best_model_name,
        "selected_threshold": best_threshold,
        "threshold_rule": f"highest threshold where validation recall >= {MIN_RECALL}",
        "model_artifact_path": str(MODEL_OUTPUT_PATH),
        "model_artifact_sha256": artifact_hash,
        "validation_results": {
            model_name: {
                "threshold_info": result["threshold_info"],
                "train_metrics": result["train_metrics"],
                "validation_metrics": result["validation_metrics"],
                "train_val_auc_gap": result["train_val_auc_gap"],
            }
            for model_name, result in results.items()
        },
        "test_metrics": test_metrics,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_columns": ["duration"],
        "special_handling": {
            "duration": "Dropped because it leaks the target.",
            "pdays": "Added pdays_was_999 flag because pdays=999 is a sentinel value.",
            "unknown": "Kept unknown as a real category.",
        },
        "environment": env_meta,
    }

    with REPORT_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, indent=2)

    write_model_card(
        model_name=MODEL_NAME,
        selected_model=best_model_name,
        artifact_hash=artifact_hash,
        env_meta=env_meta,
        split_info=split_info,
        threshold=best_threshold,
        test_metrics=test_metrics,
        train_val_auc_gap=results[best_model_name]["train_val_auc_gap"],
    )

    setup_mlflow()

    signature = infer_signature(
        X_train.head(100),
        best_pipeline.predict_proba(X_train.head(100)),
    )

    with mlflow.start_run(run_name=f"train-{best_model_name}") as run:
        mlflow.log_params(
            {
                "selected_model": best_model_name,
                "random_state": SEED,
                "min_recall": MIN_RECALL,
                "operating_threshold": best_threshold,
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
            }
        )

        mlflow.log_metrics(
            {
                "test_auc": test_metrics["roc_auc"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_accuracy": test_metrics["accuracy"],
                "train_val_auc_gap": results[best_model_name]["train_val_auc_gap"],
            }
        )

        for key, value in env_meta.items():
            mlflow.set_tag(f"env.{key}", value)

        mlflow.set_tag("artifact.sha256", artifact_hash)
        mlflow.set_tag("dataset", "UCI Bank Marketing bank-additional-full.csv")
        mlflow.set_tag("dropped.duration", "true")
        mlflow.set_tag("pdays_999_handling", "pdays_was_999 flag")

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(2),
            registered_model_name=MODEL_NAME,
        )

        mlflow.log_artifact(str(REPORT_OUTPUT_PATH), artifact_path="reports")
        mlflow.log_artifact(str(THRESHOLD_OUTPUT_PATH), artifact_path="reports")
        mlflow.log_artifact(str(MODEL_CARD_PATH), artifact_path="docs")
        mlflow.log_artifact(str(ENV_OUTPUT_PATH), artifact_path="reports")

        run_id = run.info.run_id

    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = sorted(versions, key=lambda mv: int(mv.version))[-1]

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version.version,
        stage="Staging",
        archive_existing_versions=True,
    )

    staging_uri = f"models:/{MODEL_NAME}/Staging"
    registered_pipeline = mlflow.sklearn.load_model(staging_uri)

    source_proba = best_pipeline.predict_proba(X_val.head(10))
    registered_proba = registered_pipeline.predict_proba(X_val.head(10))

    max_diff = float(np.max(np.abs(source_proba - registered_proba)))

    assert np.allclose(
        source_proba,
        registered_proba,
        atol=1e-12,
    ), "Registered model predictions differ from source model."

    print("\n==============================")
    print("Training complete")
    print("==============================")
    print("Best model:", best_model_name)
    print("Selected threshold:", best_threshold)
    print("Test recall:", round(test_metrics["recall"], 4))
    print("Test precision:", round(test_metrics["precision"], 4))
    print("Test F1:", round(test_metrics["f1"], 4))
    print("Test AUC:", round(test_metrics["roc_auc"], 4))

    print("\nArtifacts:")
    print(f"Model joblib: {MODEL_OUTPUT_PATH}")
    print(f"Training report: {REPORT_OUTPUT_PATH}")
    print(f"Threshold config: {THRESHOLD_OUTPUT_PATH}")
    print(f"Model card: {MODEL_CARD_PATH}")
    print(f"Environment: {ENV_OUTPUT_PATH}")

    print("\nMLflow:")
    print(f"Run ID: {run_id}")
    print(f"Registered model: {MODEL_NAME}")
    print(f"Version: {latest_version.version}")
    print(f"Stage URI: {staging_uri}")
    print(f"Max registry fidelity diff: {max_diff:.2e}")
    print("Registered model matches source model to 1e-12.")


if __name__ == "__main__":
    main()