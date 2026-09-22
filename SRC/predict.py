"""
Student Dropout Prediction
Reusable prediction utility for the final Random Forest model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "Models" / "random_forest.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "Artifacts" / "preprocessor.joblib"
FEATURES_PATH = PROJECT_ROOT / "Artifacts" / "selected_features.joblib"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_engineered_features(data):
    """
    Recreate the feature engineering used during Step 3.
    """

    data = data.copy()

    # Age groups used by the final model
    data["Age_Group"] = pd.cut(
        data["Age at enrollment"],
        bins=[0, 19, 24, 29, np.inf],
        labels=["Under 20", "20-24", "25-29", "30+"],
        right=True
    )

    # Parents share the same qualification category
    data["Parental_Qualification_Same"] = (
        data["Mother's qualification"]
        == data["Father's qualification"]
    ).astype(int)

    # Parents share the same occupation category
    data["Parental_Occupation_Same"] = (
        data["Mother's occupation"]
        == data["Father's occupation"]
    ).astype(int)

    # Early financial-risk indicator
    data["Financial_Risk_Flag"] = (
        (data["Debtor"] == 1)
        | (data["Tuition fees up to date"] == 0)
    ).astype(int)

    return data


# ============================================================
# LOAD MODEL ARTEFACTS
# ============================================================

def load_model_artifacts():
    """Load the final model and preprocessing artefacts."""

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    selected_features = joblib.load(FEATURES_PATH)

    return model, preprocessor, selected_features


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_dropout(data):
    """
    Generate dropout predictions and probabilities.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw student-level input data.

    Returns
    -------
    pandas.DataFrame
        Input data with predicted dropout class
        and dropout probability.
    """

    model, preprocessor, selected_features = load_model_artifacts()

    # Recreate Step 3 engineered features
    data_engineered = create_engineered_features(data)

    # Apply saved preprocessing
    X_processed = preprocessor.transform(data_engineered)

    # Convert processed data to DataFrame
    X_processed = pd.DataFrame(
        X_processed,
        columns=preprocessor.get_feature_names_out()
    )

    # Keep the selected features used by the final model
    X_selected = X_processed[selected_features]

    # Generate predictions
    predictions = model.predict(X_selected)

    # Generate dropout probabilities
    probabilities = model.predict_proba(X_selected)[:, 1]

    # Return original data plus predictions
    results = data.copy()

    results["Predicted_Dropout"] = predictions
    results["Dropout_Probability"] = probabilities

    return results


# ============================================================
# BASIC SCRIPT CHECK
# ============================================================

if __name__ == "__main__":

    print("Student Dropout Prediction")
    print("---------------------------")
    print("Model:", MODEL_PATH)
    print("Preprocessor:", PREPROCESSOR_PATH)
    print("Selected features:", FEATURES_PATH)
    print("Ready for prediction.")