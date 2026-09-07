# ============================================================
# PREDICTION SERVICE — Hospital Readmission ML System
# ============================================================

import pandas as pd
import logging

from src.readmission_engine import score_patient
from src.data_loader        import add_engineered_features

logger = logging.getLogger(__name__)


# ============================================================
# PREPARE FEATURES — for API inference
# ============================================================

def prepare_features(input_data: dict) -> pd.DataFrame:
    """
    Takes raw API input dict -> returns engineered feature DataFrame.
    Mirrors the training pipeline feature engineering exactly.
    """
    df = pd.DataFrame([input_data])
    df = add_engineered_features(df)
    return df


# ============================================================
# PREDICT — single patient
# ============================================================

def predict_patient(model, input_data: dict, threshold: float) -> dict:
    """
    Full prediction flow for one discharged patient:
      1. Feature engineering
      2. Model probability (30-day readmission risk)
      3. Readmission engine scoring (clinical rules + ML)

    Returns structured readmission-risk output dict.
    """
    df = prepare_features(input_data)

    try:
        prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        logger.error("predict_proba failed: %s", e)
        prob = 0.5

    row = df.iloc[0].to_dict()

    result = score_patient(row, prob, threshold)

    logger.info(
        "Prediction  |  prob=%.4f  band=%s  decision=%s",
        result["readmission_probability"],
        result["risk_band"],
        result["decision"]
    )

    return result
