# ============================================================
# READMISSION ENGINE — Hospital Readmission ML System
# ============================================================
# Hospital-operations-grade 3-tier discharge decision system:
#   ROUTINE_DISCHARGE    -> low risk, standard discharge
#   FOLLOW_UP_REQUIRED   -> borderline, schedule follow-up call / clinic visit
#   CARE_MANAGEMENT      -> high risk, enroll in transitional care program
#
# Decision hierarchy:
#   1. Hard clinical rules (override ML — patient safety first)
#   2. ML model probability + risk bands
# ============================================================

import pandas as pd
import logging

from src.config import (
    RISK_BANDS, MAX_PRIOR_INPATIENT_RULE,
    MAX_TIME_IN_HOSPITAL_RULE, POLYPHARMACY_RULE
)

logger = logging.getLogger(__name__)


# ============================================================
# RISK BAND — probability -> LOW / MEDIUM / HIGH
# ============================================================

def get_risk_band(prob: float, threshold: float = None) -> str:
    """
    Threshold-relative by default: bands are defined relative to the
    model's actual (post-calibration) decision threshold, using the same
    cutoffs score_patient() uses for CARE_MANAGEMENT / FOLLOW_UP_REQUIRED.
    This guarantees the displayed risk band can never contradict the
    discharge decision (e.g. "LOW" risk band next to a CARE_MANAGEMENT
    decision) — a real mismatch that showed up in production once
    isotonic calibration pushed the threshold down to ~0.125.

    If no threshold is given, falls back to the fixed RISK_BANDS from
    config (kept for cases with no trained/calibrated model yet).
    """
    if threshold is not None and threshold > 0:
        if prob >= threshold:
            return "HIGH"
        elif prob >= threshold * 0.6:
            return "MEDIUM"
        else:
            return "LOW"

    for band, (low, high) in RISK_BANDS.items():
        if low <= prob < high:
            return band
    return "HIGH"


# ============================================================
# READMISSION ENGINE — row-level decisions
# ============================================================

def readmission_engine(patient_df: pd.DataFrame, probs, threshold: float) -> list:
    """
    For each discharged patient -> returns discharge-planning decision.

    Rule priority:
      1. number_inpatient >= 3 (frequent flyer)  -> CARE_MANAGEMENT (hard rule)
      2. time_in_hospital >= 10 (long stay)       -> FOLLOW_UP_REQUIRED (flag)
      3. num_medications  >= 15 (polypharmacy)    -> FOLLOW_UP_REQUIRED (flag)
      4. prob >= threshold                        -> CARE_MANAGEMENT (ML model)
      5. prob >= threshold * 0.6                  -> FOLLOW_UP_REQUIRED (ML model)
      6. else                                     -> ROUTINE_DISCHARGE
    """
    decisions = []

    for idx, (_, row) in enumerate(patient_df.iterrows()):

        p = probs[idx]

        # ── Hard rule: frequent prior inpatient admissions ────
        if "number_inpatient" in row and row["number_inpatient"] >= MAX_PRIOR_INPATIENT_RULE:
            decisions.append("CARE_MANAGEMENT_RULE")
            continue

        # ── Soft rule: unusually long inpatient stay ──────────
        if "time_in_hospital" in row and row["time_in_hospital"] >= MAX_TIME_IN_HOSPITAL_RULE:
            decisions.append("FOLLOW_UP_LONG_STAY")
            continue

        # ── Soft rule: polypharmacy ─────────────────────────────
        if "num_medications" in row and row["num_medications"] >= POLYPHARMACY_RULE:
            decisions.append("FOLLOW_UP_POLYPHARMACY")
            continue

        # ── ML model decisions ─────────────────────────────────
        if p >= threshold:
            decisions.append("CARE_MANAGEMENT_MODEL")

        elif p >= threshold * 0.6:
            decisions.append("FOLLOW_UP_MODEL")

        else:
            decisions.append("ROUTINE_DISCHARGE")

    return decisions


# ============================================================
# READMISSION SCORING — single patient (for API)
# ============================================================

def score_patient(row: dict, prob: float, threshold: float) -> dict:
    """
    Returns structured readmission-risk output for a single patient.
    Used by the FastAPI prediction endpoint.
    """
    risk_band = get_risk_band(prob, threshold)

    rule_triggered = None

    if row.get("number_inpatient", 0) >= MAX_PRIOR_INPATIENT_RULE:
        decision       = "CARE_MANAGEMENT"
        rule_triggered = "FREQUENT_PRIOR_INPATIENT"

    elif row.get("time_in_hospital", 0) >= MAX_TIME_IN_HOSPITAL_RULE:
        decision       = "FOLLOW_UP_REQUIRED"
        rule_triggered = "LONG_INPATIENT_STAY"

    elif row.get("num_medications", 0) >= POLYPHARMACY_RULE:
        decision       = "FOLLOW_UP_REQUIRED"
        rule_triggered = "POLYPHARMACY"

    elif prob >= threshold:
        decision = "CARE_MANAGEMENT"

    elif prob >= threshold * 0.6:
        decision = "FOLLOW_UP_REQUIRED"

    else:
        decision = "ROUTINE_DISCHARGE"

    return {
        "readmission_probability": round(float(prob), 4),
        "risk_band":               risk_band,
        "decision":                decision,
        "rule_triggered":          rule_triggered,
    }