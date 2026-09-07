# ============================================================
# CONFIGURATION — Hospital Readmission Prediction ML System
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE   = 42
N_JOBS         = -1

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS             = 3
RANDOM_SEARCH_ITERS  = 5

# ── Feature selection ────────────────────────────────────────
SELECT_K = 25

# ── Outlier clipping ─────────────────────────────────────────
CLIP_FOLD = 1.5

# ── Feature engineering thresholds ───────────────────────────
ORDINAL_UNIQUE_THRESHOLD = 12

# ── Risk bands (probability → tier) ──────────────────────────
# Clinical readmission risk tiers used to drive discharge workflow
RISK_BANDS = {
    "LOW":    (0.00, 0.20),
    "MEDIUM": (0.20, 0.45),
    "HIGH":   (0.45, 1.01),
}

# ── Business rule thresholds (clinical hard rules) ────────────
MAX_PRIOR_INPATIENT_RULE   = 3     # >=3 inpatient visits in prior year -> auto flag
MAX_TIME_IN_HOSPITAL_RULE  = 10    # long stay (days) -> auto flag for discharge planning
POLYPHARMACY_RULE          = 15    # >=15 distinct medications -> flag for pharmacist review

# ── PSI drift thresholds ──────────────────────────────────────
PSI_MODERATE         = 0.10    # PSI >= 0.10 -> moderate drift, monitor
PSI_HIGH             = 0.20    # PSI >= 0.20 -> critical drift, retrain

# ── Score monitoring alert ────────────────────────────────────
SCORE_MEAN_ALERT     = 0.30

# ── Challenger promotion gates ────────────────────────────────
MIN_F1_IMPROVEMENT      = 0.005
# NOTE: 0.68 was the credit-risk project's gate (where ROC-AUC reached 0.999).
# 30-day readmission on this dataset has a published ceiling of ~0.65-0.68
# ROC-AUC (see README "Model Results"); a 0.68 gate would permanently reject
# every challenger, defeating the purpose of the promotion system. 0.60 is
# the realistic bar for this specific domain/dataset.
MIN_ROCAUC_THRESHOLD    = 0.60
MAX_GENERALIZATION_GAP  = 0.10

# ── Paths ────────────────────────────────────────────────────
MODEL_DIR   = "readmission_models"
METRICS_LOG = "readmission_models/metrics_log.csv"
TESTS_DIR   = "tests"
SERVING_DIR = "serving"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)