# ============================================================
# DATA LOADER + FEATURE ENGINEERING — Hospital Readmission ML System
# ============================================================
# Dataset: Diabetes 130-US hospitals for years 1999-2008 (UCI / Kaggle)
# Target : readmitted_30d  (1 = readmitted within 30 days of discharge)
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Required raw columns from the source dataset ─────────────
REQUIRED_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses", "readmitted",
]

# ── Discharge disposition codes meaning "expired" or "hospice" ───
# These encounters cannot be readmitted -> excluded to prevent label noise
_TERMINAL_DISCHARGE_CODES = {11, 13, 14, 19, 20, 21}

# ── Near-zero-variance columns (>99% single value in source data) ─
_DROP_COLS = ["encounter_id", "patient_nbr", "weight", "payer_code",
              "examide", "citoglipton"]

_MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

_DIAG_GROUP_MAP_RANGES = [
    (390, 459, "circulatory"), (460, 519, "respiratory"),
    (520, 579, "digestive"),   (800, 999, "injury"),
    (710, 739, "musculoskeletal"), (580, 629, "genitourinary"),
    (140, 239, "neoplasms"),
]


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:

    # ── Normalize column names ────────────────────────────────
    df.columns = df.columns.str.strip()

    # ── Replace '?' sentinel with true NaN ────────────────────
    df.replace("?", np.nan, inplace=True)

    # ── Check required columns ───────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Drop irrelevant / near-zero-variance columns ─────────
    for col in _DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info("Dropped column: %s", col)

    # ── Fill missing categorical values ────────────────────────
    # race / medical_specialty / diag_* carry real missingness in the
    # source EHR extract — encode explicitly as "Missing" rather than
    # dropping rows (dropping would bias the sample toward complete
    # records, which is itself a form of selection leakage).
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        if col != "readmitted":
            df[col] = df[col].fillna("Missing")

    # ── Deduplicate at patient-encounter level ────────────────
    before = len(df)
    df.drop_duplicates(ignore_index=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)

    # ── Exclude expired / hospice discharges (cannot readmit) ─
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(_TERMINAL_DISCHARGE_CODES)].reset_index(drop=True)
    logger.info("Excluded %d expired/hospice encounters (non-readmittable)", before - len(df))

    # ── Build binary target: readmitted within 30 days ────────
    if df["readmitted"].dtype == object:
        df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
        df.drop(columns=["readmitted"], inplace=True)

    if not set(df["readmitted_30d"].unique()).issubset({0, 1}):
        raise ValueError("Target 'readmitted_30d' must contain only 0 and 1")

    # ── Minimum size check ────────────────────────────────────
    if df.shape[0] < 100:
        raise ValueError("Dataset too small for training (< 100 rows)")

    logger.info("Data validation passed  |  shape=%s  |  readmit_30d_rate=%.3f",
                df.shape, df["readmitted_30d"].mean())

    return df


# ============================================================
# DIAGNOSIS CODE GROUPING
# ============================================================

def _group_diag(code):
    if pd.isna(code) or str(code) == "Missing":
        return "missing"
    code = str(code)
    if code.startswith("250"):
        return "diabetes"
    if code.startswith(("V", "E")):
        return "other"
    try:
        num = float(code)
    except ValueError:
        return "other"
    for lo, hi, label in _DIAG_GROUP_MAP_RANGES:
        if lo <= num <= hi:
            return label
    return "other"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hospital-operations-grade feature engineering for 30-day
    readmission risk. All features map to real clinical / HRRP
    (Hospital Readmissions Reduction Program) signals used by
    care-management teams.
    """
    df = df.copy()

    # ── Age: midpoint of bracket e.g. "[70-80)" -> 75 ─────────
    if "age" in df.columns and df["age"].dtype == object:
        df["age_numeric"] = (
            df["age"].str.extract(r"\[(\d+)-(\d+)\)").astype(float).mean(axis=1)
        )

    # ── Prior healthcare utilization (strongest HRRP signal) ──
    df["prior_utilization"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["frequent_flyer_flag"] = (df["number_inpatient"] >= 3).astype(int)

    # ── Length-of-stay signals ─────────────────────────────────
    df["long_stay_flag"] = (df["time_in_hospital"] >= 7).astype(int)
    df["procedures_per_day"] = df["num_procedures"] / df["time_in_hospital"].replace(0, 1)
    df["labs_per_day"]       = df["num_lab_procedures"] / df["time_in_hospital"].replace(0, 1)

    # ── Medication burden (polypharmacy) ───────────────────────
    df["polypharmacy_flag"]   = (df["num_medications"] >= 15).astype(int)
    df["meds_per_day"]        = df["num_medications"] / df["time_in_hospital"].replace(0, 1)

    med_present = [c for c in _MED_COLS if c in df.columns]
    if med_present:
        df["n_diabetes_meds_active"] = (
            df[med_present].apply(lambda s: s.astype(str) != "No").sum(axis=1)
        )

    # ── Diagnosis complexity ────────────────────────────────────
    df["high_diagnosis_count_flag"] = (df["number_diagnoses"] >= 9).astype(int)

    for col in ("diag_1", "diag_2", "diag_3"):
        if col in df.columns:
            df[f"{col}_group"] = df[col].apply(_group_diag)

    # ── Discharge disposition risk grouping ─────────────────────
    # 1=home, 6=home health, 3/4/5/22/23/24=transferred facility
    transfer_codes = {2, 3, 4, 5, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30}
    if "discharge_disposition_id" in df.columns:
        df["discharged_to_facility"] = (
            df["discharge_disposition_id"].isin(transfer_codes)
        ).astype(int)

    # ── Admission type risk (1=emergency, 2=urgent, 7=trauma) ──
    if "admission_type_id" in df.columns:
        df["emergency_admission_flag"] = (
            df["admission_type_id"].isin([1, 2, 7])
        ).astype(int)

    # ── Glucose / A1C monitoring signals ────────────────────────
    if "A1Cresult" in df.columns:
        df["a1c_tested"] = (df["A1Cresult"].notna()).astype(int)
        df["a1c_abnormal"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)
    if "max_glu_serum" in df.columns:
        df["glucose_tested"] = (df["max_glu_serum"].notna()).astype(int)

    # ── Medication change during stay (destabilization signal) ─
    if "change" in df.columns:
        df["medication_changed_flag"] = (df["change"] == "Ch").astype(int)
    if "diabetesMed" in df.columns:
        df["on_diabetes_med_flag"] = (df["diabetesMed"] == "Yes").astype(int)

    # ── Interaction features ────────────────────────────────────
    df["age_x_prior_inpatient"] = df.get("age_numeric", 0) * df["number_inpatient"]
    df["los_x_meds"]            = df["time_in_hospital"] * df["num_medications"]

    logger.info("Feature engineering done  |  total columns=%d", df.shape[1])

    return df


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def detect_feature_types(df: pd.DataFrame, threshold: int = 12):
    """
    Auto-detect: ordinal / continuous / binary columns.
    Excludes target column 'readmitted_30d' and raw diag_1/2/3
    (replaced by grouped versions) plus raw diagnosis text codes.
    """
    ordinal_cols    = []
    continuous_cols = []
    binary_cols     = []

    exclude = {"readmitted_30d", "diag_1", "diag_2", "diag_3"}

    for col in df.columns:
        if col in exclude:
            continue

        dtype_name  = df[col].dtype.name
        n_unique    = df[col].nunique(dropna=False)

        if dtype_name in ("object", "category", "bool"):
            ordinal_cols.append(col)

        elif np.issubdtype(df[col].dtype, np.number):
            if n_unique == 2:
                binary_cols.append(col)
            elif 3 <= n_unique <= threshold:
                ordinal_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            ordinal_cols.append(col)

    logger.info("Feature types  |  ordinal=%d  continuous=%d  binary=%d",
                len(ordinal_cols), len(continuous_cols), len(binary_cols))

    return ordinal_cols, continuous_cols, binary_cols
