# ============================================================
# PYTEST UNIT TESTS — Hospital Readmission ML System
# ============================================================
# Run with:  pytest tests/test_pipeline_core.py -v
#            pytest tests/ -v --cov=src --cov-report=term-missing
#
# Tests cover:
#   Clipper, build_preprocessors, detect_feature_types,
#   detect_leakage, tune_threshold, psi, recall_at_k,
#   lift_at_k, ks_statistic, get_risk_band, score_patient,
#   readmission_engine, cost_sensitive_evaluation, config thresholds
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd

from src.preprocessing      import Clipper, build_preprocessors
from src.data_loader        import detect_feature_types, add_engineered_features, validate_input_data
from src.leakage_check      import detect_leakage
from src.metrics            import tune_threshold, psi, recall_at_k, lift_at_k, ks_statistic, cost_sensitive_evaluation
from src.readmission_engine import get_risk_band, score_patient, readmission_engine
from src.config              import (
    RISK_BANDS, PSI_MODERATE, PSI_HIGH,
    MIN_F1_IMPROVEMENT, MIN_ROCAUC_THRESHOLD, MAX_GENERALIZATION_GAP,
    MAX_PRIOR_INPATIENT_RULE, MAX_TIME_IN_HOSPITAL_RULE, POLYPHARMACY_RULE
)


# ============================================================
# CLIPPER TESTS  (6 tests)
# ============================================================

class TestClipper:

    def test_fit_transform_shape(self):
        X = np.array([[1.0], [1000.0], [2.0], [3.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape == X.shape

    def test_clips_outliers(self):
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).max() < 9999.0, "Outlier should have been clipped"

    def test_no_change_on_normal_data(self):
        X = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        np.testing.assert_array_almost_equal(X, clip.transform(X), decimal=3)

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0, 1000.0])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape[0] == 4

    def test_get_feature_names_out(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        clip = Clipper()
        clip.fit(X)
        names = clip.get_feature_names_out(["a", "b"])
        assert list(names) == ["a", "b"]

    def test_fit_on_train_applied_to_test(self):
        """Clipper fitted on train must use training bounds on test — not re-fit (leakage prevention)."""
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        X_test  = np.array([[100.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X_train)
        clipped = clip.transform(X_test)
        assert clipped[0, 0] < 100.0, "Test outlier should be clipped to training bounds"


# ============================================================
# PREPROCESSOR TESTS  (5 tests)
# ============================================================

class TestBuildPreprocessors:

    def _sample_df(self):
        return pd.DataFrame({
            "time_in_hospital":  [1, 3, 5, 7, 9],
            "num_medications":   [3.0, 8.0, 15.0, 22.0, 30.0],
            "number_diagnoses":  [1, 3, 5, 7, 9],
            "admission_type_id": [1, 2, 3, 1, 2],
            "on_diabetes_med_flag": [0, 1, 0, 1, 0],
        })

    def test_returns_four_outputs(self):
        df = self._sample_df()
        result = build_preprocessors(
            ["admission_type_id", "number_diagnoses"],
            ["time_in_hospital", "num_medications"],
            ["on_diabetes_med_flag"], df
        )
        assert len(result) == 4

    def test_categorical_indices_are_list(self):
        df = self._sample_df()
        _, _, cat_idx, _ = build_preprocessors(
            ["admission_type_id"], ["time_in_hospital", "num_medications"],
            ["on_diabetes_med_flag"], df
        )
        assert isinstance(cat_idx, list)

    def test_feature_order_coverage(self):
        df = self._sample_df()
        ord_cols  = ["admission_type_id", "number_diagnoses"]
        cont_cols = ["time_in_hospital", "num_medications"]
        bin_cols  = ["on_diabetes_med_flag"]
        _, _, _, feat_order = build_preprocessors(ord_cols, cont_cols, bin_cols, df)
        for col in ord_cols + cont_cols + bin_cols:
            assert col in feat_order

    def test_scaled_preprocessor_transforms(self):
        df = self._sample_df()
        pre_scaled, _, _, _ = build_preprocessors(
            ["admission_type_id"], ["time_in_hospital", "num_medications"],
            ["on_diabetes_med_flag"], df
        )
        out = pre_scaled.fit_transform(df)
        assert not np.isnan(out).any()

    def test_unscaled_preprocessor_transforms(self):
        df = self._sample_df()
        _, pre_unscaled, _, _ = build_preprocessors(
            ["admission_type_id"], ["time_in_hospital", "num_medications"],
            ["on_diabetes_med_flag"], df
        )
        out = pre_unscaled.fit_transform(df)
        assert not np.isnan(out).any()


# ============================================================
# FEATURE TYPE DETECTION TESTS  (3 tests)
# ============================================================

class TestDetectFeatureTypes:

    def test_binary_detected(self):
        df = pd.DataFrame({
            "flag":            [0, 1, 0, 1, 0],
            "readmitted_30d":  [0, 1, 0, 0, 1]
        })
        _, _, bin_cols = detect_feature_types(df, threshold=10)
        assert "flag" in bin_cols

    def test_target_excluded(self):
        df = pd.DataFrame({
            "num_medications":  [10.0, 20.0, 30.0, 40.0, 50.0],
            "readmitted_30d":   [0, 1, 0, 0, 1]
        })
        ord_cols, cont_cols, bin_cols = detect_feature_types(df, threshold=10)
        assert "readmitted_30d" not in ord_cols + cont_cols + bin_cols

    def test_continuous_detected(self):
        df = pd.DataFrame({
            "num_lab_procedures": [10.5, 20.1, 55.3, 80.0, 110.5, 200.3,
                                    15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
            "readmitted_30d":     [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        _, cont_cols, _ = detect_feature_types(df, threshold=10)
        assert "num_lab_procedures" in cont_cols


# ============================================================
# LEAKAGE CHECK TESTS  (4 tests)
# ============================================================

class TestDetectLeakage:

    def test_catches_identical_feature(self):
        X = pd.DataFrame({"a": [1, 0, 1, 0, 1]})
        y = pd.Series(        [1, 0, 1, 0, 1])
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) > 0
        assert any("a" in w for w in warnings)

    def test_no_false_positives_on_clean_data(self):
        np.random.seed(42)
        X = pd.DataFrame({"num_medications": np.random.uniform(1, 30, 100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) == 0

    def test_catches_high_correlation(self):
        vals = np.arange(50, dtype=float)
        X    = pd.DataFrame({"leaky_feat": vals})
        y    = pd.Series((vals > 25).astype(int))
        warnings = detect_leakage(X, y, threshold_corr=0.85)
        assert len(warnings) > 0

    def test_empty_dataframe_no_crash(self):
        X = pd.DataFrame()
        y = pd.Series([0, 1, 0, 1])
        warnings = detect_leakage(X, y)
        assert isinstance(warnings, list)


# ============================================================
# THRESHOLD TUNING TESTS  (2 tests)
# ============================================================

class TestTuneThreshold:

    def test_returns_float_in_range(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0, 1, 200)
        thr = tune_threshold(y_true, y_prob)
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_perfect_prob_gives_low_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        thr = tune_threshold(y_true, y_prob)
        assert thr <= 0.9, f"Threshold {thr} should be <= 0.9 for perfectly separated data"


# ============================================================
# PSI TESTS  (3 tests)
# ============================================================

class TestPSI:

    def test_identical_distributions(self):
        x = np.random.normal(0, 1, 500)
        assert psi(x, x) < 0.05

    def test_shifted_distribution_higher_psi(self):
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 1000)
        new = rng.normal(3, 1, 1000)
        assert psi(ref, new) > psi(ref, ref)

    def test_uses_reference_edges_not_actual(self):
        rng = np.random.RandomState(0)
        ref = rng.normal(0, 1, 500)
        new = rng.normal(5, 1, 500)
        score = psi(ref, new)
        assert score > 0.20, f"Completely shifted distribution should have PSI >> 0.20, got {score:.4f}"


# ============================================================
# RECALL @ K TESTS  (2 tests)
# ============================================================

class TestRecallAtK:

    def test_top_scores_captured(self):
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        p = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.05, 0.1, 0.15, 0.2, 0.7])
        r = recall_at_k(y, p, k=0.3)
        assert r >= 0.66

    def test_returns_float(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.3, 0.8])
        assert isinstance(recall_at_k(y, p), float)


# ============================================================
# LIFT @ K TESTS  (2 tests)
# ============================================================

class TestLiftAtK:

    def test_lift_above_one_for_good_model(self):
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        p = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.05, 0.1, 0.15, 0.2, 0.7])
        assert lift_at_k(y, p, k=0.3) > 1.0

    def test_returns_float(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.3, 0.8])
        assert isinstance(lift_at_k(y, p), float)


# ============================================================
# KS STATISTIC TESTS  (2 tests)
# ============================================================

class TestKSStatistic:

    def test_returns_float_in_range(self):
        np.random.seed(1)
        y = np.random.randint(0, 2, 100)
        p = np.random.uniform(0, 1, 100)
        ks = ks_statistic(y, p)
        assert isinstance(ks, float)
        assert 0.0 <= ks <= 1.0

    def test_perfect_model_high_ks(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
        assert ks_statistic(y, p) > 0.7


# ============================================================
# COST-SENSITIVE EVALUATION TESTS  (2 tests)
# ============================================================

class TestCostSensitiveEvaluation:

    def test_all_correct_zero_cost(self):
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = cost_sensitive_evaluation(X, y_true, y_pred)
        assert result["total_estimated_cost"] == 0.0

    def test_false_negative_costs_more_than_false_positive(self):
        X = pd.DataFrame({"a": [1, 2]})
        y_true_fn = pd.Series([1, 0])
        y_pred_fn = np.array([0, 0])  # missed the readmission -> FN
        result_fn = cost_sensitive_evaluation(X, y_true_fn, y_pred_fn)

        y_true_fp = pd.Series([0, 0])
        y_pred_fp = np.array([1, 0])  # unnecessary intervention -> FP
        result_fp = cost_sensitive_evaluation(X, y_true_fp, y_pred_fp)

        assert result_fn["estimated_fn_loss"] > result_fp["intervention_cost"]


# ============================================================
# READMISSION ENGINE TESTS  (9 tests)
# ============================================================

class TestReadmissionEngine:

    def test_get_risk_band_low(self):
        assert get_risk_band(0.05) == "LOW"

    def test_get_risk_band_medium(self):
        assert get_risk_band(0.30) == "MEDIUM"

    def test_get_risk_band_high(self):
        assert get_risk_band(0.75) == "HIGH"

    def test_get_risk_band_relative_to_threshold(self):
        """Risk band must never contradict the decision it sits next to —
        the bug this guards against: a fixed 0-0.20 'LOW' band showed up
        next to a CARE_MANAGEMENT decision once calibration pushed the
        real threshold down to ~0.125 (0.1845 was 'LOW' by fixed bands but
        already above the calibrated threshold -> CARE_MANAGEMENT)."""
        threshold = 0.1253
        assert get_risk_band(0.1845, threshold) == "HIGH"
        assert get_risk_band(0.10,   threshold) == "MEDIUM"
        assert get_risk_band(0.05,   threshold) == "LOW"

    def test_score_patient_routine(self):
        row = {"number_inpatient": 0, "time_in_hospital": 3, "num_medications": 5}
        result = score_patient(row, prob=0.05, threshold=0.5)
        assert result["decision"] == "ROUTINE_DISCHARGE"
        assert result["risk_band"] == "LOW"

    def test_score_patient_care_management_rule(self):
        """Frequent prior inpatient visits must trigger CARE_MANAGEMENT hard rule."""
        row = {"number_inpatient": 4, "time_in_hospital": 3, "num_medications": 5}
        result = score_patient(row, prob=0.05, threshold=0.5)
        assert result["decision"] == "CARE_MANAGEMENT"
        assert result["rule_triggered"] == "FREQUENT_PRIOR_INPATIENT"

    def test_score_patient_followup_long_stay_rule(self):
        row = {"number_inpatient": 0, "time_in_hospital": 12, "num_medications": 5}
        result = score_patient(row, prob=0.05, threshold=0.5)
        assert result["decision"] == "FOLLOW_UP_REQUIRED"
        assert result["rule_triggered"] == "LONG_INPATIENT_STAY"

    def test_score_patient_followup_polypharmacy_rule(self):
        row = {"number_inpatient": 0, "time_in_hospital": 3, "num_medications": 20}
        result = score_patient(row, prob=0.05, threshold=0.5)
        assert result["decision"] == "FOLLOW_UP_REQUIRED"
        assert result["rule_triggered"] == "POLYPHARMACY"

    def test_score_patient_care_management_model(self):
        """High probability (no rule triggered) must lead to CARE_MANAGEMENT via model."""
        row = {"number_inpatient": 0, "time_in_hospital": 3, "num_medications": 5}
        result = score_patient(row, prob=0.9, threshold=0.5)
        assert result["decision"] == "CARE_MANAGEMENT"
        assert result["rule_triggered"] is None

    def test_output_keys_complete(self):
        row = {"number_inpatient": 0, "time_in_hospital": 3, "num_medications": 5}
        result = score_patient(row, prob=0.2, threshold=0.5)
        for key in ("readmission_probability", "risk_band", "decision", "rule_triggered"):
            assert key in result

    def test_readmission_engine_batch(self):
        """readmission_engine must return one decision per row."""
        df = pd.DataFrame({
            "number_inpatient":  [0, 4, 0],
            "time_in_hospital":  [3, 3, 12],
            "num_medications":   [5, 5, 5],
        })
        probs = [0.1, 0.1, 0.1]
        decisions = readmission_engine(df, probs, threshold=0.5)
        assert len(decisions) == 3
        assert decisions[1] == "CARE_MANAGEMENT_RULE"


# ============================================================
# FEATURE ENGINEERING TESTS  (2 tests)
# ============================================================

class TestFeatureEngineering:

    def test_polypharmacy_flag(self):
        df = pd.DataFrame({
            "num_medications":  [20, 3],
            "time_in_hospital": [5, 5],
            "num_procedures":   [1, 1],
            "num_lab_procedures":[40, 40],
            "number_outpatient":[0, 0],
            "number_emergency": [0, 0],
            "number_inpatient": [0, 0],
            "number_diagnoses": [5, 5],
            "discharge_disposition_id": [1, 1],
            "admission_type_id": [1, 1],
        })
        out = add_engineered_features(df)
        assert out["polypharmacy_flag"].tolist() == [1, 0]

    def test_frequent_flyer_flag(self):
        df = pd.DataFrame({
            "num_medications":  [10, 10],
            "time_in_hospital": [5, 5],
            "num_procedures":   [1, 1],
            "num_lab_procedures":[40, 40],
            "number_outpatient":[0, 0],
            "number_emergency": [0, 0],
            "number_inpatient": [3, 0],
            "number_diagnoses": [5, 5],
            "discharge_disposition_id": [1, 1],
            "admission_type_id": [1, 1],
        })
        out = add_engineered_features(df)
        assert out["frequent_flyer_flag"].tolist() == [1, 0]


# ============================================================
# CONFIG TESTS  (4 tests)
# ============================================================

class TestConfig:

    def test_psi_thresholds_ordered(self):
        assert PSI_MODERATE < PSI_HIGH

    def test_challenger_gates_reasonable(self):
        assert 0.0 < MIN_F1_IMPROVEMENT    < 0.1
        assert 0.5 < MIN_ROCAUC_THRESHOLD  < 1.0
        assert 0.0 < MAX_GENERALIZATION_GAP < 1.0

    def test_risk_band_boundaries_valid(self):
        for band, (low, high) in RISK_BANDS.items():
            assert 0.0 <= low  <= 1.0, f"{band} low boundary invalid"
            assert 0.0 <= high <= 1.1, f"{band} high boundary invalid"

    def test_business_rule_thresholds_positive(self):
        assert MAX_PRIOR_INPATIENT_RULE  > 0
        assert MAX_TIME_IN_HOSPITAL_RULE > 0
        assert POLYPHARMACY_RULE         > 0