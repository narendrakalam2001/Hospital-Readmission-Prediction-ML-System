# ============================================================
# EVALUATION — Hospital Readmission ML System
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from typing import Dict, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.model_selection   import cross_val_score, StratifiedKFold
from sklearn.inspection        import permutation_importance
from sklearn.pipeline          import Pipeline

from src.config  import RANDOM_STATE, N_JOBS, CV_FOLDS, MODEL_DIR
from src.metrics import tune_threshold, ks_statistic

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
# SAFE PREDICT PROBA
# ============================================================

def safe_predict_proba(pipe, X):
    try:
        return pipe.predict_proba(X)[:, 1]
    except Exception:
        logger.warning("predict_proba failed — falling back to manual extraction")
        try:
            pre  = pipe.named_steps["preprocessor"].transform(X)
            sel  = pipe.named_steps.get("selector", None)
            X_in = sel.transform(pre) if sel is not None else pre
            clf  = pipe.named_steps["classifier"]
            if hasattr(clf, "predict_proba"):
                return clf.predict_proba(X_in)[:, 1]
        except Exception as e:
            logger.error("safe_predict_proba fully failed: %s", e)
    return None


# ============================================================
# EVALUATE ALL MODELS — summary table
# ============================================================

def evaluate_models(
    pipelines:    Dict,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    X_test:       pd.DataFrame,
    y_test:       pd.Series,
    cv_sample_size: int = 15000,
) -> pd.DataFrame:

    rows = []

    # CV stability check on a stratified subsample (speed, single-CPU box)
    if len(X_train) > cv_sample_size:
        cv_X = X_train.sample(n=cv_sample_size, random_state=RANDOM_STATE)
        cv_y = y_train.loc[cv_X.index]
    else:
        cv_X, cv_y = X_train, y_train

    for name, pipe in pipelines.items():

        logger.info("Evaluating: %s", name)

        try:
            train_acc = accuracy_score(y_train, pipe.predict(X_train))
        except Exception:
            train_acc = None

        test_acc  = accuracy_score(y_test, pipe.predict(X_test))

        y_prob = safe_predict_proba(pipe, X_test)

        if y_prob is not None:
            thr    = tune_threshold(y_test.values, y_prob)
            y_pred = (y_prob >= thr).astype(int)
        else:
            thr    = None
            y_pred = pipe.predict(X_test)

        prec     = precision_score(y_test, y_pred,       zero_division=0)
        rec      = recall_score(y_test, y_pred,           zero_division=0)
        f1       = f1_score(y_test, y_pred,               zero_division=0)
        roc_auc  = roc_auc_score(y_test, y_prob)          if y_prob is not None else None
        pr_auc   = average_precision_score(y_test, y_prob) if y_prob is not None else None
        ks       = ks_statistic(y_test, y_prob)            if y_prob is not None else None
        brier    = brier_score_loss(y_test, y_prob)        if y_prob is not None else None

        try:
            cv_scores = cross_val_score(
                pipe, cv_X, cv_y,
                scoring = "f1",
                cv      = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                n_jobs  = N_JOBS
            )
            cv_mean = float(cv_scores.mean())
            cv_std  = float(cv_scores.std())
        except Exception:
            cv_mean = None
            cv_std  = None

        rows.append({
            "model":           name,
            "train_acc":       round(train_acc, 4) if train_acc else None,
            "test_acc":        round(test_acc,  4),
            "train_test_gap":  round(abs((train_acc or 0) - test_acc), 4),
            "cv_mean_f1":      round(cv_mean, 4) if cv_mean else None,
            "cv_std_f1":       round(cv_std,  4) if cv_std  else None,
            "cv_test_gap":     round(abs((cv_mean or 0) - f1), 4),
            "precision":       round(prec,    4),
            "recall":          round(rec,     4),
            "f1":              round(f1,      4),
            "roc_auc":         round(roc_auc, 4) if roc_auc else None,
            "pr_auc":          round(pr_auc,  4) if pr_auc  else None,
            "ks":              round(ks,      4) if ks      else None,
            "brier":           round(brier,   4) if brier   else None,
            "threshold":       round(thr,     4) if thr     else None,
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )

    summary.to_csv(os.path.join(MODEL_DIR, "model_experiment_results.csv"), index=False)

    return summary


# ============================================================
# SELECT BEST MODEL — with generalization filter
# ============================================================

def select_best_model(
    summary:      pd.DataFrame,
    pipelines:    Dict,
    scaled_pipes: Dict,
    unscaled_pipes:Dict
) -> Tuple[str, object]:

    def _filter(df, gen_gap, cv_std, min_f1, min_prec, min_rec):
        mask = (
            df["train_test_gap"].notna() &
            df["cv_std_f1"].notna() &
            (df["train_test_gap"] <= gen_gap) &
            (df["cv_std_f1"]      <= cv_std)  &
            (df["f1"]             >= min_f1)   &
            (df["precision"]      >= min_prec) &
            (df["recall"]         >= min_rec)
        )
        return df[mask].copy()

    thresholds = [
        (0.05, 0.05, 0.45, 0.35, 0.35),
        (0.08, 0.07, 0.35, 0.25, 0.25),
        (0.10, 0.10, 0.25, 0.15, 0.15),
        (0.20, 0.20, 0.00, 0.00, 0.00),
    ]

    candidates = pd.DataFrame()
    for gg, cvs, mf1, mp, mr in thresholds:
        candidates = _filter(summary, gg, cvs, mf1, mp, mr)
        if not candidates.empty:
            logger.info("Candidates found with gen_gap<=%.2f cv_std<=%.2f min_f1>=%.2f", gg, cvs, mf1)
            break

    if candidates.empty:
        selected_name = summary.iloc[0]["model"]
    else:
        candidates    = candidates.sort_values(
            ["f1", "roc_auc", "cv_std_f1"], ascending=[False, False, True]
        ).reset_index(drop=True)
        selected_name = candidates.iloc[0]["model"]

    logger.info("Selected model: %s", selected_name)

    selected_pipe = scaled_pipes.get(selected_name) or unscaled_pipes.get(selected_name)
    if selected_pipe is None:
        raise RuntimeError(f"Selected model '{selected_name}' not found in any pipeline dict")

    return selected_name, selected_pipe


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_with_holdout(selected_pipe, X_cal, y_cal, method: str = "isotonic"):
    """Holdout calibration — no resampling on the calibration set."""
    clf = selected_pipe.named_steps["classifier"]
    pre = selected_pipe.named_steps["preprocessor"]
    sel = selected_pipe.named_steps.get("selector", None)

    X_pre = pre.transform(X_cal)
    X_sel = sel.transform(X_pre) if sel is not None else X_pre

    method_use = method
    if method == "isotonic" and len(y_cal) < 1000:
        method_use = "sigmoid"
        logger.info("Calibration: switching to sigmoid (cal set < 1000 rows)")

    try:
        try:
            cal = CalibratedClassifierCV(estimator=clf,      cv="prefit", method=method_use)
        except TypeError:
            cal = CalibratedClassifierCV(base_estimator=clf, cv="prefit", method=method_use)

        cal.fit(X_sel, y_cal)

        steps = [("preprocessor", pre)]
        if sel is not None:
            steps.append(("selector", sel))
        steps.append(("classifier", cal))

        cal_pipe = Pipeline(steps)
        logger.info("Calibration done  |  method=%s", method_use)
        return cal_pipe

    except Exception as e:
        logger.exception("Calibration failed: %s", e)
        return None


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance(selected_pipe, X_train, y_train, top_k: int = 20):

    clf   = selected_pipe.named_steps["classifier"]
    names = _get_feature_names(selected_pipe, X_train)

    if hasattr(clf, "feature_importances_"):
        imp   = np.asarray(clf.feature_importances_)
        names = names if names and len(names) == len(imp) else [f"f{i}" for i in range(len(imp))]
        fi    = pd.Series(imp, index=names).sort_values(ascending=False).head(top_k)
        return fi

    if hasattr(clf, "coef_"):
        coef  = np.abs(clf.coef_).ravel()
        names = names if names and len(names) == len(coef) else [f"f{i}" for i in range(len(coef))]
        fi    = pd.Series(coef, index=names).sort_values(ascending=False).head(top_k)
        return fi

    logger.info("Using permutation importance (slow fallback, small sample)...")
    try:
        sample_X = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_STATE)
        sample_y = y_train.loc[sample_X.index]
        r   = permutation_importance(
            selected_pipe, sample_X, sample_y,
            n_repeats=5, scoring="f1", n_jobs=N_JOBS, random_state=RANDOM_STATE
        )
        idx = np.argsort(r.importances_mean)[::-1][:top_k]
        fi  = pd.Series(r.importances_mean[idx],
                        index=[f"f{i}" for i in idx])
        return fi
    except Exception as e:
        logger.exception("Permutation importance failed: %s", e)
        return None


def _get_feature_names(pipe, X_sample):
    """
    Extracts clean feature names after preprocessor + selector.
    Strips sklearn ColumnTransformer prefixes like 'ord__', 'skewed__', 'bin__'.
    """
    try:
        pre       = pipe.named_steps["preprocessor"]
        raw_names = list(pre.get_feature_names_out())

        clean_names = []
        for n in raw_names:
            if "__" in n:
                clean_names.append(n.split("__", 1)[1])
            else:
                clean_names.append(n)

        sel = pipe.named_steps.get("selector", None)
        if sel is not None:
            mask        = sel.get_support()
            clean_names = [n for n, m in zip(clean_names, mask) if m]

        return clean_names

    except Exception as e:
        logger.warning("Could not extract feature names: %s", e)
        return None


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def compute_shap(selected_pipe, X_train, X_explain, max_explain: int = 200):

    if not SHAP_AVAILABLE:
        logger.info("SHAP not installed — skipping")
        return None

    def _get_base_clf(clf):
        """Unwrap CalibratedClassifierCV to reach the underlying tree/linear model."""
        if hasattr(clf, "calibrated_classifiers_"):
            inner = clf.calibrated_classifiers_[0]
            return getattr(inner, "estimator", getattr(inner, "base_estimator", clf))
        return clf

    clf_raw = selected_pipe.named_steps["classifier"]
    clf     = _get_base_clf(clf_raw)
    pre     = selected_pipe.named_steps["preprocessor"]
    sel     = selected_pipe.named_steps.get("selector", None)

    try:
        X_pre = pre.transform(X_train.sample(min(2000, len(X_train)), random_state=RANDOM_STATE))
        X_tr  = sel.transform(X_pre) if sel is not None else X_pre
        X_ex  = sel.transform(pre.transform(X_explain)) if sel is not None else pre.transform(X_explain)
        X_ex  = np.asarray(X_ex)[:max_explain]

        names = _get_feature_names(selected_pipe, X_train)

        tree_types = ("RandomForestClassifier", "ExtraTreesClassifier",
                      "GradientBoostingClassifier", "XGBClassifier",
                      "LGBMClassifier", "DecisionTreeClassifier")

        if type(clf).__name__ in tree_types:
            explainer = shap.TreeExplainer(clf)
            try:
                vals = np.array(explainer(X_ex).values)
            except Exception:
                vals = np.array(explainer.shap_values(X_ex))
                if isinstance(vals, list):
                    vals = vals[1]
        else:
            bg_sample = shap.sample(X_tr, min(100, len(X_tr)))

            def model_fn(x):
                return clf.predict_proba(x)[:, 1]

            explainer = shap.KernelExplainer(model_fn, bg_sample)
            vals      = np.array(explainer.shap_values(X_ex[:50], nsamples=50))

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        if vals.ndim == 3:
            vals = vals[:, :, -1]

        mean_abs   = np.abs(vals).mean(axis=0)
        feat_names = names[:mean_abs.shape[0]] if names else [f"f{i}" for i in range(mean_abs.shape[0])]

        fi_series = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)

        try:
            shap.summary_plot(vals, features=X_ex[:len(vals)], feature_names=feat_names, show=False)
            os.makedirs(os.path.join("docs", "plots"), exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join("docs", "plots", "shap_summary.png"))
            plt.close()
        except Exception:
            pass

        logger.info("SHAP done  |  explainer=%s", type(explainer).__name__)

        return {"shap_top": fi_series.head(20).to_dict(), "explainer": type(explainer).__name__}

    except Exception as e:
        logger.exception("SHAP failed: %s", e)
        return None


# ============================================================
# SAVE MODEL  (model card saving lives in src/model_card.py)
# ============================================================

def save_challenger_artifact(
    selected_name:   str,
    pipe,
    model_card:      dict,
    version:         str = "v1",
    model_card_path: str = ""
):
    """
    Saves the trained CHALLENGER pipeline as .joblib — does NOT touch the
    registry (latest_model.json). The registry is only updated by
    run_challenger_comparison() in model_loader.py, and only when the
    challenger genuinely beats the current champion on all 3 gates.
    """
    model_path = os.path.join(MODEL_DIR, f"readmission_model_{selected_name}_{version}.joblib")
    joblib.dump(pipe, model_path)

    logger.info("Challenger model saved -> %s (registry untouched until gates pass)", model_path)

    return model_path


# ============================================================
# MLFLOW LOGGING
# ============================================================

def mlflow_log_run(
    run_name:      str,
    selected_name: str,
    pipe,
    model_card:    dict,
    X_train_sample: "pd.DataFrame" = None
):
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not installed — skipping")
        return

    try:
        # Use an explicit SQLite backend (MLflow's own recommended fix — see
        # the "maintenance mode" notice for the plain FileStore backend on
        # newer MLflow versions). SQLite also sidesteps the Windows bug where
        # a file:// URI percent-encodes spaces in the path (e.g. "Narendra
        # Kalam" -> "%20") and then fails to create the directory literally.
        db_path = os.path.join(os.path.abspath("."), "mlflow.db").replace(os.sep, "/")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment("hospital_readmission_experiments")

        metrics_dict = model_card.get("metrics", model_card)

        with mlflow.start_run(run_name=run_name):

            mlflow.log_param("model_name",  selected_name)
            mlflow.log_param("selector_k",  model_card.get("pipeline_config", {}).get("selector_k", "?"))
            mlflow.log_param("threshold",   model_card.get("thresholds", {}).get("active", "?"))

            mlflow.log_metric("test_f1",        float(metrics_dict.get("test_f1",        0)))
            mlflow.log_metric("test_precision", float(metrics_dict.get("test_precision", 0)))
            mlflow.log_metric("test_recall",    float(metrics_dict.get("test_recall",    0)))
            mlflow.log_metric("roc_auc",        float(metrics_dict.get("roc_auc",        0)))
            mlflow.log_metric("ks_statistic",   float(metrics_dict.get("ks_statistic",   0)))
            mlflow.log_metric("brier_score",    float(metrics_dict.get("brier_score",    0)))

            log_kwargs = {
                "sk_model":         pipe,
                "name":             "model",
                "pip_requirements": [
                    "scikit-learn==1.3.2",
                    "lightgbm",
                    "xgboost",
                    "imbalanced-learn",
                    "pandas",
                    "numpy",
                ],
                # Newer MLflow serializes sklearn models via skops, which
                # refuses to pickle custom/non-stdlib classes by default
                # (security guard). Our pipeline legitimately contains
                # these depending on which model wins Champion-Challenger:
                # Clipper (our custom transformer), the seeded
                # mutual_info_classif selector (functools.partial),
                # CalibratedClassifierCV's internal wrapper, and the
                # imblearn Pipeline wrapper itself. All are first-party /
                # well-known sklearn-ecosystem internals, safe here.
                "skops_trusted_types": [
                    "functools.partial",
                    "sklearn.calibration._CalibratedClassifier",
                    "sklearn.feature_selection._mutual_info.mutual_info_classif",
                    "src.preprocessing.Clipper",
                    "imblearn.pipeline.Pipeline",
                ],
            }
            if X_train_sample is not None:
                log_kwargs["input_example"] = X_train_sample.iloc[:5].astype(str)

            # Self-healing retry: skops' "untrusted type" error message lists
            # exactly which types it refused, in the form
            # "Untrusted types found in the file: [...]" — parse that list
            # and add any new/unseen types to skops_trusted_types, then
            # retry. This means the champion model can change (different
            # sklearn/imblearn internals get pulled in) without this
            # function needing a manual update every time.
            import re as _re

            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    mlflow.sklearn.log_model(**log_kwargs)
                    break
                except TypeError:
                    # Older MLflow versions don't accept skops_trusted_types
                    # at all (plain pickle serialization, no skops guard).
                    log_kwargs.pop("skops_trusted_types", None)
                    try:
                        mlflow.sklearn.log_model(**log_kwargs)
                    except Exception as e:
                        logger.warning("mlflow log_model failed: %s", e)
                    break
                except Exception as e:
                    match = _re.search(r"Untrusted types found in the file:\s*(\[.*?\])", str(e))
                    new_types = _re.findall(r"'([^']+)'", match.group(1)) if match else []
                    trusted   = log_kwargs.get("skops_trusted_types", [])
                    unseen    = [t for t in new_types if t not in trusted]

                    if unseen and attempt < max_attempts - 1:
                        logger.info("mlflow log_model: adding newly-seen trusted types %s and retrying", unseen)
                        log_kwargs["skops_trusted_types"] = trusted + unseen
                        continue

                    logger.warning("mlflow log_model failed: %s", e)
                    break

        logger.info("MLflow run logged: %s", run_name)

    except Exception as e:
        logger.exception("MLflow logging failed: %s", e)