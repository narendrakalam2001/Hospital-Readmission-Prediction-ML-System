# ============================================================
# MODEL TUNING — Hospital Readmission ML System
# ============================================================

import numpy as np
import logging

from typing import Dict, List, Tuple

from sklearn.base              import BaseEstimator
from sklearn.compose           import ColumnTransformer
from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection   import RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier, ExtraTreesClassifier
from xgboost                import XGBClassifier

from imblearn.pipeline      import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from src.config        import RANDOM_STATE, N_JOBS, CV_FOLDS, RANDOM_SEARCH_ITERS, SELECT_K
from src.preprocessing import safe_k

logger = logging.getLogger(__name__)


# ============================================================
# HELPER — compute safe n_iter for RandomizedSearchCV
# ============================================================

def _compute_n_iter(param_dist: dict, budget: int) -> int:
    if not param_dist:
        return 1
    prod = 1
    for v in param_dist.values():
        try:
            prod *= len(v)
        except TypeError:
            prod *= budget
    return min(budget, max(1, prod))


# ============================================================
# MODEL GRIDS
# ============================================================
# NOTE: this project runs on a single-CPU CI/serverless-style box, so the
# search space is intentionally compact compared to a full production
# sweep (which would include KNN/SVM/CatBoost/MLP with wider grids on
# a multi-core cluster). Models chosen below cover the standard
# linear-vs-tree-vs-boosting comparison expected in a hospital risk
# scoring system while keeping wall-clock time reasonable on ~100K rows.

# Linear model -> needs scaled input
scaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "LogisticRegression": (
        LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        {
            "classifier__C":            [0.01, 0.1, 1, 10],
            "classifier__penalty":      ["l2"],
            "classifier__solver":       ["lbfgs"],
            "classifier__class_weight": [None, "balanced"],
        }
    ),
}

# Tree-based models -> work on raw / clipped input
# class_weight="balanced" / scale_pos_weight handle the ~11% positive
# class imbalance directly inside the model (SMOTENC is skipped for
# the full-size search here purely for tuning speed on this hardware —
# see README "Design Decisions" section).
unscaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__max_depth":        [5, 10, 20],
            "classifier__min_samples_leaf": [5, 20, 50],
        }
    ),

    "RandomForest": (
        RandomForestClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__n_estimators":     [150, 300],
            "classifier__max_depth":        [10, 20, None],
            "classifier__min_samples_leaf": [2, 5],
        }
    ),

    "ExtraTrees": (
        ExtraTreesClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__n_estimators": [150, 300],
            "classifier__max_depth":    [10, 20, None],
        }
    ),

    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=N_JOBS),
        {
            "classifier__n_estimators":     [150, 300],
            "classifier__learning_rate":    [0.05, 0.1],
            "classifier__max_depth":        [3, 5, 7],
            "classifier__scale_pos_weight": [1, 4, 8],
        }
    ),
}

if LGBMClassifier is not None:
    unscaled_models["LightGBM"] = (
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, is_unbalance=True, n_jobs=N_JOBS),
        {
            "classifier__n_estimators":  [150, 300],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth":     [-1, 8, 12],
            "classifier__num_leaves":    [31, 63],
        }
    )


# ============================================================
# TUNE MODELS
# ============================================================

def tune_models(
    models:         Dict[str, Tuple[BaseEstimator, dict]],
    preprocessor:   ColumnTransformer,
    cat_indices:    List[int],
    X_train:        "pd.DataFrame",
    y_train:        "pd.Series",
    use_smote:      bool = False,
    selector_k:     int  = SELECT_K
) -> Dict[str, ImbPipeline]:
    """
    For each model:
      preprocessor -> [SMOTENC] -> SelectKBest -> classifier
    Tuned with RandomizedSearchCV (scoring = F1).
    Returns dict of {model_name: best_pipeline}.
    """

    final_pipelines: Dict[str, ImbPipeline] = {}

    k_safe = safe_k(selector_k, preprocessor, X_train)
    logger.info("Selector k set to %d (requested %d)", k_safe, selector_k)

    for name, (clf, param_dist) in models.items():

        logger.info("Tuning: %s", name)

        steps = [("preprocessor", preprocessor)]

        if use_smote and len(cat_indices) > 0:
            steps.append((
                "smote",
                SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)
            ))

        # mutual_info_classif adds internal random noise to break ties on
        # continuous features — MUST be seeded, otherwise SelectKBest picks a
        # (slightly) different feature subset on every run, and F1/ROC-AUC
        # become non-reproducible even with a fixed RANDOM_STATE everywhere else.
        mi_scorer = partial(mutual_info_classif, random_state=RANDOM_STATE)
        steps.append(("selector", SelectKBest(mi_scorer, k=k_safe)))
        steps.append(("classifier", clf))

        pipe = ImbPipeline(steps)

        n_iter = _compute_n_iter(param_dist, RANDOM_SEARCH_ITERS)

        search = RandomizedSearchCV(
            pipe,
            param_distributions = param_dist,
            n_iter              = n_iter,
            scoring             = "f1",
            cv                  = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs              = N_JOBS,
            random_state        = RANDOM_STATE,
            verbose             = 0
        )

        search.fit(X_train, y_train)

        logger.info("%s best params: %s", name, search.best_params_)

        final_pipelines[name] = search.best_estimator_

    return final_pipelines