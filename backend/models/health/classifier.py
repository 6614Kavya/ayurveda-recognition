
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score

SEVERITY_LEVELS = ["moderate", "high"]  # was ["low", "mid", "high"] -- see module docstring


def fuse_top_bottom(top_row: dict, bottom_row: dict, feature_cols: List[str]) -> dict:
    """
    Build one leaf-level feature row from two per-view rows: for every
    feature, keep top_<f>, bottom_<f>, AND worst_<f> = max(top_<f>, bottom_<f>).
    The worst_* columns directly encode worst-side-wins at the FEATURE
    level, not just the final label.
    """
    fused = {}
    for f in feature_cols:
        t = top_row.get(f, np.nan)
        b = bottom_row.get(f, np.nan)
        fused[f"top_{f}"] = t
        fused[f"bottom_{f}"] = b
        try:
            fused[f"worst_{f}"] = max(t, b)
        except TypeError:
            fused[f"worst_{f}"] = np.nan
    return fused


@dataclass
class TwoStageHealthClassifier:
    stage1: Optional[Pipeline] = None
    stage2: Optional[Pipeline] = None
    stage1_threshold: float = 0.5   # tuned in fit() if tune_threshold=True

    def _build_stage1(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
        ])

    def _build_stage2(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced",
                        probability=True, random_state=42)),
        ])

    def fit(self, X: np.ndarray, y_binary: np.ndarray, y_severity: np.ndarray,
            tune_threshold: bool = True, threshold_cv_splits: int = 3):
        """
        X            : leaf-level fused feature matrix (top_*, bottom_*, worst_*, [species one-hot]).
        y_binary     : "healthy" / "unhealthy" per leaf.
        y_severity   : "moderate"/"high" per leaf -- only rows where
                       y_binary == "unhealthy" are used to fit Stage 2.
        tune_threshold : if True, picks the F1-macro-optimal Stage-1
            decision threshold from cross-validated out-of-fold
            probabilities computed ONLY on this fit's training data (X,
            y_binary) -- never on the caller's held-out CV fold or test
            set, so this is a legitimate calibration step. Set False to
            keep the raw 0.5 default (e.g. for quick debugging).
        """
        self.stage1 = self._build_stage1()

        if tune_threshold:
            classes_preview = np.unique(y_binary)
            n_splits = min(threshold_cv_splits, int(np.min(np.unique(y_binary, return_counts=True)[1])))
            n_splits = max(n_splits, 2)
            oof_proba = cross_val_predict(
                self.stage1, X, y_binary,
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                method="predict_proba",
            )
            # locate the "unhealthy" column regardless of class ordering
            tmp_fit_classes = np.unique(y_binary)  # cross_val_predict uses sorted class order internally
            unhealthy_col = list(sorted(classes_preview)).index("unhealthy")
            proba_unhealthy = oof_proba[:, unhealthy_col]
            self.stage1_threshold = self._best_threshold(y_binary, proba_unhealthy)
            print(f"[stage1] tuned decision threshold = {self.stage1_threshold:.3f} "
                  f"(0.5 = untuned default)")

        self.stage1.fit(X, y_binary)

        unhealthy_mask = np.asarray(y_binary) == "unhealthy"
        self.stage2 = self._build_stage2()
        self.stage2.fit(X[unhealthy_mask], np.asarray(y_severity)[unhealthy_mask])
        return self

    @staticmethod
    def _best_threshold(y_true: np.ndarray, proba_unhealthy: np.ndarray,
                         grid=np.arange(0.30, 0.71, 0.02)) -> float:
        """F1-macro-optimal threshold on 'unhealthy' probability, sweeping
        a grid rather than just accepting the default 0.5 -- addresses the
        Stage-1 healthy-recall bias (0.53-0.56) that class_weight=
        'balanced' alone didn't fully fix."""
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            preds = np.where(proba_unhealthy >= t, "unhealthy", "healthy")
            f1 = f1_score(y_true, preds, average="macro")
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        return best_t

    def predict_with_confidence(self, X: np.ndarray) -> List[dict]:
        stage1_proba = self.stage1.predict_proba(X)
        classes1 = list(self.stage1.classes_)
        unhealthy_col = classes1.index("unhealthy")
        # tuned threshold on P(unhealthy) instead of raw .predict() @ 0.5
        stage1_pred = np.where(
            stage1_proba[:, unhealthy_col] >= self.stage1_threshold, "unhealthy", "healthy"
        )

        unhealthy_idx = np.where(stage1_pred == "unhealthy")[0]
        stage2_pred = np.array([None] * len(X), dtype=object)
        stage2_proba = np.zeros((len(X), len(SEVERITY_LEVELS)))

        if len(unhealthy_idx) > 0:
            s2p = self.stage2.predict(X[unhealthy_idx])
            s2pr = self.stage2.predict_proba(X[unhealthy_idx])
            classes2 = list(self.stage2.classes_)
            stage2_pred[unhealthy_idx] = s2p
            for j, idx in enumerate(unhealthy_idx):
                for k, cls in enumerate(classes2):
                    stage2_proba[idx, SEVERITY_LEVELS.index(cls)] = s2pr[j, k]

        results = []
        for i in range(len(X)):
            row = {
                "stage1_label": stage1_pred[i],
                "stage1_confidence": float(stage1_proba[i, classes1.index(stage1_pred[i])]),
            }
            if stage1_pred[i] == "healthy":
                row["final_level"] = "healthy"
                row["final_confidence"] = row["stage1_confidence"]
            else:
                row["final_level"] = stage2_pred[i]
                lvl_idx = SEVERITY_LEVELS.index(stage2_pred[i])
                row["final_confidence"] = float(stage2_proba[i, lvl_idx])
            results.append(row)
        return results