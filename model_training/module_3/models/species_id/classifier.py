import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix


def make_stage1_pipeline(random_state=42) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    svm = SVC(
        C=10, gamma="scale", class_weight="balanced", probability=True, random_state=random_state
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=150, class_weight="balanced", random_state=random_state
    )
    vc = VotingClassifier(estimators=[("rf", rf), ("svm", svm), ("hgb", hgb)], voting="soft")
    return Pipeline([("scaler", StandardScaler()), ("ensemble", vc)])


class HierarchicalSpeciesClassifier(BaseEstimator, ClassifierMixin):
    """
    Stage 1: flat soft-voting ensemble across all species.
    Stage 2: when Stage-1's top-2 predictions are close AND both belong to
             the same auto-discovered confusable cluster, a specialist model
             trained only on that cluster's species re-decides.

    Clusters are discovered via an inner StratifiedGroupKFold on the
    training data only (never touches the outer/held-out fold).
    """

    def __init__(self, confusion_threshold=10, margin_threshold=0.12,
                 inner_splits=4, random_state=42):
        self.confusion_threshold = confusion_threshold
        self.margin_threshold = margin_threshold
        self.inner_splits = inner_splits
        self.random_state = random_state

    def _discover_clusters(self, X, y, groups):
        cv = StratifiedGroupKFold(n_splits=self.inner_splits, shuffle=True,
                                   random_state=self.random_state)
        oof_pred = np.empty_like(y)
        for tr, te in cv.split(X, y, groups):
            m = make_stage1_pipeline(self.random_state)
            m.fit(X[tr], y[tr])
            oof_pred[te] = m.predict(X[te])

        labels = sorted(set(y))
        cm = confusion_matrix(y, oof_pred, labels=labels)
        adj = {l: set() for l in labels}
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                if i != j and (cm[i, j] + cm[j, i]) >= self.confusion_threshold:
                    adj[li].add(lj)
                    adj[lj].add(li)

        seen, clusters = set(), []
        for l in labels:
            if l in seen:
                continue
            stack, comp = [l], set()
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                seen.add(u)
                stack.extend(adj[u] - comp)
            if len(comp) > 1:
                clusters.append(sorted(comp))
        return clusters

    def fit(self, X, y, groups=None):
        y = np.asarray(y)
        if groups is None:
            groups = np.arange(len(y))

        self.classes_ = np.array(sorted(set(y)))
        self.clusters_ = self._discover_clusters(X, y, groups)

        self.stage1_ = make_stage1_pipeline(self.random_state)
        self.stage1_.fit(X, y)

        self.stage2_ = {}
        for cl in self.clusters_:
            mask = np.isin(y, cl)
            m = make_stage1_pipeline(self.random_state)
            m.fit(X[mask], y[mask])
            self.stage2_[tuple(cl)] = m
        return self

    def predict(self, X):
        proba = self.stage1_.predict_proba(X)
        s1_classes = self.stage1_.classes_
        order = np.argsort(-proba, axis=1)
        top1_idx, top2_idx = order[:, 0], order[:, 1]
        top1 = s1_classes[top1_idx]
        top2 = s1_classes[top2_idx]
        p1 = proba[np.arange(len(X)), top1_idx]
        p2 = proba[np.arange(len(X)), top2_idx]
        margin = p1 - p2

        final = top1.copy()
        for cl in self.clusters_:
            clset = set(cl)
            in_cluster = np.array([
                (top1[i] in clset and top2[i] in clset and margin[i] < self.margin_threshold)
                for i in range(len(X))
            ])
            if not in_cluster.any():
                continue
            specialist = self.stage2_[tuple(cl)]
            final[in_cluster] = specialist.predict(X[in_cluster])
        return final