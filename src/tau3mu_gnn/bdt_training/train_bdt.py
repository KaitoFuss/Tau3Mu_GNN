"""ISO BDT training (port of the ISO mode of Train_XGB_W.py).

Events and derived features come from the stage 2 code (BDT_ScoreProducer._load / _add_features,
bdt_inputs.bdt_branches), so training and inference see the same columns in the same order:
one row per event (candidate 0), rows with NaN dropped. Signal: all events. Background: events
with all candidates in the sideband. The classifier pickle {fold: XGBClassifier} is the format
stage 2 reads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import awkward as ak
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from tau3mu_gnn.bdt_training.BDT_ScoreProducer import PREFIX, _add_features, _load  # noqa: E402
from tau3mu_gnn.bdt_training.bdt_inputs import bdt_branches  # noqa: E402


def sideband_filter(tau_mass: float, half_width: float, prefix: str = PREFIX) -> Callable[[ak.Array], ak.Array]:
    """Event mask: all candidates have abs(fitted_mass - tau_mass) > half_width."""
    def keep(events):
        return ak.all(abs(events[f"{prefix}TauTo3Mu_fitted_mass"] - tau_mass) > half_width, axis=1)
    return keep


def load_training_data(signal_files, background_files, tree_name: str, channel: str, tau_mass: float,
                       sideband_half_width: float, max_events_per_class: int,
                       sample_seed: int) -> tuple[pd.DataFrame, list[str]]:
    """Signal + sideband background rows (columns target, weight, id, inputs, derived features).
    Returns (data, bdt_features). Raise ValueError if a class has no events."""
    features, branches = bdt_branches(channel, PREFIX)
    sig = _load(list(signal_files), tree_name, branches, "signal")
    bkg = _load(list(background_files), tree_name, branches, "bkg",
                event_filter=sideband_filter(tau_mass, sideband_half_width))
    for name, df in (("signal", sig), ("background", bkg)):
        if len(df) == 0:
            raise ValueError(f"BDT training: no {name} events (background: sideband "
                             f"abs(m - {tau_mass}) > {sideband_half_width} only)")
    sig = sig.sample(n=min(max_events_per_class, len(sig)), random_state=sample_seed).reset_index(drop=True)
    bkg = bkg.sample(n=min(max_events_per_class, len(bkg)), random_state=sample_seed).reset_index(drop=True)
    bdt_features = _add_features(sig, bkg, features, channel, PREFIX)
    for col in bdt_features:  # as stage 2 before scoring
        if sig[col].dtype == object:
            sig[col] = sig[col].astype(int)
            bkg[col] = bkg[col].astype(int)
    sig["target"], bkg["target"] = 1, 0
    sig["weight"], bkg["weight"] = 1., 1.
    data = pd.concat([sig, bkg], ignore_index=True)
    data["id"] = np.arange(len(data))
    return data, bdt_features


def train_iso_bdt(data: pd.DataFrame, bdt_features: list[str],
                  params) -> tuple[dict[int, XGBClassifier], dict[str, Any], pd.DataFrame]:
    """k-fold XGBoost on the train split. params: BDTTrainingConfig.
    Returns (classifiers {fold: clf}, summary, test split with column 'score' = mean fold score)."""
    train, test = train_test_split(data, test_size=params.test_size, random_state=params.split_seed)
    counts = train["target"].value_counts()
    if min(counts.get(0, 0), counts.get(1, 0)) < params.kfold:
        raise ValueError(f"BDT training: kfold = {params.kfold} needs at least that many events per class in "
                         f"the train split, got signal {counts.get(1, 0)}, background {counts.get(0, 0)}")
    skf = StratifiedKFold(n_splits=params.kfold, random_state=params.split_seed, shuffle=True)
    classifiers, logloss, best = {}, {}, []
    test = test.copy()
    test["score"] = 0.
    for i, (train_index, valid_index) in enumerate(skf.split(train[bdt_features].values, train["target"].values)):
        fold, valid = train.iloc[train_index], train.iloc[valid_index]
        clf = XGBClassifier(
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            n_estimators=params.n_estimators,
            verbosity=1,
            subsample=params.subsample,
            colsample_bytree=params.colsample_bytree,
            min_child_weight=params.min_child_weight_fraction * np.sum(fold.weight),
            gamma=params.gamma,
            seed=params.split_seed,
            reg_alpha=params.reg_alpha,
            reg_lambda=params.reg_lambda,
            early_stopping_rounds=params.early_stopping_rounds,
            eval_metric="logloss",
        )
        clf.fit(fold[bdt_features], fold["target"],
                eval_set=[(fold[bdt_features], fold["target"]), (valid[bdt_features], valid["target"])],
                verbose=False, sample_weight=fold["weight"])
        classifiers[i] = clf
        result = clf.evals_result()
        logloss[i] = {"train": list(result["validation_0"]["logloss"]),
                      "validation": list(result["validation_1"]["logloss"])}
        best.append(int(clf.best_iteration))
        test["score"] += clf.predict_proba(test[bdt_features])[:, 1] / params.kfold
    summary = {
        "n_signal": int((data.target == 1).sum()),
        "n_background": int((data.target == 0).sum()),
        "n_train": len(train),
        "n_test": len(test),
        "auc_test": float(roc_auc_score(test["target"], test["score"])),
        "best_iteration": best,
        "features": list(bdt_features),
        "logloss": logloss,
    }
    return classifiers, summary, test


def plot_training(summary: dict, test: pd.DataFrame, classifiers: dict, bdt_features: list[str],
                  out_dir: Path) -> list[Path]:
    """logloss_fold_<i>.png per fold (train vs validation) and roc.png (test split, mean fold score)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, curves in summary["logloss"].items():
        plt.figure()
        plt.plot(curves["train"], label="Train")
        plt.plot(curves["validation"], label="Validation")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Fold {i + 1} - Loss over Epochs (ISO)")
        plt.grid(True)
        paths.append(out_dir / f"logloss_fold_{i + 1}.png")
        plt.savefig(paths[-1])
        plt.close()
    fpr, tpr, _ = roc_curve(test["target"], test["score"])
    plt.figure()
    plt.plot(fpr, tpr, label=f"ISO BDT, AUC = {summary['auc_test']:.4f}")
    plt.xscale("log")
    plt.xlabel("Background efficiency")
    plt.ylabel("Signal efficiency")
    plt.legend()
    plt.grid(True)
    paths.append(out_dir / "roc.png")
    plt.savefig(paths[-1])
    plt.close()
    return paths
