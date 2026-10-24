"""ISO BDT training (port of the ISO mode of Train_XGB_W.py).

Events and derived features come from the stage 2 code (BDT_ScoreProducer._load / _add_features,
bdt_inputs.bdt_branches), so training and inference see the same columns in the same order:
one row per event (candidate 0), rows with NaN or inf dropped. Signal: all events. Background:
events with all candidates in the sideband. Background weight 1, signal rescaled to the background total. The classifier pickle {fold: XGBClassifier} is the format
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
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from tau3mu_gnn.bdt_training.BDT_ScoreProducer import PREFIX, _add_features, _load  # noqa: E402
from tau3mu_gnn.bdt_training.bdt_inputs import bdt_branches  # noqa: E402

ROC_WORKING_POINTS = [0.80, 0.82, 0.84, 0.86, 0.89]   # BDT cuts marked on the ROC (as Train_XGB_W.py)
MASS = f"{PREFIX}TauTo3Mu_fitted_mass"


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
    # XGBoost fit rejects inf (e.g. chi2 / Ndof with Ndof = 0); dropna above does not remove it
    sig, bkg = (df[np.isfinite(df[bdt_features].to_numpy(dtype=float)).all(axis=1)].reset_index(drop=True)
                for df in (sig, bkg))
    sig["target"], bkg["target"] = 1, 0
    # Background weights stay 1; signal is rescaled to the background total (Kaito, PR #28). This
    # corrects for dropped rows and a class below max_events_per_class. The old script used 1 for both.
    sig["weight"], bkg["weight"] = len(bkg) / len(sig), 1.
    data = pd.concat([sig, bkg], ignore_index=True)
    data["id"] = np.arange(len(data))
    return data, bdt_features


def train_iso_bdt(data: pd.DataFrame, bdt_features: list[str],
                  params) -> tuple[dict[int, XGBClassifier], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """k-fold XGBoost on the train split. params: BDTTrainingConfig.
    Returns (classifiers {fold: clf}, summary, train split, test split); both splits get the column
    'score' = mean fold score (as stage 2). summary['ks_pvalue']: KS test train vs test score per class."""
    train, test = train_test_split(data, test_size=params.test_size, random_state=params.split_seed)
    counts = train["target"].value_counts()
    if min(counts.get(0, 0), counts.get(1, 0)) < params.kfold:
        raise ValueError(f"BDT training: kfold = {params.kfold} needs at least that many events per class in "
                         f"the train split, got signal {counts.get(1, 0)}, background {counts.get(0, 0)}")
    skf = StratifiedKFold(n_splits=params.kfold, random_state=params.split_seed, shuffle=True)
    classifiers, logloss, best = {}, {}, []
    train, test = train.copy(), test.copy()
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
    train["score"] = np.mean([c.predict_proba(train[bdt_features])[:, 1] for c in classifiers.values()], axis=0)
    summary = {
        "n_signal": int((data.target == 1).sum()),
        "n_background": int((data.target == 0).sum()),
        "n_train": len(train),
        "n_test": len(test),
        "auc_test": float(roc_auc_score(test["target"], test["score"])),
        "ks_pvalue": {name: float(ks_2samp(train.score[train.target == t], test.score[test.target == t]).pvalue)
                      for name, t in (("signal", 1), ("background", 0))},
        "best_iteration": best,
        "features": list(bdt_features),
        "logloss": logloss,
    }
    return classifiers, summary, train, test


def roc_data(test: pd.DataFrame) -> dict[str, Any]:
    """ROC of the test split: {fpr, tpr, threshold, auc} (the old _roc_log*.pck)."""
    fpr, tpr, thr = roc_curve(test["target"], test["score"])
    return {"fpr": fpr, "tpr": tpr, "threshold": thr, "auc": float(roc_auc_score(test["target"], test["score"]))}


def working_points(fpr, tpr, threshold, cuts) -> tuple[list[float], list[float]]:
    """(fpr, tpr) of each cut 'score > cut'. roc_curve point k is 'score >= threshold[k]' and the
    thresholds fall, so the cut is the last point with threshold > cut (threshold[0] = inf)."""
    idx = [max(int((threshold > c).sum()) - 1, 0) for c in cuts]
    return [float(fpr[i]) for i in idx], [float(tpr[i]) for i in idx]


def feature_importance(classifiers: dict, bdt_features: list[str]) -> dict[str, float]:
    """Relative F-score (number of splits) per feature, summed over the folds; sums to 1."""
    scores = dict.fromkeys(bdt_features, 0.)
    for clf in classifiers.values():
        for name, n in clf.get_booster().get_fscore().items():
            scores[name] += n
    total = sum(scores.values())
    return {name: n / total for name, n in scores.items()}


def _label(feature: str) -> str:
    return feature.replace(f"{PREFIX}TauTo3Mu_", "")


def _plot_roc(roc: dict, path: Path, log_x: bool) -> None:
    plt.figure()
    plt.plot(roc["fpr"], roc["tpr"], color="r", label=f"ISO BDT (test), AUC = {roc['auc']:.4f}")
    wp_x, wp_y = working_points(roc["fpr"], roc["tpr"], roc["threshold"], ROC_WORKING_POINTS)
    plt.scatter(wp_x, wp_y)
    for cut, x, y in zip(ROC_WORKING_POINTS, wp_x, wp_y):
        plt.annotate(f"{cut:.2f}", (x, y))
    if log_x:
        plt.xscale("log")
        plt.xlim(1e-5, 1)
    plt.ylim(0, 1)
    plt.xlabel("Background efficiency")
    plt.ylabel("Signal efficiency")
    plt.title("ROC log" if log_x else "ROC")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_overtrain(summary: dict, train: pd.DataFrame, test: pd.DataFrame, paths: tuple[Path, Path]) -> None:
    """Score of train (filled) vs test (points, Poisson errors) per class, unit area; linear and log y."""
    bins = np.linspace(0, 1, 51)
    centre = (bins[:-1] + bins[1:]) / 2
    plt.figure()
    for name, t, colour in (("S", 1, "r"), ("B", 0, "b")):
        plt.hist(train.score[train.target == t], bins=bins, density=True, histtype="stepfilled", alpha=0.5,
                 color=colour, label=f"{name} (train)")
        scores = test.score[test.target == t]
        counts, _ = np.histogram(scores, bins=bins)
        norm = len(scores) * (bins[1] - bins[0])
        plt.errorbar(centre, counts / norm, yerr=np.sqrt(counts) / norm, fmt="o", color=colour, label=f"{name} (test)")
    ks = summary["ks_pvalue"]
    plt.title(f"KS p-values: sig = {ks['signal'] * 100:.2f}%, bkg = {ks['background'] * 100:.2f}%")
    plt.xlabel("BDT output")
    plt.ylabel("Count density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(paths[0])
    plt.yscale("log")
    plt.savefig(paths[1])
    plt.close()


def _plot_feature_importance(classifiers: dict, bdt_features: list[str], path: Path) -> None:
    scores = sorted(feature_importance(classifiers, bdt_features).items(), key=lambda kv: kv[1])
    plt.figure(figsize=(8, 6))
    plt.barh(np.arange(len(scores)), [v for _, v in scores])
    plt.yticks(np.arange(len(scores)), [_label(k) for k, _ in scores])
    plt.xlabel("relative F-score")
    plt.ylabel("feature")
    plt.title("Feature Importance - ISO BDT")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_correlation(rows: pd.DataFrame, columns: list[str], title: str, path: Path) -> None:
    """Linear correlation matrix with the values written in the cells."""
    corr = rows[columns].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, shrink=0.8)
    labels = [_label(c) for c in columns]
    ax.set_xticks(np.arange(len(columns)), labels, rotation="vertical")
    ax.set_yticks(np.arange(len(columns)), labels)
    for (i, j), v in np.ndenumerate(corr):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_training(summary: dict, train: pd.DataFrame, test: pd.DataFrame, classifiers: dict,
                  bdt_features: list[str], out_dir: Path) -> list[Path]:
    """The plots of the old Train_XGB_W.py (ISO only): logloss_fold_<i>.png per fold (train vs
    validation); roc.png / roc_log.png (test split, ROC_WORKING_POINTS marked); overtrain.png /
    overtrain_log.png (train vs test score, KS p-values); feat_importance.png; corr_sig.png /
    corr_bkg.png (features, score and fitted mass, all events of the class)."""
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
    roc = roc_data(test)
    for name, log_x in (("roc.png", False), ("roc_log.png", True)):
        paths.append(out_dir / name)
        _plot_roc(roc, paths[-1], log_x)
    paths += [out_dir / "overtrain.png", out_dir / "overtrain_log.png"]
    _plot_overtrain(summary, train, test, (paths[-2], paths[-1]))
    paths.append(out_dir / "feat_importance.png")
    _plot_feature_importance(classifiers, bdt_features, paths[-1])
    scored = pd.concat([train, test])
    for name, t, cls in (("corr_sig.png", 1, "signal"), ("corr_bkg.png", 0, "background")):
        paths.append(out_dir / name)
        _plot_correlation(scored[scored.target == t], list(bdt_features) + ["score", MASS],
                          f"linear correlation matrix - {cls} - ISO BDT", paths[-1])
    return paths
