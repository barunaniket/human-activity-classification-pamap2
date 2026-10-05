"""Run the paper reproduction and the two extensions.

  A  paper     : row-level samples, random 85/15 split (as in the paper)
  B  subject   : row-level samples, subject-wise 3-fold group CV (no subject in train & test)
  C  window    : 2 s windows + statistical features, random split  vs  subject-wise
  D  cnn       : 1D-CNN on raw windows, subject-wise (the paper's "future work")

Usage: python -m src.experiments [A B C D]
"""
import json
import sys
import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, train_test_split

from . import data
from .models import TorchClassifier, get_models

RES = data.ROOT / "results"
SEED = 0
SLOW_MAX_TRAIN = 40000  # RBF-SVM / AdaBoost are too slow on ~160k rows: cap their training set
SLOW_MODELS = ("SVM", "AdaBoost")


def _fit_eval(name, model, Xtr, ytr, Xte, yte):
    if name.startswith(SLOW_MODELS) and len(Xtr) > SLOW_MAX_TRAIN:
        idx = np.random.RandomState(SEED).choice(len(Xtr), SLOW_MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    t = time.time()
    m = clone(model).fit(Xtr, ytr)
    fit_s = time.time() - t
    t = time.time()
    pred = m.predict(Xte)
    pred_s = time.time() - t
    n_cls = len(data.LABEL_IDS)
    return {
        "train_acc": float(accuracy_score(ytr, m.predict(Xtr))),
        "test_acc": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "fit_s": fit_s, "predict_s_per_1k": pred_s / len(Xte) * 1000,
        "confusion": confusion_matrix(yte, pred, labels=range(n_cls)).tolist(),
    }


def _log(tag, name, fs, r):
    print(f"[{tag}] {fs:8s} {name:20s} train={r['train_acc']:.3f} test={r['test_acc']:.3f} "
          f"f1={r['macro_f1']:.3f} fit={r['fit_s']:.0f}s", flush=True)


def exp_paper(df):
    out = {}
    for fs, feats in data.FEATURE_SETS.items():
        X, y, _ = data.row_level(df, feats, step=10)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=SEED)
        for name, model in get_models(SEED).items():
            r = _fit_eval(name, model, Xtr, ytr, Xte, yte)
            out[f"{fs}|{name}"] = r
            _log("A", name, fs, r)
    return out


def _group_cv(X, y, g, model, name, k=3):
    folds = []
    for tr, te in GroupKFold(n_splits=k).split(X, y, g):
        r = _fit_eval(name, model, X[tr], y[tr], X[te], y[te])
        r["test_subjects"] = sorted(set(g[te].tolist()))
        folds.append(r)
    conf = np.sum([f.pop("confusion") for f in folds], axis=0)
    agg = {k_: float(np.mean([f[k_] for f in folds]))
           for k_ in ("train_acc", "test_acc", "macro_f1", "fit_s", "predict_s_per_1k")}
    agg["test_acc_std"] = float(np.std([f["test_acc"] for f in folds]))
    agg["folds"] = folds
    agg["confusion"] = conf.tolist()
    return agg


def exp_subject(df):
    out = {}
    for fs, feats in data.FEATURE_SETS.items():
        X, y, g = data.row_level(df, feats, step=10)
        for name, model in get_models(SEED).items():
            r = _group_cv(X, y, g, model, name)
            out[f"{fs}|{name}"] = r
            _log("B", name, fs, r)
    return out


def exp_window(df):
    out = {}
    for fs, feats in data.FEATURE_SETS.items():
        Xw, y, g = data.make_windows(df, feats)
        X = data.window_stats(Xw)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
        for name, model in get_models(SEED).items():
            r = _fit_eval(name, model, Xtr, ytr, Xte, yte)
            out[f"random|{fs}|{name}"] = r
            _log("C-rand", name, fs, r)
            r = _group_cv(X, y, g, model, name)
            out[f"subject|{fs}|{name}"] = r
            _log("C-subj", name, fs, r)
    return out


def exp_cnn(df):
    out = {}
    for fs, feats in data.FEATURE_SETS.items():
        Xw, y, g = data.make_windows(df, feats)
        r = _group_cv(Xw, y, g, TorchClassifier("cnn", epochs=25, batch_size=128, seed=SEED),
                      "CNN")
        out[f"subject|{fs}|1D-CNN (PyTorch)"] = r
        _log("D", "1D-CNN", fs, r)
    return out


EXPERIMENTS = {"A": ("paper", exp_paper), "B": ("subject", exp_subject),
               "C": ("window", exp_window), "D": ("cnn", exp_cnn)}

if __name__ == "__main__":
    which = sys.argv[1:] or list(EXPERIMENTS)
    RES.mkdir(exist_ok=True)
    df = data.load_clean()
    print(f"data: {len(df):,} rows, subjects={sorted(df.subject.unique())}", flush=True)
    for key in which:
        name, fn = EXPERIMENTS[key]
        res = fn(df)
        (RES / f"{name}.json").write_text(json.dumps(res))
        print(f"saved results/{name}.json", flush=True)
