"""Live-style demo: stream a held-out subject's sensor data and classify each 2 s window.

  python demo.py                          # subject 106, all sensors
  python demo.py --subject 105 --features limited --n 40  # smartwatch-like sensors

The model is trained on all OTHER subjects, so the demo shows true cross-person generalisation.
"""
import argparse
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import data

p = argparse.ArgumentParser()
p.add_argument("--subject", type=int, default=106)
p.add_argument("--features", choices=["full", "limited"], default="full")
p.add_argument("--n", type=int, default=25, help="number of windows to stream")
p.add_argument("--delay", type=float, default=0.15, help="seconds between windows")
a = p.parse_args()

feats = data.FEATURE_SETS[a.features]
df = data.load_clean()
Xw, y, g = data.make_windows(df, feats)
X = data.window_stats(Xw)

model_path = data.ROOT / "results" / f"demo_rf_{a.features}_no{a.subject}.joblib"
if model_path.exists():
    clf = joblib.load(model_path)
else:
    print(f"Training Random Forest on subjects != {a.subject} ({a.features} features)...")
    tr = g != a.subject
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=0)
    clf.fit(X[tr], y[tr])
    joblib.dump(clf, model_path)

te = np.where(g == a.subject)[0]
te = te[np.linspace(0, len(te) - 1, a.n).astype(int)]  # spread across the whole recording
correct = 0
print(f"\nStreaming {a.n} windows from subject {a.subject} ({a.features} sensors)\n")
print(f"{'#':>3}  {'true activity':18} {'predicted':18} conf")
for i, idx in enumerate(te, 1):
    proba = clf.predict_proba(X[idx:idx + 1])[0]
    pred = clf.classes_[proba.argmax()]
    ok = pred == y[idx]
    correct += ok
    print(f"{i:>3}  {data.CLASS_NAMES[y[idx]]:18} {data.CLASS_NAMES[pred]:18} "
          f"{proba.max():.2f} {'OK' if ok else 'X'}")
    time.sleep(a.delay)
print(f"\n{correct}/{a.n} correct ({100 * correct / a.n:.0f}%)")
