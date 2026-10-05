# Human Activity Classification from Wearable Sensor Data

UE24CS352A Machine Learning mini-project. Re-implementation and extension of
[*Classifying Human Activity Using Sensor Data* (CS229, 2018)](https://cs229.stanford.edu/proj2018/report/6.pdf)
on the PAMAP2 dataset (9 subjects, 3 IMUs + heart rate, 12 activities, ~1.94M samples @100 Hz).

## What is in here
| Experiment | Description |
|---|---|
| **A** paper | Row-level samples (10 Hz), random 85/15 split — the paper's protocol. 7 models x {full 52 features, limited = hand IMU + heart rate (18)} |
| **B** subject | Same, but **subject-wise** 3-fold group CV (test subjects never seen in training) |
| **C** window | 2 s windows (50 % overlap) with statistical features (mean/std/min/max/mean-abs-diff), random vs subject-wise |
| **D** cnn | 1D-CNN (PyTorch) on raw windows, subject-wise (the paper's "future work") |

Models: Logistic Regression, RBF-SVM, Decision Tree, AdaBoost, Gradient Boosting, Random Forest,
MLP (PyTorch, 512-512, dropout 0.5), 1D-CNN (PyTorch).
RBF-SVM and AdaBoost are trained on a 40k-row random subset in the row-level experiments (too slow otherwise).

## Setup
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate     # or python -m venv
uv pip install -r requirements.txt                           # or pip install -r requirements.txt
mkdir -p data && cd data
curl -L -o pamap2.zip https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip
unzip pamap2.zip && unzip PAMAP2_Dataset.zip && cd ..        # -> data/PAMAP2_Dataset/Protocol/*.dat
```
(The UCI server sometimes stalls mid-download; just re-run the `curl`.)

## Run
```bash
python -m src.experiments A B C D     # writes results/*.json (~20 min on 16 cores + GPU)
python -m src.report                  # results/summary.md + figures
python demo.py                        # live demo: stream a held-out subject, classify each window
python demo.py --subject 105 --features limited --n 40
```

## Key results (test accuracy %, full features)
| | Random split | Subject-wise |
|---|---|---|
| Row-level, best model | ~100 (Gradient Boosting) | 62 (SVM) |
| 2 s windows, best model | 99.5 | **84.9** (MLP) |

The paper's ~98 % comes from randomly splitting 100 Hz samples, so near-identical neighbouring rows land in both
train and test (leakage). Evaluated on unseen people, row-level accuracy collapses (35-62 %), and windowed
features recover most of it (~80-85 %). Full tables: `results/summary.md`; figures: `results/*.png`.

## Layout
`src/data.py` loading/cleaning/windowing · `src/models.py` models · `src/experiments.py` experiments ·
`src/report.py` tables+figures · `demo.py` live demo
