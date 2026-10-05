## Row-level: random split (paper) vs subject-wise (test accuracy %)

| Model | Full: random | Full: subject | Limited: random | Limited: subject |
|---|---|---|---|---|
| Logistic Regression | 84.5 | 55.6 ± 6.2 | 66.6 | 52.2 ± 5.5 |
| SVM (RBF) | 98.5 | 62.0 ± 6.5 | 93.9 | 51.3 ± 2.3 |
| Decision Tree | 97.7 | 35.3 ± 5.4 | 96.7 | 36.6 ± 4.5 |
| AdaBoost | 99.8 | 46.7 ± 3.3 | 98.5 | 37.5 ± 1.9 |
| Gradient Boosting | 100.0 | 47.5 ± 0.8 | 99.8 | 37.6 ± 4.3 |
| Random Forest | 99.7 | 60.1 ± 8.9 | 98.3 | 50.6 ± 2.7 |
| MLP (PyTorch) | 99.4 | 60.0 ± 10.1 | 94.7 | 53.5 ± 2.5 |

## 2 s windows + statistical features (test accuracy %)

| Model | Full: random | Full: subject | Limited: random | Limited: subject |
|---|---|---|---|---|
| Logistic Regression | 95.1 | 81.0 ± 5.4 | 88.3 | 71.8 ± 6.2 |
| SVM (RBF) | 99.2 | 80.0 ± 7.3 | 98.0 | 71.1 ± 10.5 |
| Decision Tree | 94.6 | 56.5 ± 6.5 | 91.7 | 50.8 ± 16.3 |
| AdaBoost | 99.5 | 78.5 ± 2.2 | 98.4 | 67.9 ± 11.9 |
| Gradient Boosting | 99.5 | 74.1 ± 0.7 | 99.0 | 63.4 ± 12.2 |
| Random Forest | 98.1 | 83.2 ± 5.8 | 97.0 | 72.5 ± 9.9 |
| MLP (PyTorch) | 98.9 | 84.9 ± 4.0 | 95.7 | 72.9 ± 9.2 |
| 1D-CNN (raw windows) | - | 71.4 ± 7.5 | - | 65.5 ± 13.5 |
