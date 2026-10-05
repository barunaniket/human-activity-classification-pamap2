"""Model zoo: six sklearn/PyTorch classifiers (paper) + a 1D-CNN on raw windows (extension)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- PyTorch MLP
class _MLP(nn.Module):
    """Paper architecture: n -> 512 -> 512 -> k, ReLU, dropout 0.5 on each hidden layer."""

    def __init__(self, n_in, n_out, hidden=512, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, n_out),  # softmax is folded into the cross-entropy loss
        )

    def forward(self, x):
        return self.net(x)


class _CNN1D(nn.Module):
    """Small 1D-CNN over raw (T, C) windows."""

    def __init__(self, n_ch, n_out, p=0.5):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(),
                                 nn.MaxPool1d(2))
        self.features = nn.Sequential(block(n_ch, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(p),
                                  nn.Linear(128, n_out))

    def forward(self, x):  # x: (B, T, C)
        return self.head(self.features(x.transpose(1, 2)))


class TorchClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-style wrapper so the torch nets plug into the same evaluation code.

    arch='mlp' expects (N, F) input, arch='cnn' expects (N, T, C). Inputs are standardised
    with train statistics (per feature / per channel).
    """

    def __init__(self, arch="mlp", epochs=30, lr=1e-3, batch_size=512, weight_decay=1e-4,
                 seed=0):
        self.arch, self.epochs, self.lr = arch, epochs, lr
        self.batch_size, self.weight_decay, self.seed = batch_size, weight_decay, seed

    def _prep(self, X):
        return torch.as_tensor((X - self.mu_) / self.sd_, dtype=torch.float32)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        axes = tuple(range(X.ndim - 1))
        self.mu_ = X.mean(axes, keepdims=True)
        self.sd_ = X.std(axes, keepdims=True) + 1e-6
        self.classes_ = np.unique(y)
        y_idx = torch.as_tensor(np.searchsorted(self.classes_, y))
        Xt = self._prep(X)
        n_out = len(self.classes_)
        self.model_ = (_MLP(X.shape[-1], n_out) if self.arch == "mlp"
                       else _CNN1D(X.shape[-1], n_out)).to(DEVICE)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        loss_fn = nn.CrossEntropyLoss()
        Xt, y_idx = Xt.to(DEVICE), y_idx.to(DEVICE)
        n = len(Xt)
        self.history_ = []
        for _ in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(n, device=DEVICE)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(Xt[idx]), y_idx[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            sched.step()
            self.history_.append(total / n)
        return self

    @torch.no_grad()
    def predict(self, X):
        self.model_.eval()
        Xt = self._prep(X)
        out = [self.model_(Xt[i:i + 4096].to(DEVICE)).argmax(1).cpu()
               for i in range(0, len(Xt), 4096)]
        return self.classes_[torch.cat(out).numpy()]


# ---------------------------------------------------------------- model zoo
def get_models(seed=0, svm_c=100.0, lr_c=0.01):
    """Hyperparameters follow the paper's CV results unless tuned in `tune.py`."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=lr_c, solver="saga", max_iter=300, random_state=seed)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=svm_c, kernel="rbf", cache_size=2000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=seed),
        "AdaBoost": AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=9), n_estimators=100, random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=seed),
        "MLP (PyTorch)": TorchClassifier("mlp", epochs=30, seed=seed),
    }
