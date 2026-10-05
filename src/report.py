"""Turn results/*.json into tables (results/summary.md) and figures (results/*.png)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .data import CLASS_NAMES, ROOT

RES = ROOT / "results"


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def acc(r):
    s = f"{100 * r['test_acc']:.1f}"
    return s + (f" ± {100 * r['test_acc_std']:.1f}" if "test_acc_std" in r else "")


def main():
    paper, subj, win, cnn = load("paper"), load("subject"), load("window"), load("cnn")
    md = []
    models = sorted({k.split("|")[-1] for k in paper} | {k.split("|")[-1] for k in subj},
                    key=lambda m: m)
    if paper and subj:
        rows = [[m, acc(paper[f"full|{m}"]), acc(subj[f"full|{m}"]),
                 acc(paper[f"limited|{m}"]), acc(subj[f"limited|{m}"])]
                for m in paper_order(paper)]
        md += ["## Row-level: random split (paper) vs subject-wise (test accuracy %)",
               table(rows, ["Model", "Full: random", "Full: subject", "Limited: random",
                            "Limited: subject"])]
    if win:
        order = paper_order({k.split("|", 1)[1]: v for k, v in win.items()})
        rows = [[m] + [acc(win[f"{s}|{f}|{m}"]) for f in ("full", "limited")
                       for s in ("random", "subject")] for m in order]
        extra = [[ "1D-CNN (raw windows)", "-", acc(cnn["subject|full|1D-CNN (PyTorch)"]), "-",
                  acc(cnn["subject|limited|1D-CNN (PyTorch)"])]] if cnn else []
        md += ["## 2 s windows + statistical features (test accuracy %)",
               table(rows + extra, ["Model", "Full: random", "Full: subject", "Limited: random",
                                    "Limited: subject"])]
    (RES / "summary.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))

    # confusion matrix of the best subject-wise windowed model (full & limited)
    if win:
        for fs in ("full", "limited"):
            cands = {k: v for k, v in win.items() if k.startswith(f"subject|{fs}|")}
            best = max(cands, key=lambda k: cands[k]["test_acc"])
            cm = np.array(cands[best]["confusion"], dtype=float)
            cm = cm / cm.sum(1, keepdims=True).clip(1)
            plt.figure(figsize=(8, 6.5))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES,
                        yticklabels=CLASS_NAMES, cbar=False)
            plt.title(f"{best.split('|')[-1]} - {fs} features, subject-wise (row-normalised)")
            plt.xlabel("predicted"); plt.ylabel("true"); plt.xticks(rotation=45, ha="right")
            plt.tight_layout(); plt.savefig(RES / f"confusion_{fs}.png", dpi=150); plt.close()

    # random-vs-subject gap bar chart (row-level, full features)
    if paper and subj:
        ms = paper_order(paper)
        x = np.arange(len(ms)); w = 0.38
        plt.figure(figsize=(9, 4.5))
        plt.bar(x - w / 2, [100 * paper[f"full|{m}"]["test_acc"] for m in ms], w,
                label="random split (paper)")
        plt.bar(x + w / 2, [100 * subj[f"full|{m}"]["test_acc"] for m in ms], w,
                label="subject-wise split")
        plt.xticks(x, ms, rotation=25, ha="right"); plt.ylabel("test accuracy (%)")
        plt.title("Effect of evaluation protocol (full features, row-level)")
        plt.legend(); plt.tight_layout(); plt.savefig(RES / "random_vs_subject.png", dpi=150)
        plt.close()


def paper_order(d):
    keys = {k.split("|")[-1] for k in d}
    pref = ["Logistic Regression", "SVM (RBF)", "Decision Tree", "AdaBoost", "Gradient Boosting", "Random Forest",
            "MLP (PyTorch)"]
    return [m for m in pref if m in keys]


if __name__ == "__main__":
    main()
