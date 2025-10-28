from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc(y_true, y_score, out_path: Path):
    """Рисует ROC-кривую и сохраняет картинку."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(out_path)
    plt.show()
    #plt.close()
    print(f"ROC saved to: {out_path}")


def evaluate_and_plot(metrics, artifacts_dir):
    """Печатает метрики и рисует ROC, если есть val_predictions.pkl."""
    print("Metrics:", metrics)

    val_path = Path(artifacts_dir) / "val_predictions.pkl"
    if not val_path.exists():
        print("No val_predictions.pkl found — skipping ROC.")
        return

    vp = joblib.load(val_path)
    y_true, y_proba = vp.get("y_val"), vp.get("proba")
    if y_true is None or y_proba is None:
        print("Invalid val_predictions.pkl — missing keys.")
        return

    out_path = Path(artifacts_dir) / "roc.png"
    plot_roc(y_true, y_proba, out_path)
