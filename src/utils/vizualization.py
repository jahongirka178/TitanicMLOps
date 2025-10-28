from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import optuna
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
    # plt.close()
    print(f"ROC saved to: {out_path}")


def show_metrics(metrics):
    """Печатает метрики"""
    print(f"Metrics: {metrics}")


def plot_roc_from_artifacts(artifacts_dir):
    """считывает val_predictions и рисует по ним roc-auc"""
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


def plot_optuna_results(study: optuna.Study, out_dir: Path):
    """Рисует базовые графики по результатам Optuna и сохраняет картинки."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. История оптимизации
    fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig1.figure.savefig(out_dir / "optuna_history.png", bbox_inches="tight")
    plt.show()
    #plt.close(fig1.figure)

    # 2. Параллельные координаты
    fig2 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    fig2.figure.savefig(out_dir / "optuna_parallel_coords.png", bbox_inches="tight")
    plt.show()
    #plt.close(fig2.figure)

    # 3. Важность параметров
    fig3 = optuna.visualization.matplotlib.plot_param_importances(study)
    fig3.figure.savefig(out_dir / "optuna_param_importances.png", bbox_inches="tight")
    plt.show()
    #plt.close(fig3.figure)

    print(f"Optuna visualizations saved to: {out_dir}")
