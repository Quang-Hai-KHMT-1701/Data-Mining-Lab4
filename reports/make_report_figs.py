from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


def load_json(*candidates: Path) -> tuple[dict, Path]:
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), path
    raise FileNotFoundError(
        "No JSON file found in candidates: " + ", ".join(str(p) for p in candidates)
    )


def plot_classification(conf_json: dict) -> None:
    labels: list[str] = conf_json["labels"]
    cm = np.array(conf_json["confusion_matrix"], dtype=int)
    report = conf_json["report"]
    per_class = [
        (
            label,
            report[label]["precision"],
            report[label]["recall"],
            report[label]["f1-score"],
        )
        for label in labels
        if label in report
    ]
    metrics_df = pd.DataFrame(per_class, columns=["label", "precision", "recall", "f1"])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title("Confusion matrix (classification AQI)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "classification_confusion_matrix.png", dpi=200)
    plt.close()

    melted = metrics_df.melt(
        id_vars="label",
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="label", y="score", hue="metric", palette="Set2")
    plt.ylim(0, 1)
    plt.title("Precision/Recall/F1 theo lớp")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "classification_prf_per_class.png", dpi=200)
    plt.close()


def plot_regression(reg_df: pd.DataFrame, reg_metrics: dict) -> None:
    sample = reg_df.sample(min(len(reg_df), 4000), random_state=42)
    min_val = float(min(sample["y_true"].min(), sample["y_pred"].min()))
    max_val = float(max(sample["y_true"].max(), sample["y_pred"].max()))

    plt.figure(figsize=(6.5, 6))
    plt.scatter(sample["y_true"], sample["y_pred"], alpha=0.35, s=12, color="#1f77b4")
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        label="y = x",
    )
    plt.xlabel("Thực tế PM2.5")
    plt.ylabel("Dự đoán PM2.5")
    plt.title("Regression: y_true vs y_pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regression_scatter.png", dpi=200)
    plt.close()

    residuals = reg_df["y_pred"] - reg_df["y_true"]
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, bins=60, kde=True, color="#ff7f0e")
    plt.title("Phân phối residual (y_pred - y_true)")
    plt.xlabel("Residual")
    plt.ylabel("Tần suất")
    plt.axvline(residuals.mean(), color="k", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regression_residuals.png", dpi=200)
    plt.close()

    # RMSE/MAE inset text for quick reference
    plt.figure(figsize=(4.5, 2.2))
    plt.axis("off")
    plt.text(0.02, 0.7, f"RMSE: {reg_metrics.get('rmse', 'N/A'):.2f}", fontsize=11)
    plt.text(0.02, 0.45, f"MAE:  {reg_metrics.get('mae', 'N/A'):.2f}", fontsize=11)
    plt.text(0.02, 0.2, f"R2:   {reg_metrics.get('r2', 'N/A'):.3f}", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regression_metrics_card.png", dpi=200)
    plt.close()


def plot_arima(arima_df: pd.DataFrame, summary: dict) -> None:
    arima_df["datetime"] = pd.to_datetime(arima_df["datetime"])
    subset = arima_df.head(240)

    plt.figure(figsize=(10, 4.5))
    plt.plot(subset["datetime"], subset["y_true"], label="Thực tế", color="#1f77b4")
    plt.plot(subset["datetime"], subset["y_pred"], label="Dự báo", color="#ff7f0e")
    plt.fill_between(
        subset["datetime"],
        subset["lower"],
        subset["upper"],
        color="#ff7f0e",
        alpha=0.15,
        label="Khoảng tin cậy",
    )
    plt.title(f"ARIMA {tuple(summary.get('best_order', []))} dự báo PM2.5")
    plt.xlabel("Thời gian")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "arima_forecast_window.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 2.5))
    plt.axis("off")
    plt.text(0.02, 0.7, f"RMSE: {summary.get('rmse', 'N/A'):.2f}", fontsize=11)
    plt.text(0.02, 0.45, f"MAE:  {summary.get('mae', 'N/A'):.2f}", fontsize=11)
    plt.text(
        0.02, 0.2, f"Best (p,d,q): {tuple(summary.get('best_order', []))}", fontsize=11
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "arima_metrics_card.png", dpi=200)
    plt.close()


def main() -> None:
    clf_metrics, clf_path = load_json(
        ROOT / "data" / "processed" / "metrics.json",
        ROOT.parent / "data" / "processed" / "metrics.json",
    )
    reg_metrics, _ = load_json(ROOT / "data" / "processed" / "regression_metrics.json")
    arima_summary, _ = load_json(
        ROOT / "data" / "processed" / "arima_pm25_summary.json"
    )

    reg_df = pd.read_csv(
        ROOT / "data" / "processed" / "regression_predictions_sample.csv"
    )
    arima_df = pd.read_csv(ROOT / "data" / "processed" / "arima_pm25_predictions.csv")

    plot_classification(clf_metrics)
    plot_regression(reg_df, reg_metrics)
    plot_arima(arima_df, arima_summary)

    print("Figures written to", OUT_DIR)
    print("Classification metrics source:", clf_path)


if __name__ == "__main__":
    main()
