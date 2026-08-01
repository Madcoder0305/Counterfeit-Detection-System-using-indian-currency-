# ============================================================
# Model Comparison — Indian Rupee Counterfeit Detection
# Compares CNN, SVC, and Random Forest performance on INR data
# ============================================================

import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "models", "results")

def load_metrics():
    metrics = {}
    for name, fname in [("CNN","cnn_metrics.json"),("SVC","svc_metrics.json"),("Random Forest","rf_metrics.json")]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                metrics[name] = json.load(f)
        else:
            print(f"  ⚠️ Missing: {fname}")
    return metrics

def compare():
    print("\n" + "━"*62)
    print("  INR Counterfeit Detection — Model Performance Comparison")
    print("━"*62 + "\n")
    metrics = load_metrics()
    if not metrics:
        print("  No metrics found. Train models first."); return
    header = f"  {'Model':<16}| {'Accuracy':>9} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9}"
    print(header)
    print("  " + "-"*len(header.strip()))
    report = {"currency": "INR", "models": {}}
    for name, m in metrics.items():
        print(f"  {name:<16}| {m['accuracy']*100:>8.1f}% | {m['precision']*100:>8.1f}% | {m['recall']*100:>8.1f}% | {m['f1_score']*100:>8.1f}%")
        report["models"][name] = m
    with open(os.path.join(RESULTS_DIR, "comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    # Chart
    names = list(metrics.keys())
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0A0A0F")
    ax.set_facecolor("#12121A")
    colors = ["#3D8EF0", "#00D4AA", "#FF6B6B", "#FFB84D"]
    for i, metric_key in enumerate(["accuracy","precision","recall","f1_score"]):
        vals = [metrics[n][metric_key]*100 for n in names]
        bars = ax.bar(x + i*w, vals, w, label=metric_key.replace("_"," ").title(), color=colors[i], alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xlabel("Model", color="white", fontsize=12)
    ax.set_ylabel("Score (%)", color="white", fontsize=12)
    ax.set_title("INR Counterfeit Detection — Model Performance Comparison", color="white", fontsize=14, fontweight="bold")
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(names, color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 110)
    ax.legend(facecolor="#1a1a2e", edgecolor="#3D8EF0", labelcolor="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    chart_path = os.path.join(RESULTS_DIR, "comparison_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Chart saved → {chart_path}")
    print("  ✅ Comparison complete.\n")

if __name__ == "__main__":
    compare()
