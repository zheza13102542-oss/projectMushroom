import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#2A2118"
GREEN = "#33513F"
OCHRE = "#A85C32"
TEAL = "#3E6E71"

results = pd.read_json("model_comparison.json")
metrics = ["accuracy", "precision", "recall", "f1"]

fig, ax = plt.subplots(figsize=(7, 4.2))
x = range(len(results))
width = 0.2
colors = [GREEN, OCHRE, TEAL, "#8A7B5C"]
for i, m in enumerate(metrics):
    ax.bar([xi + i * width for xi in x], results[m], width=width, label=m.capitalize(), color=colors[i])
ax.set_xticks([xi + width * 1.5 for xi in x])
ax.set_xticklabels(results["model"])
ax.set_ylim(0.98, 1.005)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Decision Tree vs Random Forest vs KNN")
ax.legend(loc="lower right", ncol=4, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150)
print("saved comparison_chart.png")

imp = pd.read_json("dt_feature_importance.json", typ="series")
fig2, ax2 = plt.subplots(figsize=(7, 4.2))
imp.sort_values().plot(kind="barh", ax=ax2, color=GREEN)
ax2.set_xlabel("Feature importance")
ax2.set_title("Top 10 Most Important Features (Decision Tree)")
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("saved feature_importance.png")
