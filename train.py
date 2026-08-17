"""
Mushroom Classification - Decision Tree mini-project
โจทย์: จำแนกเห็ดกินได้ (edible) / มีพิษ (poisonous) จากลักษณะทางกายภาพ
"""
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------- 1. Load & decode ----------
attr_names = []
attr_maps = {}
with open("descriptors.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, rest = line.split(":", 1)
        attr_names.append(name)
        pairs = [p.split("=") for p in rest.split(",")]
        attr_maps[name] = {code: label for label, code in pairs}

df = pd.read_csv("agaricus-lepiota.data.txt", names=attr_names)
for col in df.columns:
    df[col] = df[col].map(attr_maps[col])

df = df.rename(columns={"poisonous": "class"})
print("Shape:", df.shape)
print(df.head(3))
print("\nMissing (?) counts before decode-check:")
print((df == "?").sum()[lambda s: s > 0])

# ---------- 2. Preprocessing ----------
# 2a. veil-type is constant -> drop (no predictive information)
nunique = df.nunique()
constant_cols = nunique[nunique == 1].index.tolist()
print("\nConstant columns dropped:", constant_cols)
df = df.drop(columns=constant_cols)

# 2b. stalk-root missing values ('missing' after decode) kept as its own category
print("\nstalk-root value counts:\n", df["stalk-root"].value_counts())

# 2c. split features / target
y = (df["class"] == "poisonous").astype(int)  # 1 = poisonous, 0 = edible
X = df.drop(columns=["class"])

# save category options (human-readable) for building the Streamlit UI, in original column order
category_options = {col: sorted(X[col].unique().tolist()) for col in X.columns}
with open("category_options.json", "w", encoding="utf-8") as f:
    json.dump(category_options, f, ensure_ascii=False, indent=2)

# 2d. one-hot encode
X_encoded = pd.get_dummies(X, columns=X.columns)
feature_columns = X_encoded.columns.tolist()
with open("feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(feature_columns, f, ensure_ascii=False, indent=2)

print("\nEncoded feature matrix shape:", X_encoded.shape)

# ---------- 3. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain/Test sizes:", X_train.shape, X_test.shape)

# ---------- 4. Train models ----------
models = {
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

results = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append({
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    })
    fitted[name] = model

results_df = pd.DataFrame(results)
print("\n=== Model comparison ===")
print(results_df)
results_df.to_json("model_comparison.json", orient="records", indent=2)

# confusion matrix for primary model (Decision Tree)
cm = confusion_matrix(y_test, fitted["Decision Tree"].predict(X_test))
print("\nDecision Tree confusion matrix:\n", cm)
np.save("dt_confusion_matrix.npy", cm)

# ---------- 5. Feature importance (Decision Tree) ----------
importances = pd.Series(fitted["Decision Tree"].feature_importances_, index=feature_columns)
top_importances = importances.sort_values(ascending=False).head(10)
print("\nTop 10 feature importances (Decision Tree):\n", top_importances)
top_importances.to_json("dt_feature_importance.json", indent=2)

# tree rules (first few levels) - useful for explaining model theory
tree_rules = export_text(fitted["Decision Tree"], feature_names=feature_columns, max_depth=3)
with open("dt_rules_preview.txt", "w", encoding="utf-8") as f:
    f.write(tree_rules)
print("\n--- Decision tree rules (preview) ---")
print(tree_rules[:1500])

# ---------- 6. Save primary model ----------
joblib.dump(fitted["Decision Tree"], "dt_model.pkl")
print("\nSaved dt_model.pkl, feature_columns.json, category_options.json, model_comparison.json")
