import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 讀取資料
df = pd.read_csv('0050_cleaned_data_5years.csv')

# 2. 定義特徵 (X) 與 目標 (y)
feature_columns = ['ma5_bias', 'kline_body', 'amplitude', 'vol_change']
X = df[feature_columns]
y = df['target']

# 3. 切割資料 (80% 用於訓練模型，20% 作為從未見過的測試資料)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 初始化決策樹模型 (限制深度為 3 層，避免 Overfitting)
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# 讓模型從訓練資料中尋找模式 (開始學習)
clf.fit(X_train, y_train)

# 5. 預測並計算準確率
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"模型預測準確率: {accuracy * 100:.2f}%")
print("\n詳細分類報告:")
print(classification_report(y_test, y_pred))

# 建立視覺化輸出資料夾，將圖片保留在本機
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

# 6. 視覺化：決策樹結構
fig, ax = plt.subplots(figsize=(18, 9))
plot_tree(
	clf,
	feature_names=X.columns,
	class_names=[str(c) for c in clf.classes_],
	filled=True,
	rounded=True,
	fontsize=8,
	ax=ax,
)
ax.set_title("Decision Tree Structure")
plt.tight_layout()
tree_path = output_dir / "decision_tree_structure.png"
fig.savefig(tree_path, dpi=300, bbox_inches="tight")
print(f"已儲存視覺化圖片: {tree_path}")
plt.show()

# 7. 視覺化：混淆矩陣
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix")
plt.tight_layout()
cm_path = output_dir / "confusion_matrix.png"
fig.savefig(cm_path, dpi=300, bbox_inches="tight")
print(f"已儲存視覺化圖片: {cm_path}")
plt.show()

# 8. 視覺化：特徵重要度
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
feature_importance.plot(kind="bar", ax=ax)
ax.set_title("Feature Importance")
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
fi_path = output_dir / "feature_importance.png"
fig.savefig(fi_path, dpi=300, bbox_inches="tight")
print(f"已儲存視覺化圖片: {fi_path}")
plt.show()