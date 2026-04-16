import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 讀取資料
df = pd.read_csv('0050_cleaned_data_5years.csv')

# 2. 定義特徵 (X) 與 目標 (y)
feature_columns = [
	'daily_return_lag1',
	'daily_return_lag2',
	'ma5_bias',
	'kline_body',
	'amplitude',
	'vol_change',
	'rsi_14',
	'volatility_5d',
]
X = df[feature_columns]
y = df['target']

# 3. 切割資料 (80% 訓練，20% 測試)
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
)

# 4. 初始化 Random Forest 模型
rf = RandomForestClassifier(
	n_estimators=300,
	max_depth=8,
	min_samples_split=15,
	min_samples_leaf=5,
	random_state=42,
	n_jobs=-1,
)

# 訓練模型
rf.fit(X_train, y_train)

# 5. 預測並計算準確率
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest 模型預測準確率: {accuracy * 100:.2f}%")
print("\n詳細分類報告:")
print(classification_report(y_test, y_pred))

# 建立視覺化輸出資料夾
output_dir = Path('visualizations')
output_dir.mkdir(exist_ok=True)

# 6. 視覺化：混淆矩陣
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, cmap='Blues', ax=ax)
ax.set_title('Random Forest Confusion Matrix')
plt.tight_layout()
cm_path = output_dir / 'random_forest_confusion_matrix.png'
fig.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"已儲存視覺化圖片: {cm_path}")
plt.show()

# 7. 視覺化：特徵重要度
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(11, 6))
feature_importance.plot(kind='bar', ax=ax)
ax.set_title('Random Forest Feature Importance')
ax.set_xlabel('Features')
ax.set_ylabel('Importance')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
fi_path = output_dir / 'random_forest_feature_importance.png'
fig.savefig(fi_path, dpi=300, bbox_inches='tight')
print(f"已儲存視覺化圖片: {fi_path}")
plt.show()
