import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, precision_score
import matplotlib.pyplot as plt

# 1. 讀取你加入進階特徵後的資料
df = pd.read_csv('0050_cleaned_data_5years.csv')

# 2. 定義特徵 (X) 與 目標 (y)
X = df.drop(columns=['date', 'target'])
y = df['target']

# 3. 初始化 XGBoost 模型 (設定防止 Overfitting 的參數)
clf = xgb.XGBClassifier(
    n_estimators=200,        # 總共種 200 棵樹
    learning_rate=0.05,      # 學習率調低，讓模型學得慢但學得精
    max_depth=4,             # 限制樹的深度，避免死記硬背
    subsample=0.8,           # 每次只隨機抽取 80% 的資料來訓練 (增加泛化能力)
    colsample_bytree=0.8,    # 每次只隨機抽取 80% 的特徵來訓練
    random_state=42,
    eval_metric='logloss'
)

# ==========================================
# 實作時間序列滾動驗證 (Time Series Split)
# ==========================================
tscv = TimeSeriesSplit(n_splits=5)  # 將資料切成 5 段進行滾動驗證
fold = 1
accuracies = []
precisions = []

print("開始進行 XGBoost 滾動驗證...\n")

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 訓練模型
    clf.fit(X_train, y_train)
    
    # 預測「機率」而不是直接預測 0 或 1
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # ★ 策略優化：提高出手門檻 (大於 60% 把握才算預測會漲)
    threshold = 0.60
    y_pred_custom = (y_pred_proba >= threshold).astype(int)
    
    # 計算效能
    acc = accuracy_score(y_test, y_pred_custom)
    prec = precision_score(y_test, y_pred_custom, zero_division=0)
    
    accuracies.append(acc)
    precisions.append(prec)
    print(f"Fold {fold}: 準確率(Accuracy)={acc*100:.2f}%, 猜漲的勝率(Precision)={prec*100:.2f}%")
    fold += 1

print("\n==========================================")
print(f"XGBoost 平均準確率: {sum(accuracies)/len(accuracies)*100:.2f}%")
print(f"XGBoost 平均看漲勝率: {sum(precisions)/len(precisions)*100:.2f}%")
print("==========================================")

# 視覺化 XGBoost 認為最重要的特徵
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(clf, ax=ax, importance_type='gain', max_num_features=10)
plt.title("XGBoost Feature Importance (Gain)")
plt.show()