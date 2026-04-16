import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. 讀取資料
df = pd.read_csv('0050_cleaned_data_5years.csv')

# 2. 定義特徵 (X) 與 目標 (y)
X = df.drop(columns=['日期', '漲跌價差', '明日收盤價', 'Target'])
y = df['Target']

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