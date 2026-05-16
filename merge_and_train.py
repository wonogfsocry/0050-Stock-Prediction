import pandas as pd
import glob
import os

# 1. 自動抓取 data 資料夾下所有的 csv 檔案路徑
file_pattern = os.path.join('data', '*.csv')
file_list = glob.glob(file_pattern)

print(f"總共找到 {len(file_list)} 個 CSV 檔案，準備開始合併...")

# 2. 建立一個空的 list 來存放每個月份的 DataFrame
all_dataframes = []

for file in file_list:
    try:
        # 逐一讀取，跳過第一行的標題
        temp_df = pd.read_csv(file, skiprows=1, encoding='utf-8')
    except UnicodeDecodeError:
        temp_df = pd.read_csv(file, skiprows=1, encoding='big5')
    
    all_dataframes.append(temp_df)

# 3. 將這些資料合併成一個巨大的表格
df = pd.concat(all_dataframes, ignore_index=True)

print(f"合併完成！原始資料共有 {len(df)} 筆。開始進行資料清洗...")

# ====================
# 以下是統一的資料清洗流程 (合併後再一起洗，效率最高)
# ====================

# 移除檔案結尾的說明文字 (只保留日期格式為 xxx/xx/xx 的列)
df = df[df['日期'].str.contains(r'\d{3}/\d{2}/\d{2}', na=False)].copy()

# 移除不必要的空白欄位或註記
cols_to_drop = [col for col in df.columns if 'Unnamed' in col or '註記' in col]
df = df.drop(columns=cols_to_drop)

# 清除千位數逗號並轉換為浮點數
numeric_cols = ['成交股數', '成交金額', '開盤價', '最高價', '最低價', '收盤價', '成交筆數']
for col in numeric_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '').astype(float)

# ★ 新增優化：單獨處理「漲跌價差」，移除正號與逗號並轉為浮點數，以供還原股價計算 ★
df['漲跌價差'] = df['漲跌價差'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=False)
df['漲跌價差'] = pd.to_numeric(df['漲跌價差'], errors='coerce').fillna(0.0)

# 處理民國年日期轉換為西元年
def convert_roc_date(date_str):
    parts = str(date_str).split('/')
    if len(parts) == 3:
        year = int(parts[0]) + 1911
        return f"{year}-{parts[1]}-{parts[2]}"
    return date_str

df['日期'] = df['日期'].apply(convert_roc_date)
df['日期'] = pd.to_datetime(df['日期'])

# ★ 關鍵步驟：確保資料是按照時間「由舊到新」嚴格排序 ★
df = df.sort_values('日期').reset_index(drop=True)

# =========================================================================
# ★ 核心優化：自主重建「還原股價 (Adjusted Prices)」以解決除權息跳空陷阱 ★
# =========================================================================
# 以最後一天(最新)的原始收盤價為基準，逆向利用「漲跌價差」回推歷史還原收盤價
df['adj_close'] = df['收盤價'].iloc[-1] - df['漲跌價差'].iloc[::-1].cumsum().iloc[::-1].shift(-1).fillna(0)

# 利用收盤價的還原比例，同比例縮放當日的開盤、最高、最低價，完美保留 K 線實體與振幅型態
adj_ratio = df['adj_close'] / df['收盤價']
df['adj_open'] = df['開盤價'] * adj_ratio
df['adj_high'] = df['最高價'] * adj_ratio
df['adj_low'] = df['最低價'] * adj_ratio

# ====================
# 特徵工程 (特徵改由「還原股價變數」進行計算)
# ====================

# 1. 建立預測目標 (Target) - 改用還原收盤價對比，避開除息日被誤標記為下跌(0)的錯誤
df['Target'] = (df['adj_close'].shift(-1) > df['adj_close']).astype(int)

# 2. 建立歷史基礎指標 (供後續計算使用)
df['MA5'] = df['adj_close'].rolling(window=5).mean()
df['前1日還原收盤'] = df['adj_close'].shift(1)
df['前1日成交股數'] = df['成交股數'].shift(1)

# ★ 3. 將絕對數值轉換為「相對特徵」 ★
# 日報酬率 (昨日到今日的真實漲跌幅)
df['日報酬率'] = (df['adj_close'] - df['前1日還原收盤']) / df['前1日還原收盤']

# 歷史平移特徵：加入前天與大前天的日報酬率
df['前1日日報酬率'] = df['日報酬率'].shift(1)
df['前2日日報酬率'] = df['日報酬率'].shift(2)

# MA5 乖離率
df['MA5乖離率'] = (df['adj_close'] - df['MA5']) / df['MA5']

# K線實體長度 (使用還原後的收盤與開盤)
df['K線實體'] = (df['adj_close'] - df['adj_open']) / df['adj_open']

# 振幅 (使用還原後的最高與最低)
df['振幅'] = (df['adj_high'] - df['adj_low']) / df['前1日還原收盤']

# 成交量變化率
df['成交量變化率'] = (df['成交股數'] - df['前1日成交股數']) / df['前1日成交股數']

# RSI(14)：動能與超買超賣指標
price_delta = df['adj_close'].diff()
gain = price_delta.clip(lower=0)
loss = -price_delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI14'] = 100 - (100 / (1 + rs))

# 5日波動率：過去 5 天日報酬率標準差
df['5日波動率'] = df['日報酬率'].rolling(window=5).std()

# ==========================================
# 進階技術指標與市場特徵 (XGBoost 優化版)
# ==========================================

# 中長期趨勢 (MA20 月線)
df['MA20'] = df['adj_close'].rolling(window=20).mean()
df['MA20乖離率'] = (df['adj_close'] - df['MA20']) / df['MA20']
df['均線多空'] = (df['MA5'] > df['MA20']).astype(int)

# MACD 指補 (12日與26日 EMA)
exp1 = df['adj_close'].ewm(span=12, adjust=False).mean()
exp2 = df['adj_close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD柱狀圖'] = df['MACD'] - df['MACD_Signal']

# 布林通道 %B (Bollinger Bands)
df['MA20_std'] = df['adj_close'].rolling(window=20).std()
df['BB_Upper'] = df['MA20'] + (df['MA20_std'] * 2)
df['BB_Lower'] = df['MA20'] - (df['MA20_std'] * 2)
df['布林位置'] = (df['adj_close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

# 跳空缺口 (今日還原開盤價相對於昨日還原收盤價的跳空幅度)
df['跳空幅度'] = (df['adj_open'] - df['前1日還原收盤']) / df['前1日還原收盤']

# 星期幾特徵 (0=週一, ..., 4=週五)
df['day_of_week'] = df['日期'].dt.dayofweek

# 將星期幾進行獨熱編碼 (One-Hot Encoding)
df = pd.get_dummies(df, columns=['day_of_week'], prefix='day')

# 確保產生出來的 True/False 轉換為 1/0 的整數格式
for col in [f'day_{i}' for i in range(5)]:
    if col in df.columns:
        df[col] = df[col].astype(int)

# 4. 重新命名欄位
column_mapping = {
    '日期': 'date',
    '日報酬率': 'daily_return',
    '前1日日報酬率': 'daily_return_lag1',
    '前2日日報酬率': 'daily_return_lag2',
    'MA5乖離率': 'ma5_bias',
    'MA20乖離率': 'ma20_bias',
    '均線多空': 'ma_trend',
    'MACD柱狀圖': 'macd_hist',
    '布林位置': 'bb_position',
    '跳空幅度': 'gap_pct',
    'K線實體': 'kline_body',
    '振幅': 'amplitude',
    '成交量變化率': 'vol_change',
    'RSI14': 'rsi_14',
    '5日波動率': 'volatility_5d',
    'Target': 'target',
}
df = df.rename(columns=column_mapping)

# 5. 挑選最終特徵 (動態將 day_0 到 day_4 一併加入)
features_to_keep = [
    'date', 'daily_return', 'daily_return_lag1', 'daily_return_lag2',
    'ma5_bias', 'ma20_bias', 'ma_trend', 'macd_hist', 'bb_position', 'gap_pct', 
    'kline_body', 'amplitude', 'vol_change', 'rsi_14', 'volatility_5d', 'target'
]
day_cols = [col for col in df.columns if col.startswith('day_')]
features_to_keep.extend(day_cols)

# 移除空值 NaN (因為計算中長期指標，大約會自動捨去最前面的 26 天)
df_final = df[features_to_keep].dropna().copy()

print("\n資料準備完畢！")
print(f"最終可用於訓練的資料筆數：{len(df_final)} 筆")

# 觀察真實的漲跌不平衡比例
print("\n真實漲跌天數分佈 (目標標籤)：")
print(df_final['target'].value_counts(normalize=True))

# 存檔
df_final.to_csv('0050_cleaned_data_5years.csv', index=False, encoding='utf-8-sig')
print("\n檔案已成功儲存為 0050_cleaned_data_5years.csv (已完成除權息還原修正)")