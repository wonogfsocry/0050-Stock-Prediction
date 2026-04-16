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

# 3. 將這 60 個月的資料合併成一個巨大的表格
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

# 處理民國年日期轉換為西元年
def convert_roc_date(date_str):
    parts = str(date_str).split('/')
    if len(parts) == 3:
        year = int(parts[0]) + 1911
        return f"{year}-{parts[1]}-{parts[2]}"
    return date_str

df['日期'] = df['日期'].apply(convert_roc_date)
df['日期'] = pd.to_datetime(df['日期'])

# ★ 關鍵步驟：確保這 4 年的資料是按照時間「由舊到新」嚴格排序 ★
df = df.sort_values('日期').reset_index(drop=True)

# ====================
# 特徵工程 (特徵必須在排序後才能計算)
# ====================

# 1. 建立預測目標 (Target)
df['Target'] = (df['收盤價'].shift(-1) > df['收盤價']).astype(int)

# 2. 建立歷史基礎指標 (供後續計算使用)
df['MA5'] = df['收盤價'].rolling(window=5).mean()
df['前1日收盤價'] = df['收盤價'].shift(1)
df['前1日成交股數'] = df['成交股數'].shift(1)

# ★ 3. 核心優化：將絕對數值轉換為「相對特徵」 ★
# 日報酬率 (昨日到今日的漲跌幅)
df['日報酬率'] = (df['收盤價'] - df['前1日收盤價']) / df['前1日收盤價']

# MA5 乖離率 (衡量當前股價是否偏離 5日均線太多，捕捉均值回歸)
df['MA5乖離率'] = (df['收盤價'] - df['MA5']) / df['MA5']

# K線實體長度 (今日收盤與開盤的差距，代表當日多空力道)
df['K線實體'] = (df['收盤價'] - df['開盤價']) / df['開盤價']

# 振幅 (今日最高與最低的差距，衡量波動度)
df['振幅'] = (df['最高價'] - df['最低價']) / df['前1日收盤價']

# 成交量變化率 (衡量是否爆量或量縮)
df['成交量變化率'] = (df['成交股數'] - df['前1日成交股數']) / df['前1日成交股數']

# 4. 重新命名欄位
column_mapping = {
    '日期': 'date',
    '日報酬率': 'daily_return',
    'MA5乖離率': 'ma5_bias',
    'K線實體': 'kline_body',
    '振幅': 'amplitude',
    '成交量變化率': 'vol_change',
    'Target': 'target',
}
df = df.rename(columns=column_mapping)

# 5. 挑選最終特徵 (★ 丟棄原本的開高低收等絕對數值 ★)
features_to_keep = [
    'date', 'daily_return', 'ma5_bias', 'kline_body', 
    'amplitude', 'vol_change', 'target'
]
df_final = df[features_to_keep].dropna().copy()

print("資料準備完畢！")
print(f"最終可用於訓練的資料筆數：{len(df_final)} 筆")
print("\n前五筆資料檢查：")
print(df_final.head())

# 存檔
df_final.to_csv('0050_cleaned_data_5years.csv', index=False, encoding='utf-8-sig')