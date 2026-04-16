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

# 建立預測目標 (Target)
df['明日收盤價'] = df['收盤價'].shift(-1)
df['Target'] = (df['明日收盤價'] > df['收盤價']).astype(int)

# 建立歷史特徵
df['前1日收盤價'] = df['收盤價'].shift(1)
df['前2日收盤價'] = df['收盤價'].shift(2)
df['MA5'] = df['收盤價'].rolling(window=5).mean()

# 捨棄因為 shift 和 rolling 產生的 NaN 值 (最前面幾天與最後一天)
df_final = df.dropna().copy()

print("資料準備完畢！")
print(f"最終可用於訓練的資料筆數：{len(df_final)} 筆")
print("\n前五筆資料檢查：")
print(df_final[['日期', '收盤價', '前1日收盤價', 'MA5', '明日收盤價', 'Target']].head())

# 將清理好的資料存成一個新的 CSV，以後就不用每次都重新合併了
df_final.to_csv('0050_cleaned_data_5years.csv', index=False, encoding='utf-8-sig')