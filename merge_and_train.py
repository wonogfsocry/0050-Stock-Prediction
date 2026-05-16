import pandas as pd
import glob
import os

# =========================================================================
# 1. 建立通用的「讀取與資料清洗」函數
# =========================================================================
def load_and_clean_stock(stock_code):
    """讀取指定股票代號的所有 CSV 並進行基礎清洗與還原股價計算"""
    # ★ 關鍵修改：只抓取檔名包含該股票代號的檔案，避免 0050 與 2330 混在一起
    file_pattern = os.path.join('data', f'*{stock_code}*.csv')
    file_list = glob.glob(file_pattern)
    
    print(f"找到 {len(file_list)} 個 {stock_code} 的 CSV 檔案，開始合併清洗...")
    
    if not file_list:
        raise ValueError(f"找不到任何 {stock_code} 的檔案，請確認 data 資料夾！")
        
    all_dataframes = []
    for file in file_list:
        try:
            temp_df = pd.read_csv(file, skiprows=1, encoding='utf-8')
        except UnicodeDecodeError:
            temp_df = pd.read_csv(file, skiprows=1, encoding='big5')
        all_dataframes.append(temp_df)
        
    df = pd.concat(all_dataframes, ignore_index=True)
    
    # 基礎清洗
    df = df[df['日期'].str.contains(r'\d{3}/\d{2}/\d{2}', na=False)].copy()
    cols_to_drop = [col for col in df.columns if 'Unnamed' in col or '註記' in col]
    df = df.drop(columns=cols_to_drop)
    
    numeric_cols = ['成交股數', '成交金額', '開盤價', '最高價', '最低價', '收盤價', '成交筆數']
    for col in numeric_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
            
    df['漲跌價差'] = df['漲跌價差'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=False)
    df['漲跌價差'] = pd.to_numeric(df['漲跌價差'], errors='coerce').fillna(0.0)
    
    # 轉換日期格式
    def convert_roc_date(date_str):
        parts = str(date_str).split('/')
        if len(parts) == 3:
            year = int(parts[0]) + 1911
            return f"{year}-{parts[1]}-{parts[2]}"
        return date_str
        
    df['日期'] = df['日期'].apply(convert_roc_date)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    
    # 計算還原股價 (解決除息跳空)
    df['adj_close'] = df['收盤價'].iloc[-1] - df['漲跌價差'].iloc[::-1].cumsum().iloc[::-1].shift(-1).fillna(0)
    adj_ratio = df['adj_close'] / df['收盤價']
    df['adj_open'] = df['開盤價'] * adj_ratio
    df['adj_high'] = df['最高價'] * adj_ratio
    df['adj_low'] = df['最低價'] * adj_ratio
    
    # 計算該股票的基礎相對特徵
    df['前1日還原收盤'] = df['adj_close'].shift(1)
    df['前1日成交股數'] = df['成交股數'].shift(1)
    df['日報酬率'] = (df['adj_close'] - df['前1日還原收盤']) / df['前1日還原收盤']
    df['成交量變化率'] = (df['成交股數'] - df['前1日成交股數']) / df['前1日成交股數']
    
    return df

# =========================================================================
# 2. 分別讀取 0050 與 2330
# =========================================================================
df_0050 = load_and_clean_stock('0050')
df_2330 = load_and_clean_stock('2330')

# =========================================================================
# 3. 針對 0050 進行進階特徵工程 (主要目標預測標的)
# =========================================================================
TRANSACTION_COST = 0.002  

df_0050['明日報酬率'] = (df_0050['adj_close'].shift(-1) - df_0050['adj_close']) / df_0050['adj_close']
df_0050['Target'] = (df_0050['明日報酬率'] > TRANSACTION_COST).astype(int)

df_0050['MA5'] = df_0050['adj_close'].rolling(window=5).mean()
df_0050['前1日日報酬率'] = df_0050['日報酬率'].shift(1)
df_0050['前2日日報酬率'] = df_0050['日報酬率'].shift(2)
df_0050['MA5乖離率'] = (df_0050['adj_close'] - df_0050['MA5']) / df_0050['MA5']
df_0050['K線實體'] = (df_0050['adj_close'] - df_0050['adj_open']) / df_0050['adj_open']
df_0050['振幅'] = (df_0050['adj_high'] - df_0050['adj_low']) / df_0050['前1日還原收盤']

price_delta = df_0050['adj_close'].diff()
gain = price_delta.clip(lower=0)
loss = -price_delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df_0050['RSI14'] = 100 - (100 / (1 + rs))

df_0050['5日波動率'] = df_0050['日報酬率'].rolling(window=5).std()
df_0050['MA20'] = df_0050['adj_close'].rolling(window=20).mean()
df_0050['MA20乖離率'] = (df_0050['adj_close'] - df_0050['MA20']) / df_0050['MA20']
df_0050['均線多空'] = (df_0050['MA5'] > df_0050['MA20']).astype(int)

exp1 = df_0050['adj_close'].ewm(span=12, adjust=False).mean()
exp2 = df_0050['adj_close'].ewm(span=26, adjust=False).mean()
df_0050['MACD'] = exp1 - exp2
df_0050['MACD_Signal'] = df_0050['MACD'].ewm(span=9, adjust=False).mean()
df_0050['MACD柱狀圖'] = df_0050['MACD'] - df_0050['MACD_Signal']

df_0050['MA20_std'] = df_0050['adj_close'].rolling(window=20).std()
df_0050['BB_Upper'] = df_0050['MA20'] + (df_0050['MA20_std'] * 2)
df_0050['BB_Lower'] = df_0050['MA20'] - (df_0050['MA20_std'] * 2)
df_0050['布林位置'] = (df_0050['adj_close'] - df_0050['BB_Lower']) / (df_0050['BB_Upper'] - df_0050['BB_Lower'])

df_0050['跳空幅度'] = (df_0050['adj_open'] - df_0050['前1日還原收盤']) / df_0050['前1日還原收盤']

# =========================================================================
# 4. 提取台積電 (2330) 專屬特徵，並與 0050 進行 DataFrame 合併
# =========================================================================
# 只挑選預測有幫助的欄位
df_2330_features = df_2330[['日期', '日報酬率', '成交量變化率']].copy()
df_2330_features = df_2330_features.rename(columns={
    '日報酬率': 'tsmc_daily_return',
    '成交量變化率': 'tsmc_vol_change'
})

# 以 0050 的交易日期為基準，把 2330 的特徵「左側合併」進來
df_merged = pd.merge(df_0050, df_2330_features, on='日期', how='left')

# ★ 新增聯動特徵：大盤背離度 (台積電報酬率 - 0050報酬率)
# 金融意義：捕捉「拉積盤 (只拉台積電，其他中小型股全倒)」或「殺積盤」。這種籌碼不均的狀況常預示隔日變盤。
df_merged['tsmc_0050_spread'] = df_merged['tsmc_daily_return'] - df_merged['日報酬率']

# =========================================================================
# 5. 星期幾特徵編碼與最終資料輸出
# =========================================================================
df_merged['day_of_week'] = df_merged['日期'].dt.dayofweek
df_merged = pd.get_dummies(df_merged, columns=['day_of_week'], prefix='day')

for col in [f'day_{i}' for i in range(5)]:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col].astype(int)

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
df_merged = df_merged.rename(columns=column_mapping)

# 挑選最終特徵 (將 TSMC 的特徵加入列表)
features_to_keep = [
    'date', 'daily_return', 'daily_return_lag1', 'daily_return_lag2',
    'ma5_bias', 'ma20_bias', 'ma_trend', 'macd_hist', 'bb_position', 'gap_pct', 
    'kline_body', 'amplitude', 'vol_change', 'rsi_14', 'volatility_5d',
    'tsmc_daily_return', 'tsmc_vol_change', 'tsmc_0050_spread', # ★ 納入模型訓練
    'target'
]
day_cols = [col for col in df_merged.columns if col.startswith('day_')]
features_to_keep.extend(day_cols)

# 移除空值 NaN
df_final = df_merged[features_to_keep].dropna().copy()

print("\n資料準備完畢！")
print(f"最終可用於訓練的資料筆數：{len(df_final)} 筆")

print("\n真實漲跌天數分佈 (目標標籤)：")
print(df_final['target'].value_counts(normalize=True))

df_final.to_csv('0050_cleaned_data_5years.csv', index=False, encoding='utf-8-sig')
print("\n檔案已成功儲存為 0050_cleaned_data_5years.csv (已成功納入台積電特徵)")