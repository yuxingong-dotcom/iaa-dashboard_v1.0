import pandas as pd
import streamlit as st
import re

# ================= 辅助逻辑函数 =================

def identify_network(placement):
    """识别广告网络"""
    if pd.isnull(placement): return None
    placement = str(placement)
    if '/90851098' in placement: return 'Rek'
    elif '/60257202' in placement: return 'A4G'
    elif 'ca-mb-app-pub' in placement: return 'GAM'
    elif '/75894840' in placement: return 'Premium'
    elif '/22904705113' in placement: return 'Premium'
    return 'Other'

def extract_version(row):
    """提取轮替版本号"""
    network = row['轮替网络']
    placement = str(row['Network Placement'])
    if pd.isnull(network): return 'Unknown'
    
    try:
        # Rek: 截取 /34065401/ 后面的内容，直到遇到第一个下划线 _
        if network == 'Rek':
            marker = '/34065401/'
            if marker in placement:
                return placement.split(marker)[-1].split('_')[0]
            if '/' in placement:
                return placement.split('/')[-1].split('_')[0]
            return placement 
            
        # A4G: 强制取最后一段数字 ID
        elif network == 'A4G':
            if '/' in placement:
                return placement.split('/')[-1]
            return placement
            
        # 其他网络
        elif network == 'GAM':
            return placement.split('/')[-1]
            
        elif network == 'Premium':
            if '/75894840,21819256933/p20404/a78007/' in placement:
                return placement.split('/')[-1].split('-')[0]
            if '/22904705113,21819256933/20404:77448/' in placement:
                parts = placement.split(':')
                if len(parts) >= 2: return parts[-2]
            return placement
            
    except: return 'ParseError'
    return 'Other'

def correct_ecpm(row):
    """修正 eCPM 价格"""
    network = row['轮替网络']
    placement = str(row['Network Placement'])
    original_ecpm = row['eCPM']
    try:
        if network == 'Rek':
            return float(placement.split('_')[-1])
        elif network == 'GAM':
            match = re.search(r'[-](?:F)?(\d+)$', placement)
            if match: return float(match.group(1)) / 100
            return original_ecpm
        elif network == 'Premium':
            if '/75894840,21819256933/p20404/a78007/' in placement:
                return float(placement.split('-')[-1]) / 100
            if '/22904705113,21819256933/20404:77448/' in placement:
                return float(placement.split(':')[-1])
    except: return original_ecpm
    return original_ecpm

def assign_ecpm_range(row):
    """
    【新功能】根据 eCPM 和 广告类型 划分价格区间
    """
    ecpm = row['eCPM_修正后']
    ad_type = str(row['Ad Type']).upper() # 转大写方便匹配
    
    # 1. BANNER 和 MREC 逻辑
    if 'BANNER' in ad_type or 'MREC' in ad_type:
        if 0 <= ecpm < 0.05: return 'B01:0-0.05'
        if 0.05 <= ecpm < 0.10: return 'B02:0.05-0.10'
        if 0.10 <= ecpm < 0.20: return 'B03:0.10-0.20'
        if 0.20 <= ecpm < 0.30: return 'B04:0.20-0.30'
        if 0.30 <= ecpm < 0.50: return 'B05:0.30-0.50'
        if 0.50 <= ecpm < 0.80: return 'B06:0.50-0.80'
        if 0.80 <= ecpm < 1.20: return 'B07:0.80-1.20'
        if 1.20 <= ecpm < 2.00: return 'B08:1.20-2.00'
        if 2.00 <= ecpm < 4.00: return 'B09:2.00-4.00'
        if 4.00 <= ecpm: return 'B10:4.00-10.00+' # 包含所有大于4的
        return 'B_Other'

    # 2. INTER 和 RV 逻辑 (包含 INTERSTITIAL, REWARDED, RV)
    elif 'INTER' in ad_type or 'REWARD' in ad_type or 'RV' in ad_type:
        if 0 <= ecpm < 0.30: return 'I01:0-0.30'
        if 0.30 <= ecpm < 0.60: return 'I02:0.30-0.60' 
        if 0.60 <= ecpm < 0.80: return 'I03:0.60-0.80'
        if 0.80 <= ecpm < 1.00: return 'I04:0.80-1.00'
        if 1.00 <= ecpm < 1.20: return 'I05:1.00-1.20'
        if 1.20 <= ecpm < 1.50: return 'I06:1.20-1.50'
        if 1.50 <= ecpm < 2.00: return 'I07:1.50-2.00'
        if 2.00 <= ecpm < 3.00: return 'I08:2.00-3.00'
        if 3.00 <= ecpm < 4.00: return 'I09:3.00-4.00'
        if 4.00 <= ecpm < 5.00: return 'I10:4.00-5.00'
        if 5.00 <= ecpm < 6.00: return 'I11:5.00-6.00'
        if 6.00 <= ecpm < 7.00: return 'I12:6.00-7.00'
        if 7.00 <= ecpm < 8.00: return 'I13:7.00-8.00'
        if 8.00 <= ecpm < 10.00: return 'I14:8.00-10.00'
        if 10.00 <= ecpm < 15.00: return 'I15:10.00-15.00'
        if 15.00 <= ecpm < 20.00: return 'I16:15.00-20.00'
        if 20.00 <= ecpm < 30.00: return 'I17:20.00-30.00'
        if 30.00 <= ecpm < 50.00: return 'I18:30.00-50.00'
        if 50.00 <= ecpm < 100.00: return 'I19:50.00-100.00'
        if 100.00 <= ecpm: return 'I20:100.00-500.00+'
        return 'I_Other'
    
    return 'Unknown'

# ================= 主 ETL 函数 =================

@st.cache_data
def process_raw_data(uploaded_file):
    """
    读取并清洗数据的主入口函数
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # --- 基础列名标准化 ---
        if 'Est. Revenue' in df.columns:
            df.rename(columns={'Est. Revenue': 'Revenue'}, inplace=True)
            
        # 确保数值列格式正确
        numeric_cols = ['Attempts', 'Responses', 'Impressions', 'Revenue', 'eCPM']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # --- 日期清洗 ---
        df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
        df = df.dropna(subset=['Day'])

        # --- 应用业务逻辑 ---
        df['轮替网络'] = df['Network Placement'].apply(identify_network)
        df['轮替版本'] = df.apply(extract_version, axis=1)
        df['eCPM_修正后'] = df.apply(correct_ecpm, axis=1)
        
        # 【新增】计算价格区间
        df['eCPM_Range'] = df.apply(assign_ecpm_range, axis=1)
        
        return df

    except Exception as e:
        st.error(f"数据处理发生错误: {e}")
        return None