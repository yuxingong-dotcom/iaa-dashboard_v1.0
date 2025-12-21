import pandas as pd
import streamlit as st
import re

# ================= 辅助逻辑函数 =================

def identify_network(placement):
    """识别广告网络"""
    if pd.isnull(placement): return None
    placement = str(placement)
    if '/90851098,21819256933/34065401/' in placement: return 'Rek'
    elif '/60257202,21819256933/' in placement: return 'A4G'
    elif 'ca-mb-app-pub-2385332075335369' in placement: return 'GAM'
    elif '/75894840,21819256933/p20404/a78007/' in placement: return 'Premium'
    elif '/22904705113,21819256933/20404:77448/' in placement: return 'Premium'
    return 'Other'

def extract_version(row):
    """
    提取轮替版本号 (逻辑已更新)
    """
    network = row['轮替网络']
    placement = str(row['Network Placement'])
    if pd.isnull(network): return 'Unknown'
    
    try:
        # --- 【修改点 1】Rek: 统一逻辑，无特例 ---
        if network == 'Rek':
            # 逻辑：找到 /34065401/，取其后面的部分，然后截取到第一个下划线 _ 为止
            marker = '/34065401/'
            if marker in placement:
                # split后取[-1]拿到marker后面的部分，再split('_')[0]拿第一个下划线前的部分
                return placement.split(marker)[-1].split('_')[0]
            # 如果没找到 marker，尝试直接取最后一个斜杠后的内容再截断
            if '/' in placement:
                return placement.split('/')[-1].split('_')[0]
            return placement 
            
        # --- 【修改点 2】A4G: 强制取最后一段 ---
        elif network == 'A4G':
            # 逻辑：不管前面有多长，直接以 "/" 为分隔符，取最后一段
            # 针对 /60257202,21819256933/584859 -> 结果为 584859
            if '/' in placement:
                return placement.split('/')[-1]
            return placement
            
        # --- 其他网络保持不变 ---
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
    """修正 eCPM 价格 (逻辑保持不变)"""
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
        
        return df

    except Exception as e:
        st.error(f"数据处理发生错误: {e}")
        return None