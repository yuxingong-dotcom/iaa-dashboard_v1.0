import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 【关键点】引入我们在 data_processor.py 里写好的函数
from data_processor import process_raw_data

# ================= 0. 页面配置与 CSS 美化 =================
st.set_page_config(
    page_title="IAA 商业化运营看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .sticky-nav {
            position: sticky;
            top: 2.875rem;
            z-index: 999;
            background-color: var(--background-color); 
            backdrop-filter: blur(10px);
            padding: 15px 20px;
            border-bottom: 1px solid rgba(150, 150, 150, 0.2);
            margin-bottom: 20px;
            color: var(--text-color);
        }
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(150, 150, 150, 0.2);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        div[data-testid="stMetric"] > div { color: var(--text-color); }
        .st-emotion-cache-1629p8f h1 a, h2 a, h3 a { display: none; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 IAA 广告变现：策略诊断与轮替分析")

# ================= 1. 侧边栏与数据加载 =================

st.sidebar.header("📁 配置与筛选")
uploaded_file = st.sidebar.file_uploader("上传 Applovin 报表 (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded_file is None:
    st.info("👋 请先在左侧上传数据文件。")
    st.stop()

# 【关键点】直接调用 data_processor 里的函数
raw_df = process_raw_data(uploaded_file)
if raw_df is None: st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("📌 基础筛选 (必选)")

# 网络筛选
all_networks = sorted([x for x in raw_df['轮替网络'].unique() if x is not None])
selected_network = st.sidebar.selectbox(
    "网络 (Network):", options=all_networks, index=None, placeholder="请选择..."
)

# 广告类型筛选
all_adtypes = sorted(raw_df['Ad Type'].astype(str).unique().tolist())
selected_adtype = st.sidebar.selectbox(
    "广告类型 (Ad Type):", options=all_adtypes, index=None, placeholder="请选择..."
)

st.sidebar.caption("🔧 维度筛选 (多选)")
# 平台筛选
all_platforms = sorted(raw_df['Platform'].astype(str).unique().tolist())
selected_platforms = st.sidebar.multiselect("平台 (Platform):", options=all_platforms, default=all_platforms)

# 国家筛选
all_countries = sorted(raw_df['Country'].unique().astype(str).tolist())
selected_countries = st.sidebar.multiselect(
    "国家 (Country):", options=all_countries, default=all_countries[:5] if len(all_countries) > 5 else all_countries
)

if not selected_network or not selected_adtype:
    st.warning("👈 请在左侧侧边栏手动选择 **网络** 和 **广告类型** 以开始分析。")
    st.stop()

# 侧边栏筛选逻辑
mask_base = (
    (raw_df['轮替网络'] == selected_network) & 
    (raw_df['Ad Type'].astype(str) == selected_adtype) & 
    (raw_df['Country'].isin(selected_countries)) & 
    (raw_df['Platform'].isin(selected_platforms))
)
df_base_filtered = raw_df[mask_base].copy()

# ================= 2. 吸顶导航栏 =================

header_container = st.container()
with header_container:
    st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns([1, 2])
    with col_nav1:
        min_date, max_date = raw_df['Day'].min().date(), raw_df['Day'].max().date()
        date_range = st.date_input("📅 选择时间段:", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="top_date_input")
        start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
    with col_nav2:
        analysis_mode = st.radio("📍 选择分析模块:", ["1. 轮替效果分析 (Rotation)", "2. 瀑布流策略诊断 (Strategy)"], horizontal=True, key="top_nav_radio")
    st.markdown('</div>', unsafe_allow_html=True)

# 日期筛选逻辑
mask_date = (df_base_filtered['Day'].dt.date >= start_date) & (df_base_filtered['Day'].dt.date <= end_date)
df_filtered = df_base_filtered[mask_date].copy()

# ================= 3. KPI 与 预览 =================

if df_filtered.empty:
    st.error("⚠️ 当前筛选条件下无数据。")
    st.stop()

# 计算 KPI
kpi_rev = df_filtered['Revenue'].sum()
kpi_imp = df_filtered['Impressions'].sum()
kpi_atm = df_filtered['Attempts'].sum()
kpi_res = df_filtered['Responses'].sum()
kpi_ecpm = (kpi_rev / kpi_imp * 1000) if kpi_imp > 0 else 0
kpi_fill = (kpi_res / kpi_atm * 100) if kpi_atm > 0 else 0

st.markdown("#### 📊 核心指标 (Key Metrics)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("💰 总收入", f"${kpi_rev:,.2f}")
k2.metric("📉 平均 eCPM", f"${kpi_ecpm:,.2f}")
k3.metric("👁️ 总展示", f"{kpi_imp:,.0f}")
k4.metric("✅ 加权填充率", f"{kpi_fill:.2f}%")
st.markdown("---")

with st.expander("📥 数据明细导出 (Data Export & Preview)", expanded=False):
    c1, c2 = st.columns([3, 1])
    c1.dataframe(df_filtered.head(100), use_container_width=True)
    c2.download_button("📥 下载 CSV", df_filtered.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")

# ================= 4. 辅助函数 =================

def get_session_index(key_name, available_options):
    if key_name in st.session_state and st.session_state[key_name] in available_options:
        return available_options.index(st.session_state[key_name])
    return 0 if available_options else None

# ================= 5. 分析模块内容 =================

# --- MODULE 1: 轮替分析 ---
if analysis_mode == "1. 轮替效果分析 (Rotation)":
    st.subheader(f"🔄 轮替版本生命周期监测")
    col1, col2 = st.columns(2)
    unique_apps = sorted(df_filtered['Application'].unique().tolist())

    with col1:
        app_idx = get_session_index('s_app_p1', unique_apps)
        sel_app = st.selectbox("选择 App:", unique_apps, index=app_idx, key='s_app_p1')
        thresh = st.number_input("过滤展示量小于:", value=50, step=10)

    is_gam = (selected_network == 'GAM')
    chart_data = pd.DataFrame()
    sel_ecpm = None

    with col2:
        if sel_app:
            app_data = df_filtered[(df_filtered['Application'] == sel_app) & (df_filtered['Impressions'] > thresh)]
            if is_gam:
                st.info("ℹ️ GAM 模式：展示全量价格层趋势。")
                chart_data = app_data
            else:
                av_ecpms = sorted(app_data['eCPM_修正后'].unique())
                if av_ecpms:
                    sel_ecpm = st.selectbox("选择 eCPM 层:", av_ecpms)
                    chart_data = app_data[app_data['eCPM_修正后'] == sel_ecpm]
                else:
                    st.warning("无数据")

    # 画图
    fig = go.Figure()
    if not chart_data.empty and (is_gam or sel_ecpm is not None):
        agg = chart_data.groupby(['Day', '轮替版本']).agg({'Attempts':'sum', 'Responses':'sum'}).reset_index()
        agg['Fill Rate'] = agg.apply(lambda x: (x['Responses']/x['Attempts']*100) if x['Attempts']>0 else 0, axis=1)
        agg = agg.sort_values('Day')
        agg['Date_Str'] = agg['Day'].dt.strftime('%Y-%m-%d')
        
        if not agg.empty:
            title = f'<b>{sel_app}</b>' + (' - GAM' if is_gam else f' - Floor: ${sel_ecpm}')
            fig = px.line(agg, x='Date_Str', y='Fill Rate', color='轮替版本', markers=True, title=title)
            
    fig.update_layout(yaxis_title="Fill Rate (%)", xaxis_title="Date", hovermode="x unified", height=550, legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)

# --- MODULE 2: 策略诊断 ---
elif analysis_mode == "2. 瀑布流策略诊断 (Strategy)":
    st.subheader(f"📈 瀑布流分层诊断")
    
    # 聚合
    agg = df_filtered.groupby(['Application', 'eCPM_修正后']).agg({'Attempts':'sum', 'Responses':'sum', 'Revenue':'sum'}).reset_index()
    agg['Weighted_Fill_Rate'] = agg.apply(lambda x: (x['Responses']/x['Attempts']*100) if x['Attempts']>0 else 0, axis=1)
    agg['Rev_Share'] = (agg['Revenue'] / agg.groupby('Application')['Revenue'].transform('sum') * 100).fillna(0)
    
    # Macro View
    st.markdown("#### 1. 大盘分布")
    plot_data = agg[agg['Weighted_Fill_Rate'] > 0]
    if not plot_data.empty:
        fig_macro = px.scatter(
            plot_data, x="eCPM_修正后", y="Weighted_Fill_Rate", size="Revenue", color="Application",
            log_x=True, log_y=True, opacity=0.7, size_max=60, title=f"eCPM vs Fill Rate ({selected_network})"
        )
        fig_macro.add_hline(y=1, line_dash="dot", line_color="red")
        fig_macro.update_layout(height=600)
        st.plotly_chart(fig_macro, use_container_width=True)
    
    st.divider()

    # Micro View
    st.markdown("#### 2. 单 App 深度诊断")
    u_apps = sorted(agg['Application'].unique().tolist())
    idx_p2 = get_session_index('s_app_p2', u_apps)
    sel_app_p2 = st.selectbox("选择 App:", u_apps, index=idx_p2, key='s_app_p2')
    
    fig_micro = make_subplots(specs=[[{"secondary_y": True}]])
    if sel_app_p2:
        d_app = agg[agg['Application'] == sel_app_p2].sort_values('eCPM_修正后')
        if not d_app.empty:
            fig_micro.add_trace(go.Bar(
                x=d_app['eCPM_修正后'].astype(str), y=d_app['Weighted_Fill_Rate'], name="Fill Rate",
                marker_color='rgba(55, 128, 191, 0.7)', customdata=d_app['Revenue'],
                hovertemplate='<b>Fill: %{y:.2f}%</b><br>Rev: $%{customdata:,.2f}<extra></extra>'
            ), secondary_y=False)
            fig_micro.add_trace(go.Scatter(
                x=d_app['eCPM_修正后'].astype(str), y=d_app['Rev_Share'], name="Rev Share",
                marker_color='crimson', mode='lines+markers', customdata=d_app['Revenue'],
                hovertemplate='<b>Share: %{y:.2f}%</b><br>Rev: $%{customdata:,.2f}<extra></extra>'
            ), secondary_y=True)
            fig_micro.update_layout(title=f"<b>{sel_app_p2} Waterfall</b>")
            
    fig_micro.update_layout(xaxis_title="eCPM Layers", height=600, legend=dict(orientation="h", y=1.1), hovermode="x unified")
    st.plotly_chart(fig_micro, use_container_width=True)