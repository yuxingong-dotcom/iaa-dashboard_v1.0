import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# ================= 0. 页面配置与 CSS 美化 (自适应深色模式) =================
st.set_page_config(
    page_title="IAA 商业化运营看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入 CSS：使用 var(--...) 变量，自动适配 Light/Dark 模式
st.markdown("""
    <style>
        /* 顶部导航吸顶样式 - 适配深色模式 */
        .sticky-nav {
            position: sticky;
            top: 2.875rem;
            z-index: 999;
            /* 使用系统背景变量，并增加一点透明度 */
            background-color: var(--background-color); 
            backdrop-filter: blur(10px);
            padding: 15px 20px;
            /* 边框颜色使用半透明，在深浅模式下都可见 */
            border-bottom: 1px solid rgba(150, 150, 150, 0.2);
            margin-bottom: 20px;
            /* 确保文字颜色跟随系统 */
            color: var(--text-color);
        }

        /* KPI 指标卡样式 - 适配深色模式 */
        div[data-testid="stMetric"] {
            /* 使用次级背景色 (Light模式是浅灰，Dark模式是深灰) */
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(150, 150, 150, 0.2);
            padding: 20px;
            border-radius: 10px;
            /* 阴影适配 */
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        /* 强制指标卡的文字颜色（防止部分浏览器不继承） */
        div[data-testid="stMetric"] > div {
            color: var(--text-color);
        }

        /* 隐藏 Streamlit 默认的链接锚点 */
        .st-emotion-cache-1629p8f h1 a, h2 a, h3 a {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 IAA 广告变现：策略诊断与轮替分析")

# ================= 1. 核心 ETL 逻辑 (数据清洗) =================

@st.cache_data
def process_raw_data(uploaded_file):
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

        # --- 1. 识别轮替网络 ---
        def identify_network(placement):
            if pd.isnull(placement): return None
            placement = str(placement)
            if '/90851098,21819256933/34065401/' in placement: return 'Rek'
            elif '/60257202,21819256933/' in placement: return 'A4G'
            elif 'ca-mb-app-pub-2385332075335369' in placement: return 'GAM'
            elif '/75894840,21819256933/p20404/a78007/' in placement: return 'Premium'
            elif '/22904705113,21819256933/20404:77448/' in placement: return 'Premium'
            return 'Other'

        df['轮替网络'] = df['Network Placement'].apply(identify_network)

        # --- 2. 提取轮替版本 ---
        def extract_version(row):
            network = row['轮替网络']
            placement = str(row['Network Placement'])
            if pd.isnull(network): return 'Unknown'
            try:
                if network == 'Rek':
                    match = re.search(r'/34065401/([^_]+)_', placement)
                    if match: return match.group(1)
                    if '/ios' in placement: return 'ios'
                    return placement 
                elif network == 'A4G':
                    if '/21819256933/' in placement: return placement.split('/21819256933/')[-1]
                    return placement
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

        df['轮替版本'] = df.apply(extract_version, axis=1)

        # --- 3. 修正 eCPM ---
        def correct_ecpm(row):
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

        df['eCPM_修正后'] = df.apply(correct_ecpm, axis=1)
        return df

    except Exception as e:
        st.error(f"数据处理发生错误: {e}")
        return None

# ================= 2. 侧边栏：配置区 =================

st.sidebar.header("📁 配置与筛选")
uploaded_file = st.sidebar.file_uploader("上传 Applovin 报表", type=['xlsx', 'csv'])

if uploaded_file is None:
    st.info("👋 请先在左侧上传数据文件。")
    st.stop()

raw_df = process_raw_data(uploaded_file)
if raw_df is None: st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("📌 基础筛选 (必选)")

# 1. 网络筛选
all_networks = sorted([x for x in raw_df['轮替网络'].unique() if x is not None])
selected_network = st.sidebar.selectbox(
    "1️⃣ 网络 (Network):",
    options=all_networks,
    index=None,
    placeholder="请选择一个网络..."
)

# 2. 广告类型筛选
all_adtypes = sorted(raw_df['Ad Type'].astype(str).unique().tolist())
selected_adtype = st.sidebar.selectbox(
    "2️⃣ 广告类型 (Ad Type):",
    options=all_adtypes,
    index=None,
    placeholder="请选择一种广告类型..."
)

st.sidebar.caption("🔧 维度筛选 (多选)")

# 3. 平台筛选
all_platforms = sorted(raw_df['Platform'].astype(str).unique().tolist())
selected_platforms = st.sidebar.multiselect(
    "3️⃣ 平台 (Platform):",
    options=all_platforms,
    default=all_platforms 
)

# 4. 国家筛选
all_countries = sorted(raw_df['Country'].unique().astype(str).tolist())
selected_countries = st.sidebar.multiselect(
    "4️⃣ 国家 (Country):",
    options=all_countries,
    default=all_countries[:5] if len(all_countries) > 5 else all_countries 
)

# --- 阻断逻辑 ---
if not selected_network or not selected_adtype:
    st.warning("👈 请在左侧侧边栏手动选择 **网络 (Network)** 和 **广告类型 (Ad Type)** 以开始分析。")
    st.stop()

# --- 应用侧边栏筛选 ---
mask_network = raw_df['轮替网络'] == selected_network
mask_adtype = raw_df['Ad Type'].astype(str) == selected_adtype
mask_country = raw_df['Country'].isin(selected_countries)
mask_platform = raw_df['Platform'].isin(selected_platforms)

df_base_filtered = raw_df[mask_network & mask_country & mask_platform & mask_adtype].copy()

# ================= 3. 吸顶导航栏 (自适应颜色) =================

header_container = st.container()

with header_container:
    # 注意：这里的 div 会应用上面 CSS 定义的 sticky-nav 类
    st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
    
    col_nav1, col_nav2 = st.columns([1, 2])
    
    with col_nav1:
        # 日期筛选
        min_date = raw_df['Day'].min().date()
        max_date = raw_df['Day'].max().date()
        
        date_range = st.date_input(
            "📅 选择时间段 (Date Range):",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="top_date_input"
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

    with col_nav2:
        # 模块选择
        analysis_mode = st.radio(
            "📍 选择分析模块:",
            ["1. 轮替效果分析 (Rotation)", "2. 瀑布流策略诊断 (Strategy)"],
            horizontal=True,
            key="top_nav_radio"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- 应用日期筛选 ---
mask_date = (df_base_filtered['Day'].dt.date >= start_date) & (df_base_filtered['Day'].dt.date <= end_date)
df_filtered = df_base_filtered[mask_date].copy()

# ================= 4. 上帝视角：KPI 核心指标概览 =================

if df_filtered.empty:
    st.error(f"⚠️ 当前筛选条件下无数据。\n条件: Network={selected_network}, AdType={selected_adtype}")
    st.stop()

# 计算核心指标
total_revenue = df_filtered['Revenue'].sum()
total_imp = df_filtered['Impressions'].sum()
total_attempts = df_filtered['Attempts'].sum()
total_responses = df_filtered['Responses'].sum()
avg_ecpm = (total_revenue / total_imp * 1000) if total_imp > 0 else 0
weighted_fill_rate = (total_responses / total_attempts * 100) if total_attempts > 0 else 0

st.markdown("#### 📊 核心指标 (Key Metrics)")
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("💰 总收入 (Revenue)", f"${total_revenue:,.2f}")
with k2: st.metric("📉 平均 eCPM", f"${avg_ecpm:,.2f}")
with k3: st.metric("👁️ 总展示 (Impressions)", f"{total_imp:,.0f}")
with k4: st.metric("✅ 加权填充率 (Fill Rate)", f"{weighted_fill_rate:.2f}%")

st.markdown("---")

# ================= 5. 数据预览与导出 =================

with st.expander("📥 数据明细导出 (Data Export & Preview)", expanded=False):
    c_exp1, c_exp2 = st.columns([3, 1])
    with c_exp1:
        st.caption(f"当前筛选数据行数: {len(df_filtered)}")
        cols_to_show = ['Day', 'Application', 'Platform', 'Network Placement', '轮替网络', '轮替版本', 'eCPM_修正后', 'Revenue', 'Impressions']
        st.dataframe(df_filtered[cols_to_show].head(100), use_container_width=True)
    with c_exp2:
        st.write(" ") 
        st.write(" ") 
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载 CSV",
            data=csv_data,
            file_name='processed_iaa_data.csv',
            mime='text/csv',
            use_container_width=True
        )

# ================= 6. 状态保持辅助函数 =================

def get_session_index(key_name, available_options):
    if key_name in st.session_state:
        saved_value = st.session_state[key_name]
        if saved_value in available_options:
            return available_options.index(saved_value)
    return 0 if available_options else None

# ================= 7. 分析模块内容 =================

# -------------------------------------------------------
# MODULE 1: 轮替效果分析
# -------------------------------------------------------
if analysis_mode == "1. 轮替效果分析 (Rotation)":
    st.subheader(f"🔄 轮替版本生命周期监测")
    
    col1, col2 = st.columns(2)
    unique_apps = sorted(df_filtered['Application'].unique().tolist()) if not df_filtered.empty else []

    with col1:
        app_idx = get_session_index('selected_app_name', unique_apps)
        selected_app_p1 = st.selectbox(
            "选择 App:", unique_apps, index=app_idx, key='selected_app_name'
        )
        imp_threshold = st.number_input("过滤展示量小于:", value=50, step=10)

    is_gam_mode = (selected_network == 'GAM')
    chart_data = pd.DataFrame()
    selected_ecpm_p1 = None

    with col2:
        if selected_app_p1:
            app_data_p1 = df_filtered[
                (df_filtered['Application'] == selected_app_p1) & 
                (df_filtered['Impressions'] > imp_threshold)
            ]
            
            if is_gam_mode:
                st.info("ℹ️ GAM 网络模式：展示该 App 下所有价格层趋势。")
                chart_data = app_data_p1
            else:
                available_ecpms = sorted(app_data_p1['eCPM_修正后'].unique())
                if not available_ecpms:
                    st.warning("该 App 下无数据")
                else:
                    selected_ecpm_p1 = st.selectbox("选择修正后的 eCPM 层级:", available_ecpms)
                    chart_data = app_data_p1[app_data_p1['eCPM_修正后'] == selected_ecpm_p1]
        else:
            if not unique_apps:
                st.warning("无可用 App 数据。")

    # 画图
    fig_p1 = go.Figure()

    if not chart_data.empty:
        if not is_gam_mode and selected_ecpm_p1 is None:
            pass 
        else:
            chart_data_agg = chart_data.groupby(['Day', '轮替版本']).agg({
                'Attempts': 'sum',
                'Responses': 'sum'
            }).reset_index()
            
            chart_data_agg['Fill Rate'] = chart_data_agg.apply(
                lambda x: (x['Responses'] / x['Attempts'] * 100) if x['Attempts'] > 0 else 0, axis=1
            )
            chart_data_agg = chart_data_agg.sort_values('Day')
            chart_data_agg['Date_Str'] = chart_data_agg['Day'].dt.strftime('%Y-%m-%d')

            if not chart_data_agg.empty:
                chart_title = f'<b>{selected_app_p1}</b>' + (' - GAM All Floors' if is_gam_mode else f' - Floor: ${selected_ecpm_p1}')

                fig_p1 = px.line(
                    chart_data_agg, 
                    x='Date_Str', 
                    y='Fill Rate', 
                    color='轮替版本', 
                    markers=True, 
                    title=chart_title,
                    labels={'Fill Rate': 'Fill Rate (%)', 'Date_Str': 'Date'},
                    # 【重要】移除写死的白色模板，让它自动适配深色模式
                    # template='plotly_white' <--- 已移除
                )

    # 统一图表样式 (自适应)
    fig_p1.update_layout(
        yaxis=dict(ticksuffix="%", title="Fill Rate (%)"), 
        xaxis=dict(title="Date"),
        hovermode="x unified", 
        height=550,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        # 不强制使用白色模板，Plotly 会自动检测 Streamlit 主题
        title="无数据 (No Data Available)" if chart_data.empty else None
    )
    st.plotly_chart(fig_p1, use_container_width=True)

# -------------------------------------------------------
# MODULE 2: 瀑布流策略诊断
# -------------------------------------------------------
elif analysis_mode == "2. 瀑布流策略诊断 (Strategy)":
    st.subheader(f"📈 瀑布流分层诊断")
    
    df_p2 = df_filtered

    if df_p2.empty:
         st.warning("当前筛选无数据。")
    else:
        # 聚合数据
        df_agg = df_p2.groupby(['Application', 'eCPM_修正后']).agg({
            'Attempts': 'sum',
            'Responses': 'sum',
            'Revenue': 'sum'
        }).reset_index()

        df_agg['Weighted_Fill_Rate'] = df_agg.apply(
            lambda x: (x['Responses'] / x['Attempts'] * 100) if x['Attempts'] > 0 else 0, axis=1
        )
        
        app_total_rev = df_agg.groupby('Application')['Revenue'].transform('sum')
        df_agg['Rev_Share'] = (df_agg['Revenue'] / app_total_rev * 100).fillna(0)
        df_agg = df_agg.sort_values(by=['Application', 'eCPM_修正后'])

        # --- 图表 A: 大盘气泡图 ---
        st.markdown("#### 1. 大盘分布 (Macro View)")
        plot_data = df_agg[df_agg['Weighted_Fill_Rate'] > 0]
        
        if not plot_data.empty:
            fig_macro = px.scatter(
                plot_data, 
                x="eCPM_修正后", 
                y="Weighted_Fill_Rate",
                size="Revenue", 
                color="Application",
                hover_data=["Rev_Share", "Attempts", "Responses"],
                log_x=True, 
                log_y=True, 
                title=f"<b>eCPM vs Fill Rate ({selected_network} - {selected_adtype})</b>",
                labels={'eCPM_修正后': 'Corrected eCPM ($)', 'Weighted_Fill_Rate': 'Fill Rate (%)'},
                opacity=0.7,     
                size_max=60      
            )
            fig_macro.add_hline(y=1, line_dash="dot", line_color="red")
            # 移除白色模板
            fig_macro.update_layout(height=600)
            st.plotly_chart(fig_macro, use_container_width=True)
        else:
            st.info("数据量不足以生成大盘图。")

    st.divider()

    # --- 图表 B: 单 APP 深度诊断 ---
    st.markdown("#### 2. 单 App 深度诊断 (Deep Dive)")
    # st.info("💡 提示：点击图例可隐藏/显示数据；鼠标悬停柱子可查看具体收入金额。")

    unique_apps_p2 = sorted(df_agg['Application'].unique().tolist()) if not df_p2.empty else []
    app_idx_p2 = get_session_index('selected_app_name_p2', unique_apps_p2)

    selected_app_p2 = st.selectbox(
        "选择要诊断的 App:", 
        unique_apps_p2, 
        index=app_idx_p2,
        key='selected_app_name_p2'
    )
    
    fig_micro = make_subplots(specs=[[{"secondary_y": True}]])
    
    if selected_app_p2:
        df_app = df_agg[df_agg['Application'] == selected_app_p2].sort_values('eCPM_修正后')
        
        if not df_app.empty:
            # 左轴 (Bar)
            fig_micro.add_trace(
                go.Bar(
                    x=df_app['eCPM_修正后'].astype(str), 
                    y=df_app['Weighted_Fill_Rate'],
                    name="Fill Rate (%)",
                    marker_color='rgba(55, 128, 191, 0.7)',
                    text=df_app['Weighted_Fill_Rate'].round(2).astype(str) + '%',
                    textposition='auto',
                    customdata=df_app['Revenue'], 
                    hovertemplate='<b>Fill Rate: %{y:.2f}%</b><br>Rev: $%{customdata:,.2f}<extra></extra>' 
                ),
                secondary_y=False,
            )

            # 右轴 (Line)
            fig_micro.add_trace(
                go.Scatter(
                    x=df_app['eCPM_修正后'].astype(str),
                    y=df_app['Rev_Share'],
                    name="Revenue Share (%)",
                    marker=dict(color='crimson', size=10),
                    line=dict(width=3),
                    mode='lines+markers',
                    customdata=df_app['Revenue'], 
                    hovertemplate='<b>Rev Share: %{y:.2f}%</b><br>Rev: $%{customdata:,.2f}<extra></extra>' 
                ),
                secondary_y=True,
            )
            
            fig_micro.update_layout(title=f"<b>{selected_app_p2} Waterfall Health Check</b>")

    fig_micro.update_layout(
        xaxis_title="Corrected eCPM Layers ($)",
        legend=dict(x=0, y=1.1, orientation='h'), 
        hovermode="x unified", 
        height=600,
        # 移除白色模板
    )
    fig_micro.update_yaxes(title_text="<b>Fill Rate (%)</b>", secondary_y=False)
    fig_micro.update_yaxes(title_text="<b>Revenue Share (%)</b>", secondary_y=True)
    
    st.plotly_chart(fig_micro, use_container_width=True)