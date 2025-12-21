import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 引入逻辑层
from data_processor import process_raw_data

# ================= 0. 全局页面配置 =================
st.set_page_config(
    page_title="IAA 商业化运营看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 样式
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

# ================= 辅助函数 =================
def get_session_index(key_name, available_options):
    if key_name in st.session_state and st.session_state[key_name] in available_options:
        return available_options.index(st.session_state[key_name])
    return 0 if available_options else None

# ================= 模块 1: 🏆 大盘概览 (Benchmark) =================
def render_benchmark_dashboard(raw_df):
    """
    大盘概览
    """
    st.sidebar.markdown("### 📌 基础数据池")
    
    # 1. 广告类型 (必选)
    if 'Ad Type' not in raw_df.columns:
        st.error("❌ 数据源缺少 'Ad Type' 列，无法分析。")
        return

    all_adtypes = sorted(raw_df['Ad Type'].astype(str).unique().tolist())
    selected_adtype = st.sidebar.selectbox(
        "1️⃣ 广告类型 (Ad Type):", options=all_adtypes, index=None, placeholder="请选择...", key="bench_ad"
    )

    st.sidebar.markdown("#### 🔧 维度筛选")
    
    # 2. 平台 (多选)
    if 'Platform' in raw_df.columns:
        all_platforms = sorted(raw_df['Platform'].astype(str).unique().tolist())
        selected_platforms = st.sidebar.multiselect("2️⃣ 平台:", options=all_platforms, default=all_platforms, key="bench_plat")
    else:
        selected_platforms = []
        st.sidebar.info("ℹ️ 数据无 'Platform' 维度")

    # 3. 国家 (动态检测)
    has_country = 'Country' in raw_df.columns
    mask_country = True 

    if has_country:
        all_countries = sorted(raw_df['Country'].unique().astype(str).tolist())
        selected_countries = st.sidebar.multiselect(
            "3️⃣ 国家 (留空则默认全选):", 
            options=all_countries, 
            default=[], 
            key="bench_ctry"
        )
        target_countries = selected_countries if selected_countries else all_countries
        mask_country = raw_df['Country'].isin(target_countries)
    else:
        st.sidebar.info("🌐 数据源无 'Country' 列，展示全局数据")

    if not selected_adtype:
        st.warning("👈 请在左侧选择 **广告类型** 以开始大盘分析。")
        return

    # --- 数据过滤 ---
    mask_ad = (raw_df['Ad Type'].astype(str) == selected_adtype)
    mask_plat = raw_df['Platform'].isin(selected_platforms) if 'Platform' in raw_df.columns else True
    
    mask_base = mask_ad & mask_plat & mask_country
    df_pool = raw_df[mask_base].copy()
    
    if df_pool.empty:
        st.error("⚠️ 当前筛选条件下无数据。")
        return

    # --- 吸顶导航栏 ---
    header_container = st.container()
    with header_container:
        st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
        col_nav1, col_nav2 = st.columns([1, 2])
        
        with col_nav1:
            min_date, max_date = df_pool['Day'].min().date(), df_pool['Day'].max().date()
            date_range = st.date_input("📅 日期范围:", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="bench_date")
            start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
            
        with col_nav2:
            available_apps = sorted(df_pool['Application'].unique().tolist())
            selected_apps = st.multiselect("📱 筛选 App (留空默认全选):", options=available_apps, default=[], key="bench_app_select")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 二次过滤
    target_apps = selected_apps if selected_apps else available_apps
    mask_final = (
        (df_pool['Day'].dt.date >= start_date) & 
        (df_pool['Day'].dt.date <= end_date) &
        (df_pool['Application'].isin(target_apps))
    )
    df_filtered = df_pool[mask_final].copy()

    if df_filtered.empty:
        st.error("⚠️ 当前时间或App筛选无数据。")
        return

    # --- 聚合与可视化 ---
    st.header(f"🏆 大盘概览: {selected_adtype}")
    
    agg_matrix = df_filtered.groupby(['eCPM_Range', '轮替网络']).agg({
        'Attempts': 'sum', 'Responses': 'sum', 'Revenue': 'sum'
    }).reset_index()
    
    agg_matrix['Fill Rate'] = agg_matrix.apply(
        lambda x: (x['Responses']/x['Attempts']*100) if x['Attempts']>0 else 0, axis=1
    )
    agg_matrix['RPM'] = agg_matrix.apply(
        lambda x: (x['Revenue']/x['Attempts']*1000000) if x['Attempts']>0 else 0, axis=1
    )
    
    pivot_fill = agg_matrix.pivot(index='eCPM_Range', columns='轮替网络', values='Fill Rate').sort_index()
    pivot_rpm = agg_matrix.pivot(index='eCPM_Range', columns='轮替网络', values='RPM').sort_index()

    if pivot_fill.empty:
        st.warning("数据不足以生成图表")
        return

    # === 可视化调整部分 ===

    # 1. 填充率 (Fill Rate %)
    st.subheader("1. 填充率 (Fill Rate %)")
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.caption("🔥 热力图：动态色阶 (对比更鲜明)")
        # 优化点：text_auto='.2f' 保留两位小数，去掉 range_color 实现动态上下限
        fig_heat_fill = px.imshow(
            pivot_fill.fillna(0),
            labels=dict(x="Network", y="eCPM Range", color="Fill Rate (%)"),
            x=pivot_fill.columns,
            y=pivot_fill.index,
            text_auto='.2f',  # 👈 变动：保留2位小数
            aspect="auto",
            color_continuous_scale="RdYlGn"
            # range_color=[0, 100]  👈 变动：已移除，实现动态范围
        )
        fig_heat_fill.update_layout(height=500, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_heat_fill, use_container_width=True)
        
    with c2:
        st.caption("📋 详细数据表")
        st.dataframe(
            pivot_fill.style.format("{:.2f}%", na_rep="-")
            .background_gradient(cmap='RdYlGn', axis=None) # axis=None 使得颜色基于整个表的 Max/Min 分布
            .highlight_null(color='transparent'),
            use_container_width=True, 
            height=500
        )

    st.divider()
    
    # 2. 变现效率 (RPM)
    st.subheader("2. 变现效率 (RPM - Per 1M Requests)")
    c3, c4 = st.columns([3, 2])
    
    with c3:
        st.caption("🔥 热力图：动态色阶")
        # 优化点：text_auto='.2f'
        fig_heat_rpm = px.imshow(
            pivot_rpm.fillna(0),
            labels=dict(x="Network", y="eCPM Range", color="RPM ($)"),
            x=pivot_rpm.columns,
            y=pivot_rpm.index,
            text_auto='.2f', # 👈 变动：保留2位小数
            aspect="auto",
            color_continuous_scale="Blues"
        )
        fig_heat_rpm.update_layout(height=500, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_heat_rpm, use_container_width=True)
        
    with c4:
        st.caption("📋 详细数据表")
        st.dataframe(
            pivot_rpm.style.format("${:,.2f}", na_rep="-")
            .background_gradient(cmap='Blues', axis=None)
            .highlight_null(color='transparent'),
            use_container_width=True, 
            height=500
        )


# ================= 模块 2: 🌊 Waterfall 诊断 =================
def render_waterfall_dashboard(raw_df):
    """
    Waterfall 诊断
    """
    st.sidebar.markdown("### 📌 Waterfall 筛选")

    # 1. 网络 (必选)
    all_networks = sorted([x for x in raw_df['轮替网络'].unique() if x is not None])
    selected_network = st.sidebar.selectbox("1️⃣ 网络 (Network):", options=all_networks, index=None, key="wf_net")

    # 2. 广告类型 (必选)
    all_adtypes = sorted(raw_df['Ad Type'].astype(str).unique().tolist()) if 'Ad Type' in raw_df.columns else []
    selected_adtype = st.sidebar.selectbox("2️⃣ 广告类型 (Ad Type):", options=all_adtypes, index=None, key="wf_ad")

    st.sidebar.markdown("#### 🔧 维度筛选")

    # 3. 平台 (多选)
    if 'Platform' in raw_df.columns:
        all_platforms = sorted(raw_df['Platform'].astype(str).unique().tolist())
        selected_platforms = st.sidebar.multiselect("3️⃣ 平台:", options=all_platforms, default=all_platforms, key="wf_plat")
    else:
        selected_platforms = []

    # 4. 国家 (动态检测)
    has_country = 'Country' in raw_df.columns
    mask_country = True 

    if has_country:
        all_countries = sorted(raw_df['Country'].unique().astype(str).tolist())
        selected_countries = st.sidebar.multiselect(
            "4️⃣ 国家 (留空则默认全选):", 
            options=all_countries, 
            default=[], 
            key="wf_ctry"
        )
        target_countries = selected_countries if selected_countries else all_countries
        mask_country = raw_df['Country'].isin(target_countries)
    else:
        st.sidebar.info("🌐 数据源无 'Country' 列，展示全局数据")

    if not selected_network or not selected_adtype:
        st.warning("👈 请在左侧选择 **网络** 和 **广告类型** 以开始诊断。")
        return

    # --- 动态构建 Mask ---
    mask_net = (raw_df['轮替网络'] == selected_network)
    mask_ad = (raw_df['Ad Type'].astype(str) == selected_adtype)
    mask_plat = raw_df['Platform'].isin(selected_platforms) if 'Platform' in raw_df.columns else True
    
    mask_base = mask_net & mask_ad & mask_plat & mask_country
    df_base_filtered = raw_df[mask_base].copy()

    # --- 吸顶导航 ---
    header_container = st.container()
    with header_container:
        st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
        col_nav1, col_nav2 = st.columns([1, 2])
        with col_nav1:
            min_date, max_date = raw_df['Day'].min().date(), raw_df['Day'].max().date()
            date_range = st.date_input("📅 日期范围:", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="wf_date")
            start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
        with col_nav2:
            sub_mode = st.radio("📍 诊断视角:", ["1. 轮替效果分析", "2. 策略健康度诊断"], horizontal=True, key="wf_sub_mode")
        st.markdown('</div>', unsafe_allow_html=True)

    # 日期过滤
    mask_date = (df_base_filtered['Day'].dt.date >= start_date) & (df_base_filtered['Day'].dt.date <= end_date)
    df_filtered = df_base_filtered[mask_date].copy()

    if df_filtered.empty:
        st.error("⚠️ 当前筛选无数据。")
        return

    # --- KPI & Charts ---
    kpi_rev = df_filtered['Revenue'].sum()
    kpi_imp = df_filtered['Impressions'].sum()
    kpi_ecpm = (kpi_rev / kpi_imp * 1000) if kpi_imp > 0 else 0
    
    st.markdown(f"#### 📊 {selected_network} 核心指标")
    k1, k2, k3 = st.columns(3)
    k1.metric("💰 总收入", f"${kpi_rev:,.2f}")
    k2.metric("📉 平均 eCPM", f"${kpi_ecpm:,.2f}")
    k3.metric("👁️ 总展示", f"{kpi_imp:,.0f}")
    st.markdown("---")

    # 子视图逻辑
    if sub_mode == "1. 轮替效果分析":
        st.subheader("🔄 轮替版本生命周期")
        c1, c2 = st.columns(2)
        unique_apps = sorted(df_filtered['Application'].unique().tolist())
        with c1:
            idx = get_session_index('wf_app_1', unique_apps)
            sel_app = st.selectbox("选择 App:", unique_apps, index=idx, key='wf_app_1')
            thresh = st.number_input("过滤展示量 <", value=50, step=10, key='wf_th')
        
        chart_data = pd.DataFrame()
        sel_ecpm = None
        is_gam = (selected_network == 'GAM')

        with c2:
            if sel_app:
                app_data = df_filtered[(df_filtered['Application'] == sel_app) & (df_filtered['Impressions'] > thresh)]
                if is_gam:
                    chart_data = app_data
                else:
                    av_ecpms = sorted(app_data['eCPM_修正后'].unique())
                    if av_ecpms:
                        sel_ecpm = st.selectbox("选择 eCPM 层:", av_ecpms, key='wf_ec')
                        chart_data = app_data[app_data['eCPM_修正后'] == sel_ecpm]

        if not chart_data.empty:
            agg = chart_data.groupby(['Day', '轮替版本']).agg({'Attempts':'sum', 'Responses':'sum'}).reset_index()
            agg['Fill Rate'] = agg.apply(lambda x: (x['Responses']/x['Attempts']*100) if x['Attempts']>0 else 0, axis=1)
            agg = agg.sort_values('Day')
            agg['Date_Str'] = agg['Day'].dt.strftime('%Y-%m-%d')
            title = f'<b>{sel_app}</b>' + (' - GAM' if is_gam else f' - Floor: ${sel_ecpm}')
            fig = px.line(agg, x='Date_Str', y='Fill Rate', color='轮替版本', markers=True, title=title)
            fig.update_layout(yaxis_title="Fill Rate (%)", xaxis_title="Date", hovermode="x unified", height=500, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

    elif sub_mode == "2. 策略健康度诊断":
        st.subheader("📈 瀑布流分层诊断")
        agg = df_filtered.groupby(['Application', 'eCPM_修正后']).agg({'Attempts':'sum', 'Responses':'sum', 'Revenue':'sum'}).reset_index()
        agg['Weighted_Fill_Rate'] = agg.apply(lambda x: (x['Responses']/x['Attempts']*100) if x['Attempts']>0 else 0, axis=1)
        agg['Rev_Share'] = (agg['Revenue'] / agg.groupby('Application')['Revenue'].transform('sum') * 100).fillna(0)
        
        fig_macro = px.scatter(
            agg[agg['Weighted_Fill_Rate']>0], x="eCPM_修正后", y="Weighted_Fill_Rate", size="Revenue", color="Application",
            log_x=True, log_y=True, opacity=0.7, size_max=60, title=f"eCPM vs Fill Rate ({selected_network})"
        )
        fig_macro.add_hline(y=1, line_dash="dot", line_color="red")
        st.plotly_chart(fig_macro, use_container_width=True)

        st.divider()
        u_apps = sorted(agg['Application'].unique().tolist())
        idx_2 = get_session_index('wf_app_2', u_apps)
        sel_app_2 = st.selectbox("深度诊断 App:", u_apps, index=idx_2, key='wf_app_2')
        
        if sel_app_2:
            d_app = agg[agg['Application'] == sel_app_2].sort_values('eCPM_修正后')
            if not d_app.empty:
                fig_micro = make_subplots(specs=[[{"secondary_y": True}]])
                fig_micro.add_trace(go.Bar(
                    x=d_app['eCPM_修正后'].astype(str), y=d_app['Weighted_Fill_Rate'], name="Fill Rate",
                    marker_color='rgba(55, 128, 191, 0.7)'
                ), secondary_y=False)
                fig_micro.add_trace(go.Scatter(
                    x=d_app['eCPM_修正后'].astype(str), y=d_app['Rev_Share'], name="Rev Share",
                    marker_color='crimson', mode='lines+markers'
                ), secondary_y=True)
                fig_micro.update_layout(title=f"<b>{sel_app_2} Waterfall Structure</b>", height=550, legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_micro, use_container_width=True)


# ================= 3. 其他预留模块 =================
def render_bidding_dashboard():
    st.info("🚧 **Bidding 模块开发中**")

def render_dsp_dashboard():
    st.info("🚧 **DSP/直投 模块开发中**")

# ================= 4. 主程序入口 =================
def main():
    st.sidebar.title("🧭 业务导航")
    
    # 顶级导航
    app_mode = st.sidebar.radio(
        "选择板块:",
        ["🏆 大盘概览 (Benchmark)", "🌊 Waterfall (轮替)", "🔨 Bidding (竞价)", "🎯 DSP/直投"],
        index=0,
        key="main_nav"
    )
    st.sidebar.markdown("---")

    if app_mode in ["🏆 大盘概览 (Benchmark)", "🌊 Waterfall (轮替)"]:
        st.sidebar.markdown("### 📂 数据源")
        uploaded_file = st.sidebar.file_uploader("上传 AppLovin 报表", type=['xlsx', 'csv'], key="shared_uploader")
        
        if uploaded_file:
            raw_df = process_raw_data(uploaded_file)
            if raw_df is not None:
                if app_mode == "🏆 大盘概览 (Benchmark)":
                    render_benchmark_dashboard(raw_df)
                elif app_mode == "🌊 Waterfall (轮替)":
                    render_waterfall_dashboard(raw_df)
        else:
            st.info("👋 请先在左侧上传数据文件以开始分析。")

    elif app_mode == "🔨 Bidding (竞价)":
        render_bidding_dashboard()
    
    elif app_mode == "🎯 DSP/直投":
        render_dsp_dashboard()

if __name__ == "__main__":
    main()