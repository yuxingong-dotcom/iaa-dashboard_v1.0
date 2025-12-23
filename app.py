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

def clean_axis_labels(pivot_df):
    """
    将索引 "B05:0.30-0.50" 转换为 "$0.30 - 0.50"
    """
    new_index = []
    for label in pivot_df.index:
        label_str = str(label)
        if ':' in label_str:
            price_part = label_str.split(':')[-1]
            new_index.append(f"${price_part}")
        else:
            new_index.append(label_str)
    pivot_df.index = new_index
    return pivot_df

def get_hidden_networks_hint(raw_df, df_pool):
    """
    对比原始数据和当前过滤后的数据，找出被过滤掉的网络，用于提示用户
    """
    if raw_df is None or df_pool is None:
        return None
    
    # 原始所有网络
    all_raw_nets = set(raw_df['轮替网络'].dropna().astype(str).unique())
    # 当前池子里的网络
    current_pool_nets = set(df_pool['轮替网络'].dropna().astype(str).unique())
    
    # 差集
    hidden_nets = all_raw_nets - current_pool_nets
    return sorted(list(hidden_nets))

# ================= 统一侧边栏逻辑 =================
def render_sidebar(raw_df):
    """
    统一的左侧筛选栏，返回经过基础维度过滤后的 DataFrame
    """
    st.sidebar.title("🔍 全局筛选")
    st.sidebar.markdown("---")
    
    # 1. 广告类型 (必选)
    if 'Ad Type' not in raw_df.columns:
        st.error("❌ 数据源缺少 'Ad Type' 列，无法分析。")
        return None, None

    all_adtypes = sorted(raw_df['Ad Type'].astype(str).unique().tolist())
    selected_adtype = st.sidebar.selectbox(
        "1️⃣ 广告类型 (必选):", options=all_adtypes, index=None, placeholder="请选择...", key="global_ad"
    )

    st.sidebar.markdown("#### 🔧 维度筛选")
    
    # 2. 平台 (多选)
    if 'Platform' in raw_df.columns:
        all_platforms = sorted(raw_df['Platform'].astype(str).unique().tolist())
        selected_platforms = st.sidebar.multiselect("2️⃣ 平台:", options=all_platforms, default=all_platforms, key="global_plat")
    else:
        selected_platforms = []
        
    # 3. 国家 (动态检测，留空全选)
    has_country = 'Country' in raw_df.columns
    mask_country = True 

    if has_country:
        all_countries = sorted(raw_df['Country'].unique().astype(str).tolist())
        selected_countries = st.sidebar.multiselect(
            "3️⃣ 国家 (留空则默认全选):", 
            options=all_countries, 
            default=[], 
            key="global_ctry"
        )
        target_countries = selected_countries if selected_countries else all_countries
        mask_country = raw_df['Country'].isin(target_countries)
    else:
        st.sidebar.info("🌐 数据源无 'Country' 列，展示全局数据")

    if not selected_adtype:
        st.warning("👈 请在左侧选择 **广告类型** 以开始分析。")
        return None, selected_adtype

    # --- 基础过滤 ---
    mask_ad = (raw_df['Ad Type'].astype(str) == selected_adtype)
    mask_plat = raw_df['Platform'].isin(selected_platforms) if 'Platform' in raw_df.columns else True
    
    mask_base = mask_ad & mask_plat & mask_country
    df_pool = raw_df[mask_base].copy()
    
    return df_pool, selected_adtype


# ================= 模块 0: 数据源预览 =================
def render_data_preview_dashboard(raw_df):
    st.header("📂 数据源预览 (Processed Data)")
    if raw_df is None or raw_df.empty:
        st.warning("暂无数据")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 总行数", f"{raw_df.shape[0]:,}")
    c2.metric("📑 总列数", f"{raw_df.shape[1]}")
    if 'Day' in raw_df.columns:
        min_d, max_d = raw_df['Day'].min().date(), raw_df['Day'].max().date()
        c3.metric("📅 数据时间段", f"{min_d} ~ {max_d}")
    
    st.markdown("##### 🕸️ 识别到的所有网络")
    all_nets = sorted(raw_df['轮替网络'].dropna().astype(str).unique().tolist())
    st.write(f"共发现 {len(all_nets)} 个网络: {', '.join(all_nets)}")
    st.divider()
    st.subheader("1. 详细数据表")
    st.dataframe(raw_df, use_container_width=True, height=500)
    st.subheader("2. 导出数据")
    csv = raw_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(label="⬇️ 下载 CSV", data=csv, file_name='processed_iaa_data.csv', mime='text/csv')


# ================= 模块 1: Waterfall 全局数据概览 (UI 统一终极版) =================
def render_global_overview(df_pool, raw_df, selected_adtype):
    # --- CSS: 强制放大 Dataframe 字体 ---
    st.markdown("""
        <style>
        div[data-testid="stDataFrame"] { font-size: 1.1rem !important; }
        div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {
            font-size: 1.1rem !important; padding: 8px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if df_pool is None or df_pool.empty:
        st.error("⚠️ 当前筛选条件下无数据。")
        return

    # --- 吸顶导航栏 ---
    header_container = st.container()
    with header_container:
        st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            min_date, max_date = df_pool['Day'].min().date(), df_pool['Day'].max().date()
            date_range = st.date_input("📅 日期范围:", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="ov_date")
            start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
        with c2:
            available_apps = sorted(df_pool['Application'].unique().tolist())
            selected_apps = st.multiselect("📱 筛选 App (留空全选):", options=available_apps, default=[], key="ov_app")
        with c3:
            available_nets = sorted(df_pool['轮替网络'].dropna().astype(str).unique().tolist())
            selected_nets = st.multiselect("🕸️ 筛选 Network (留空全选):", options=available_nets, default=[], key="ov_net")
        st.markdown('</div>', unsafe_allow_html=True)

    hidden_nets = get_hidden_networks_hint(raw_df, df_pool)
    if hidden_nets:
        st.caption(f"ℹ️ **提示**: 网络 {', '.join(hidden_nets)} 不含 **{selected_adtype}** 数据。")

    target_apps = selected_apps if selected_apps else available_apps
    target_nets = selected_nets if selected_nets else available_nets
    
    mask_final = (
        (df_pool['Day'].dt.date >= start_date) & 
        (df_pool['Day'].dt.date <= end_date) &
        (df_pool['Application'].isin(target_apps)) &
        (df_pool['轮替网络'].isin(target_nets))
    )
    df_filtered = df_pool[mask_final].copy()

    if df_filtered.empty:
        st.error("⚠️ 当前筛选无数据。")
        return

    st.header(f"🌊 Waterfall 全局数据概览: {selected_adtype}")

    # ================= PART 1: 综合概览 (双列热力表) =================
    st.subheader("1. 综合概览 (Fill Rate & RPM)")
    st.markdown("💡 **绿色列**: 填充率 (Fill Rate) | **蓝色列**: 变现效率 (RPM)")

    agg_range = df_filtered.groupby(['eCPM_Range', '轮替网络']).agg({
        'Attempts': 'sum', 'Responses': 'sum', 'Revenue': 'sum'
    }).reset_index()

    agg_range['Fill Rate'] = agg_range.apply(lambda x: (x['Responses']/x['Attempts']*100) if x['Attempts']>0 else None, axis=1)
    agg_range['RPM'] = agg_range.apply(lambda x: (x['Revenue']/x['Attempts']*1000000) if x['Attempts']>0 else None, axis=1)

    pivot_df = agg_range.pivot(index='eCPM_Range', columns='轮替网络', values=['Fill Rate', 'RPM'])
    pivot_df.columns = pivot_df.columns.swaplevel(0, 1)
    pivot_df.sort_index(axis=1, inplace=True)
    pivot_df = clean_axis_labels(pivot_df)

    idx = pd.IndexSlice
    styler = pivot_df.style
    styler.format("{:.2f}%", subset=idx[:, idx[:, 'Fill Rate']], na_rep="-")
    styler.format("${:.4f}", subset=idx[:, idx[:, 'RPM']], na_rep="-")
    styler.background_gradient(cmap='Greens', subset=idx[:, idx[:, 'Fill Rate']], vmin=0, axis=None)
    styler.background_gradient(cmap='Blues', subset=idx[:, idx[:, 'RPM']], axis=None)
    styler.set_properties(**{'text-align': 'center', 'width': '100px'})
    styler.highlight_null(color='transparent')

    st.dataframe(styler, use_container_width=True, height=600)

    st.divider()

    # ================= PART 2: 每日趋势全景 (双列布局) =================
    c_title, c_toggle = st.columns([3, 2])
    with c_title:
        st.subheader("2. 每日趋势全景 (Daily Trend: Fill Rate vs RPM)")
    with c_toggle:
        view_mode = st.radio(
            "👀 选择视图模式:", 
            ["分层热力图 (Side-by-Side Heatmaps)", "分面折线图 (Grid Line Charts)"], 
            horizontal=True,
            label_visibility="collapsed"
        )

    # 1. 数据聚合
    agg_daily = df_filtered.groupby(['Day', 'eCPM_Range', '轮替网络']).agg({
        'Attempts': 'sum', 'Responses': 'sum', 'Revenue': 'sum'
    }).reset_index()
    
    agg_daily['Date_Str'] = agg_daily['Day'].dt.strftime('%Y-%m-%d')
    agg_daily['Fill Rate'] = agg_daily.apply(lambda x: (x['Responses']/x['Attempts']*100) if x['Attempts']>0 else None, axis=1)
    agg_daily['RPM'] = agg_daily.apply(lambda x: (x['Revenue']/x['Attempts']*1000000) if x['Attempts']>0 else None, axis=1)
    
    all_dates = sorted(agg_daily['Date_Str'].unique())
    available_ranges = sorted(agg_daily['eCPM_Range'].unique())
    display_networks = sorted(target_nets) 

    if not agg_daily.empty:
        
        # 准备子图标题
        subplot_titles = []
        for r in available_ranges:
            subplot_titles.append(f"{r} - Fill Rate")
            subplot_titles.append(f"{r} - RPM")
            
        rows_count = len(available_ranges)
        
        # --- 模式 A: 双列分层热力图 ---
        if view_mode == "分层热力图 (Side-by-Side Heatmaps)":
            st.markdown("💡 **左侧绿色**: Fill Rate % | **右侧蓝色**: RPM $")
            subplot_height = max(len(display_networks) * 35 + 50, 180)
            total_height = rows_count * subplot_height

            fig_stack = make_subplots(
                rows=rows_count, cols=2, 
                subplot_titles=subplot_titles, # 标题在上方
                vertical_spacing=0.06, horizontal_spacing=0.05,
                shared_xaxes=True
            )

            for i, range_val in enumerate(available_ranges):
                df_sub = agg_daily[agg_daily['eCPM_Range'] == range_val]
                
                # 左图: Fill Rate
                pivot_fr = df_sub.pivot(index='轮替网络', columns='Date_Str', values='Fill Rate')
                pivot_fr = pivot_fr.reindex(index=display_networks, columns=all_dates)
                text_fr = pivot_fr.applymap(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
                max_fr = pivot_fr.max().max()
                if pd.isna(max_fr) or max_fr == 0: max_fr = 100

                fig_stack.add_trace(go.Heatmap(
                    z=pivot_fr.values, x=pivot_fr.columns, y=pivot_fr.index,
                    text=text_fr.values, texttemplate="%{text}", textfont={"size": 12},
                    colorscale="Greens", zmin=0, zmax=max_fr, showscale=False
                ), row=i+1, col=1)

                # 右图: RPM
                pivot_rpm = df_sub.pivot(index='轮替网络', columns='Date_Str', values='RPM')
                pivot_rpm = pivot_rpm.reindex(index=display_networks, columns=all_dates)
                text_rpm = pivot_rpm.applymap(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
                max_rpm = pivot_rpm.max().max()
                if pd.isna(max_rpm) or max_rpm == 0: max_rpm = 10

                fig_stack.add_trace(go.Heatmap(
                    z=pivot_rpm.values, x=pivot_rpm.columns, y=pivot_rpm.index,
                    text=text_rpm.values, texttemplate="%{text}", textfont={"size": 12},
                    colorscale="Blues", zmin=0, zmax=max_rpm, showscale=False
                ), row=i+1, col=2)

            fig_stack.update_layout(height=total_height, margin=dict(l=0, r=0, t=40, b=0), font=dict(size=14))
            fig_stack.update_xaxes(showticklabels=True, row=rows_count, col=1)
            fig_stack.update_xaxes(showticklabels=True, row=rows_count, col=2)
            st.plotly_chart(fig_stack, use_container_width=True)

        # --- 模式 B: 双列分面折线图 (重构为手写 Scatter) ---
        else:
            st.markdown("💡 **趋势对比**: 点击图例可隐藏/显示特定网络。")
            
            # 为每个网络分配固定颜色，防止子图间颜色乱序
            colors = px.colors.qualitative.Plotly
            net_color_map = {net: colors[i % len(colors)] for i, net in enumerate(display_networks)}

            subplot_height = 250 # 折线图稍微高一点
            total_height = rows_count * subplot_height

            fig_grid = make_subplots(
                rows=rows_count, cols=2,
                subplot_titles=subplot_titles, # 标题在上方，与热力图统一
                vertical_spacing=0.08, horizontal_spacing=0.05,
                shared_xaxes=True
            )

            for i, range_val in enumerate(available_ranges):
                df_sub = agg_daily[agg_daily['eCPM_Range'] == range_val]
                
                # 遍历每个网络画线
                for net in display_networks:
                    net_data = df_sub[df_sub['轮替网络'] == net]
                    
                    # 如果该网络在该区间无数据，跳过画线 (或者画空线)
                    if net_data.empty:
                        continue

                    # 左图: Fill Rate 线
                    fig_grid.add_trace(go.Scatter(
                        x=net_data['Date_Str'], y=net_data['Fill Rate'],
                        mode='lines+markers', name=net,
                        line=dict(color=net_color_map.get(net, 'grey')),
                        legendgroup=net, # 核心：分组图例，点击一个控制所有
                        showlegend=(i==0) # 只在第一行显示图例
                    ), row=i+1, col=1)

                    # 右图: RPM 线
                    fig_grid.add_trace(go.Scatter(
                        x=net_data['Date_Str'], y=net_data['RPM'],
                        mode='lines+markers', name=net,
                        line=dict(color=net_color_map.get(net, 'grey')),
                        legendgroup=net, # 同组
                        showlegend=False # 不重复显示图例
                    ), row=i+1, col=2)

            fig_grid.update_layout(
                height=total_height, 
                margin=dict(l=0, r=0, t=60, b=0),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.005, x=0.5, xanchor="center", font=dict(size=14)),
                font=dict(size=14)
            )
            # 独立 Y 轴范围
            fig_grid.update_yaxes(showticklabels=True)
            
            st.plotly_chart(fig_grid, use_container_width=True)

    else:
        st.warning("无每日趋势数据")

# ================= 模块 2: Waterfall 细分数据 =================
def render_breakdown_dashboard(df_pool, raw_df, selected_adtype):
    if df_pool is None or df_pool.empty:
        st.error("⚠️ 当前筛选条件下无数据。")
        return
    header_container = st.container()
    with header_container:
        st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            min_date, max_date = df_pool['Day'].min().date(), df_pool['Day'].max().date()
            date_range = st.date_input("📅 日期范围:", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="bd_date")
            start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
        with col2:
            all_networks = sorted(df_pool['轮替网络'].dropna().astype(str).unique().tolist())
            selected_network = st.selectbox("🕸️ 选择网络 (必选):", options=all_networks, index=None, placeholder="请选择网络...", key="bd_net")
        with col3:
            sub_mode = st.radio("📍 诊断视角:", ["1. 策略健康度诊断", "2. 轮替效果分析"], horizontal=True, key="bd_mode")
        st.markdown('</div>', unsafe_allow_html=True)

    if not selected_network:
        st.info("👋 请在上方选择一个 **网络** 以开始细分诊断。")
        return

    mask_final = (
        (df_pool['Day'].dt.date >= start_date) & 
        (df_pool['Day'].dt.date <= end_date) &
        (df_pool['轮替网络'] == selected_network)
    )
    df_filtered = df_pool[mask_final].copy()

    if df_filtered.empty:
        st.error(f"⚠️ 网络 {selected_network} 在当前时间范围内无数据。")
        return

    kpi_rev = df_filtered['Revenue'].sum()
    kpi_imp = df_filtered['Impressions'].sum()
    kpi_ecpm = (kpi_rev / kpi_imp * 1000) if kpi_imp > 0 else 0
    
    st.header(f"🔬 细分数据: {selected_network} ({selected_adtype})")
    k1, k2, k3 = st.columns(3)
    k1.metric("💰 总收入", f"${kpi_rev:,.2f}")
    k2.metric("📉 平均 eCPM", f"${kpi_ecpm:,.2f}")
    k3.metric("👁️ 总展示", f"{kpi_imp:,.0f}")
    st.markdown("---")

    if sub_mode == "1. 策略健康度诊断":
        st.subheader("📈 瀑布流分层诊断 (Strategy Health)")
        agg = df_filtered.groupby(['Application', 'eCPM_修正后']).agg({'Attempts':'sum', 'Responses':'sum', 'Revenue':'sum'}).reset_index()
        agg['Weighted_Fill_Rate'] = agg.apply(lambda x: (x['Responses']/x['Attempts']*100) if x['Attempts']>0 else 0, axis=1)
        app_rev_sum = agg.groupby('Application')['Revenue'].transform('sum')
        agg['Rev_Share'] = (agg['Revenue'] / app_rev_sum * 100).fillna(0)
        
        fig_macro = px.scatter(
            agg[agg['Weighted_Fill_Rate']>0], x="eCPM_修正后", y="Weighted_Fill_Rate", size="Revenue", color="Application",
            log_x=True, log_y=True, opacity=0.7, size_max=60, title=f"eCPM vs Fill Rate ({selected_network})"
        )
        fig_macro.add_hline(y=1, line_dash="dot", line_color="red")
        st.plotly_chart(fig_macro, use_container_width=True)
        st.divider()
        u_apps = sorted(agg['Application'].unique().tolist())
        idx_2 = get_session_index('bd_app_diag', u_apps)
        sel_app_2 = st.selectbox("🔎 深度诊断 App (查看具体 Floor 结构):", u_apps, index=idx_2, key='bd_app_diag')
        if sel_app_2:
            d_app = agg[agg['Application'] == sel_app_2].sort_values('eCPM_修正后')
            if not d_app.empty:
                fig_micro = make_subplots(specs=[[{"secondary_y": True}]])
                fig_micro.add_trace(go.Bar(x=d_app['eCPM_修正后'].astype(str), y=d_app['Weighted_Fill_Rate'], name="Fill Rate (%)", marker_color='rgba(55, 128, 191, 0.7)'), secondary_y=False)
                fig_micro.add_trace(go.Scatter(x=d_app['eCPM_修正后'].astype(str), y=d_app['Rev_Share'], name="Revenue Share (%)", marker_color='crimson', mode='lines+markers'), secondary_y=True)
                fig_micro.update_layout(title=f"<b>{sel_app_2} Waterfall Structure</b>", height=550, legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_micro, use_container_width=True)

    elif sub_mode == "2. 轮替效果分析":
        st.subheader("🔄 轮替版本生命周期 (Rotation Analysis)")
        c1, c2 = st.columns(2)
        unique_apps = sorted(df_filtered['Application'].unique().tolist())
        with c1:
            idx = get_session_index('bd_app_rot', unique_apps)
            sel_app = st.selectbox("选择 App:", unique_apps, index=idx, key='bd_app_rot')
            thresh = st.number_input("过滤展示量 <", value=50, step=10, key='bd_th')
        
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
                        sel_ecpm = st.selectbox("选择 eCPM 层:", av_ecpms, key='bd_ec')
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
        else: st.warning("该筛选条件下无足够数据生成轮替曲线。")


# ================= 模块 3: 低填充层级透视 (新增) =================
def render_low_fill_dashboard(df_pool, raw_df, selected_adtype):
    """
    识别填充率低于底线的具体层级
    """
    if df_pool is None or df_pool.empty:
        st.error("⚠️ 当前筛选条件下无数据。")
        return

    # --- 吸顶导航栏: 阈值 + 筛选 ---
    header_container = st.container()
    with header_container:
        st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        
        # 0. 阈值设置
        with c1:
            threshold_val = st.number_input("📉 填充率底线 (%):", value=0.001, min_value=0.0, step=0.001, format="%.4f", help="筛选低于此填充率的层级")
            
        # 1. 日期
        with c2:
            min_date, max_date = df_pool['Day'].min().date(), df_pool['Day'].max().date()
            date_range = st.date_input("📅 日期范围:", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="lf_date")
            start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
            
        # 2. App
        with c3:
            available_apps = sorted(df_pool['Application'].unique().tolist())
            selected_apps = st.multiselect("📱 App (留空全选):", options=available_apps, default=[], key="lf_app")
            
        # 3. Network
        with c4:
            available_nets = sorted(df_pool['轮替网络'].dropna().astype(str).unique().tolist())
            selected_nets = st.multiselect("🕸️ Network (留空全选):", options=available_nets, default=[], key="lf_net")
            
        st.markdown('</div>', unsafe_allow_html=True)

    target_apps = selected_apps if selected_apps else available_apps
    target_nets = selected_nets if selected_nets else available_nets

    # 过滤数据
    mask_final = (
        (df_pool['Day'].dt.date >= start_date) & 
        (df_pool['Day'].dt.date <= end_date) &
        (df_pool['Application'].isin(target_apps)) &
        (df_pool['轮替网络'].isin(target_nets))
    )
    df_filtered = df_pool[mask_final].copy()

    if df_filtered.empty:
        st.warning("⚠️ 当前筛选条件下无数据。")
        return

    st.header(f"📉 低填充层级透视: {selected_adtype}")
    st.markdown(f"**筛选标准**: 填充率 < **{threshold_val}%** (且请求量 > 0)")

    # 聚合计算
    # 维度: App, Floor Price (eCPM_修正后), Network
    agg = df_filtered.groupby(['Application', 'eCPM_修正后', '轮替网络']).agg({
        'Attempts': 'sum', 
        'Responses': 'sum'
    }).reset_index()

    # 计算填充率
    agg['Fill Rate (%)'] = (agg['Responses'] / agg['Attempts'] * 100).fillna(0)

    # 核心筛选: Attempts > 0 且 Fill Rate < 阈值
    problem_df = agg[(agg['Attempts'] > 0) & (agg['Fill Rate (%)'] < threshold_val)].copy()

    if problem_df.empty:
        st.success(f"🎉 太棒了！在当前筛选范围内，没有发现填充率低于 {threshold_val}% 的层级。")
        return

    # 排序：按请求量降序，优先展示流量大的问题层级
    problem_df = problem_df.sort_values(by='Attempts', ascending=False)
    

    # 格式化展示
    problem_df['eCPM_修正后'] = problem_df['eCPM_修正后'].apply(lambda x: f"${x}")
    problem_df['Fill Rate (%)'] = problem_df['Fill Rate (%)'].map('{:.4f}%'.format)
    
    # 重命名列以更友好
    display_df = problem_df.rename(columns={
        'eCPM_修正后': 'Floor Price',
        '轮替网络': 'Network'
    })

    # 概览指标
    st.markdown(f"🚨 共发现 **{len(display_df)}** 个低填充层级，涉及总请求量 **{display_df['Attempts'].sum():,}**")

    st.dataframe(
        display_df[['Application', 'Network', 'Floor Price', 'Fill Rate (%)', 'Attempts', 'Responses']],
        use_container_width=True,
        height=600,
        hide_index=True
    )
    
    # 下载按钮
    csv = display_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="⬇️ 下载低填充报告",
        data=csv,
        file_name='low_fill_report.csv',
        mime='text/csv',
    )


# ================= 其他预留模块 =================
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
        ["📁 数据源预览", "waterfall 全局数据概览", "waterfall 细分数据", "📉 低填充层级透视", "🔨 Bidding (竞价)", "🎯 DSP/直投"],
        index=0,
        key="main_nav"
    )
    
    uploaded_file = st.sidebar.file_uploader("📂 上传报表 (xlsx/csv):", type=['xlsx', 'csv'], key="shared_uploader")
    raw_df = None
    if uploaded_file:
        raw_df = process_raw_data(uploaded_file)
    else:
        if app_mode not in ["🔨 Bidding (竞价)", "🎯 DSP/直投"]:
             st.info("👋 请先在左侧上传数据文件以开始分析。")

    if app_mode == "📁 数据源预览":
        render_data_preview_dashboard(raw_df)

    elif app_mode == "waterfall 全局数据概览":
        if raw_df is not None:
            df_pool, selected_adtype = render_sidebar(raw_df)
            if df_pool is not None:
                render_global_overview(df_pool, raw_df, selected_adtype)

    elif app_mode == "waterfall 细分数据":
        if raw_df is not None:
            df_pool, selected_adtype = render_sidebar(raw_df)
            if df_pool is not None:
                render_breakdown_dashboard(df_pool, raw_df, selected_adtype)
    
    elif app_mode == "📉 低填充层级透视":
        if raw_df is not None:
            df_pool, selected_adtype = render_sidebar(raw_df)
            if df_pool is not None:
                render_low_fill_dashboard(df_pool, raw_df, selected_adtype)

    elif app_mode == "🔨 Bidding (竞价)":
        render_bidding_dashboard()
    
    elif app_mode == "🎯 DSP/直投":
        render_dsp_dashboard()

# test123456


if __name__ == "__main__":
    main()