"""
Airレジ 売上分析・需要予測 Webアプリ（v9: 高度な需要予測システム）

新機能:
- 内部実績分析（トレンド、季節性、カテゴリー別パフォーマンス）
- 外部環境分析（天気×売上相関、カレンダー効果、検索トレンド）
- 市場・顧客分析（ターゲット層別需要、類似商品分析、コンセプト評価）
- 総合需要予測エンジン（複数要因を統合した高精度予測）
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import calendar
import re

# モジュールのインポート
import sys
sys.path.append('.')
from modules.data_loader import SheetsDataLoader, aggregate_by_products, merge_with_calendar
from modules.product_normalizer import ProductNormalizer
from modules.demand_analyzer import InternalAnalyzer, ExternalAnalyzer, MarketAnalyzer, DemandForecastEngine
import config

# ページ設定
st.set_page_config(
    page_title="Airレジ 売上分析",
    page_icon="⛩️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .analysis-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .strength-item {
        background-color: #e8f5e9;
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
    }
    .weakness-item {
        background-color: #ffebee;
        border-left: 3px solid #f44336;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
    }
    .metric-highlight {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None
if 'normalizer' not in st.session_state:
    st.session_state.normalizer = None
if 'selected_products' not in st.session_state:
    st.session_state.selected_products = []
if 'categories' not in st.session_state:
    st.session_state.categories = {}
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'forecast_total' not in st.session_state:
    st.session_state.forecast_total = 0


# =============================================================================
# ユーティリティ関数
# =============================================================================

def round_up_to_50(value: int) -> int:
    """50の倍数に切り上げ"""
    if value <= 0:
        return 0
    return ((value + 49) // 50) * 50


def init_data():
    """データを初期化"""
    if st.session_state.data_loader is None:
        try:
            st.session_state.data_loader = SheetsDataLoader()
        except Exception as e:
            st.error(f"データ接続エラー: {e}")
            return False
    
    if st.session_state.normalizer is None:
        try:
            df_items = st.session_state.data_loader.load_item_sales()
            st.session_state.normalizer = ProductNormalizer()
            st.session_state.normalizer.build_master(df_items, "商品名")
            build_categories()
        except Exception as e:
            st.error(f"授与品マスタ構築エラー: {e}")
    
    return True


def build_categories():
    """カテゴリーをD列から取得"""
    if st.session_state.data_loader is None:
        return
    
    df_items = st.session_state.data_loader.load_item_sales()
    
    if df_items.empty:
        return
    
    categories = defaultdict(list)
    
    category_col = None
    for col in df_items.columns:
        if 'カテゴリ' in col or col == 'カテゴリー' or col == 'category':
            category_col = col
            break
    
    if category_col is None and len(df_items.columns) >= 4:
        category_col = df_items.columns[3]
    
    product_col = None
    for col in df_items.columns:
        if '商品名' in col or col == '商品' or col == 'product':
            product_col = col
            break
    
    if product_col is None and len(df_items.columns) >= 3:
        product_col = df_items.columns[2]
    
    if category_col is None or product_col is None:
        return
    
    for _, row in df_items[[product_col, category_col]].drop_duplicates().iterrows():
        product_name = row[product_col]
        category = row[category_col]
        
        if pd.isna(category) or str(category).strip() == '':
            category = 'その他'
        
        if st.session_state.normalizer:
            normalized = st.session_state.normalizer.normalize(product_name)
            if normalized and normalized not in categories[category]:
                categories[category].append(normalized)
    
    st.session_state.categories = dict(categories)


def render_header():
    """ヘッダーを描画"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown('<p class="main-header">⛩️ 授与品 売上分析・需要予測</p>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 データ更新"):
            st.cache_data.clear()
            st.session_state.data_loader = None
            st.session_state.selected_products = []
            st.session_state.sales_data = None
            st.session_state.forecast_data = None
            st.rerun()
    
    if st.session_state.data_loader:
        min_date, max_date = st.session_state.data_loader.get_date_range()
        if min_date and max_date:
            st.caption(f"📅 データ期間: {min_date.strftime('%Y年%m月%d日')} 〜 {max_date.strftime('%Y年%m月%d日')}")


# =============================================================================
# メインナビゲーション
# =============================================================================

def render_main_tabs():
    """メインタブを描画"""
    tab1, tab2, tab3 = st.tabs([
        "📊 既存授与品の分析・予測",
        "✨ 新規授与品の需要予測（高度版）",
        "📈 予測精度ダッシュボード"
    ])
    
    with tab1:
        render_existing_product_analysis()
    
    with tab2:
        render_advanced_new_product_forecast()
    
    with tab3:
        render_accuracy_dashboard()


# =============================================================================
# 既存授与品の分析（従来機能 + 高度分析）
# =============================================================================

def render_existing_product_analysis():
    """既存授与品の分析・予測"""
    render_product_selection()
    start_date, end_date = render_period_selection()
    sales_data = render_sales_analysis(start_date, end_date)
    
    if sales_data is not None and not sales_data.empty:
        render_advanced_analysis(sales_data)
    
    render_forecast_section(sales_data)
    render_delivery_section()


def render_product_selection():
    """授与品選択セクション"""
    st.markdown('<p class="section-header">① 授与品を選ぶ</p>', unsafe_allow_html=True)
    
    search_query = st.text_input(
        "授与品名を入力",
        placeholder="例: 金運、お守り、御朱印帳...",
        key="search_input"
    )
    
    if search_query and st.session_state.normalizer:
        results = st.session_state.normalizer.search(search_query, limit=20)
        
        if results:
            st.write(f"**{len(results)}件** 見つかりました")
            
            cols = st.columns(4)
            for i, result in enumerate(results):
                name = result['normalized_name']
                
                with cols[i % 4]:
                    is_selected = name in st.session_state.selected_products
                    
                    if st.checkbox(name, value=is_selected, key=f"search_{name}"):
                        if name not in st.session_state.selected_products:
                            st.session_state.selected_products.append(name)
                    else:
                        if name in st.session_state.selected_products:
                            st.session_state.selected_products.remove(name)
    
    if st.session_state.selected_products:
        st.info(f"✅ 選択中: {', '.join(st.session_state.selected_products[:5])}{'...' if len(st.session_state.selected_products) > 5 else ''}")
        
        if st.button("🗑️ クリア"):
            st.session_state.selected_products = []
            st.rerun()


def render_period_selection():
    """期間選択セクション"""
    st.markdown('<p class="section-header">② 期間を選ぶ</p>', unsafe_allow_html=True)
    
    today = date.today()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        preset = st.selectbox(
            "プリセット",
            ["過去1年", "過去6ヶ月", "過去3ヶ月", "過去2年", "全期間"],
            index=0
        )
    
    presets = {
        "過去1年": (today - timedelta(days=365), today),
        "過去6ヶ月": (today - timedelta(days=180), today),
        "過去3ヶ月": (today - timedelta(days=90), today),
        "過去2年": (today - timedelta(days=730), today),
        "全期間": (date(2022, 8, 1), today)
    }
    
    default_start, default_end = presets[preset]
    
    with col2:
        start_date = st.date_input("開始日", value=default_start)
    
    with col3:
        end_date = st.date_input("終了日", value=default_end)
    
    return start_date, end_date


def render_sales_analysis(start_date: date, end_date: date):
    """売上分析セクション"""
    st.markdown('<p class="section-header">③ 売上を見る</p>', unsafe_allow_html=True)
    
    if not st.session_state.selected_products:
        st.info("授与品を選択すると、ここに売上が表示されます")
        return None
    
    df_items = st.session_state.data_loader.load_item_sales()
    
    if df_items.empty:
        st.warning("データがありません")
        return None
    
    mask = (df_items['date'] >= pd.Timestamp(start_date)) & (df_items['date'] <= pd.Timestamp(end_date))
    df_filtered = df_items[mask]
    
    original_names = st.session_state.normalizer.get_all_original_names(
        st.session_state.selected_products
    )
    
    df_agg = aggregate_by_products(df_filtered, original_names, aggregate=True)
    
    if df_agg.empty:
        st.warning("該当期間にデータがありません")
        return None
    
    df_agg = df_agg.sort_values('date').reset_index(drop=True)
    
    total_qty = int(df_agg['販売商品数'].sum())
    total_sales = df_agg['販売総売上'].sum()
    period_days = (end_date - start_date).days + 1
    avg_daily = total_qty / period_days
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🛒 販売数量", f"{total_qty:,}体")
    col2.metric("💰 売上合計", f"¥{total_sales:,.0f}")
    col3.metric("📈 平均日販", f"{avg_daily:.1f}体/日")
    col4.metric("📅 期間", f"{period_days}日間")
    
    st.session_state.sales_data = df_agg
    
    return df_agg


def render_advanced_analysis(sales_data: pd.DataFrame):
    """高度な分析セクション"""
    
    with st.expander("📊 **高度な分析を見る**", expanded=False):
        
        # 分析モジュールを初期化
        try:
            df_calendar = st.session_state.data_loader.load_calendar()
        except:
            df_calendar = None
        
        internal = InternalAnalyzer(sales_data)
        external = ExternalAnalyzer(sales_data, df_calendar)
        
        tab1, tab2, tab3 = st.tabs(["📈 トレンド分析", "🗓️ 季節性分析", "🌤️ 外部要因分析"])
        
        with tab1:
            render_trend_analysis(internal)
        
        with tab2:
            render_seasonality_analysis(internal)
        
        with tab3:
            render_external_analysis(external)


def render_trend_analysis(internal: InternalAnalyzer):
    """トレンド分析の表示"""
    trend = internal.analyze_sales_trend()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("トレンド", trend['trend_direction'])
    
    with col2:
        growth = trend['growth_rate']
        st.metric("成長率", f"{growth:+.1f}%")
    
    with col3:
        st.metric("変動性", f"{trend['volatility']:.2f}")
    
    if trend['peak_periods']:
        st.write(f"**ピーク期間**: {', '.join(trend['peak_periods'][:5])}")
    
    # グラフ
    if 'monthly_data' in trend and not trend['monthly_data'].empty:
        fig = px.line(
            trend['monthly_data'], x='period', y='販売商品数',
            title='月別販売推移',
            markers=True
        )
        fig.update_traces(line_color='#1E88E5')
        st.plotly_chart(fig, use_container_width=True)


def render_seasonality_analysis(internal: InternalAnalyzer):
    """季節性分析の表示"""
    seasonality = internal.detect_seasonality()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**月別係数**（1.0が平均）")
        
        monthly = seasonality['monthly_pattern']
        months = [f"{m}月" for m in range(1, 13)]
        values = [monthly.get(m, 1.0) for m in range(1, 13)]
        
        fig = px.bar(
            x=months, y=values,
            labels={'x': '月', 'y': '係数'},
            color=values,
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**曜日別係数**（1.0が平均）")
        
        weekday = seasonality['weekday_pattern']
        days = list(weekday.keys())
        day_values = list(weekday.values())
        
        fig = px.bar(
            x=days, y=day_values,
            labels={'x': '曜日', 'y': '係数'},
            color=day_values,
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    st.metric("季節性の強さ", f"{seasonality['seasonality_strength']:.2f}", 
              help="0に近いほど安定、1に近いほど季節変動が大きい")


def render_external_analysis(external: ExternalAnalyzer):
    """外部要因分析の表示"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**カレンダー効果**")
        
        calendar_effect = external.analyze_calendar_effect()
        
        if calendar_effect['available']:
            st.metric("休日の影響", f"×{calendar_effect['holiday_impact']:.2f}")
            
            if calendar_effect['rokuyou_impact']:
                st.write("六曜別の影響:")
                for rok, impact in sorted(calendar_effect['rokuyou_impact'].items(), 
                                         key=lambda x: x[1], reverse=True):
                    bar_len = int(impact * 20)
                    st.text(f"  {rok}: {'█' * bar_len} {impact:.2f}")
        else:
            st.info("カレンダーデータがありません")
    
    with col2:
        st.write("**天気の影響**")
        
        weather_effect = external.analyze_weather_correlation()
        
        if weather_effect['available']:
            if weather_effect['weather_impact']:
                for weather, impact in sorted(weather_effect['weather_impact'].items(), 
                                             key=lambda x: x[1], reverse=True):
                    emoji = {'晴れ': '☀️', '曇り': '☁️', '雨': '🌧️', '雪': '❄️'}.get(weather, '🌤️')
                    st.text(f"  {emoji} {weather}: ×{impact:.2f}")
        else:
            st.info("天気データがありません")


def render_forecast_section(sales_data: pd.DataFrame):
    """需要予測セクション"""
    st.markdown('<p class="section-header">④ 需要を予測する</p>', unsafe_allow_html=True)
    
    if sales_data is None or sales_data.empty:
        st.info("売上データがあると、需要予測ができます")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider("予測日数", 30, 365, 180, key="forecast_days_existing")
    
    with col2:
        method = st.selectbox(
            "予測方法",
            ["季節性考慮（おすすめ）", "移動平均法（シンプル）", "すべての方法で比較"],
            index=0
        )
    
    if st.button("🔮 需要を予測", type="primary", use_container_width=True, key="forecast_btn_existing"):
        with st.spinner("予測中..."):
            forecast = forecast_with_seasonality(sales_data, forecast_days)
            
            if forecast is not None and not forecast.empty:
                raw_total = int(forecast['predicted'].sum())
                rounded_total = round_up_to_50(raw_total)
                
                st.success("✅ 予測完了！")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("📦 予測販売総数", f"{rounded_total:,}体")
                col2.metric("📈 平均日販（予測）", f"{forecast['predicted'].mean():.1f}体/日")
                col3.metric("📅 予測期間", f"{forecast_days}日間")
                
                st.session_state.forecast_data = forecast
                st.session_state.forecast_total = rounded_total


def render_delivery_section():
    """納品計画セクション"""
    st.markdown('<p class="section-header">⑤ 納品計画を立てる</p>', unsafe_allow_html=True)
    
    forecast = st.session_state.get('forecast_data')
    
    if forecast is None or forecast.empty:
        st.info("需要予測を実行すると、納品計画を立てられます")
        return
    
    total_demand = st.session_state.get('forecast_total', 0)
    st.info(f"📦 予測された需要数: **{total_demand:,}体**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_stock = st.number_input("🏠 現在の在庫数", min_value=0, value=500, step=50, key="stock_existing")
    
    with col2:
        min_stock = st.number_input("⚠️ 安全在庫数", min_value=0, value=100, step=50, key="min_stock_existing")
    
    needed = total_demand + min_stock - current_stock
    recommended_order = round_up_to_50(max(0, needed))
    
    st.metric("推奨発注数", f"{recommended_order:,}体")


# =============================================================================
# 高度な新規授与品需要予測
# =============================================================================

def render_advanced_new_product_forecast():
    """高度な新規授与品需要予測"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; padding: 20px; color: white; margin-bottom: 20px;">
        <h2>✨ 新規授与品の需要予測（高度版）</h2>
        <p>内部実績・外部環境・市場分析を統合した高精度な需要予測を行います。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # Step 1: 基本情報の入力
    # ==========================================================================
    st.markdown('<p class="section-header">① 授与品の基本情報</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_product_name = st.text_input(
            "授与品名 *",
            placeholder="例: 縁結び水晶守",
            help="新しく作る授与品の名前"
        )
        
        new_product_category = st.selectbox(
            "カテゴリー *",
            ["お守り", "御朱印", "御朱印帳", "おみくじ", "絵馬", "お札", "縁起物", "その他"],
            help="最も近いカテゴリーを選んでください"
        )
        
        new_product_price = st.number_input(
            "価格（円） *",
            min_value=100,
            max_value=50000,
            value=1000,
            step=100,
            help="販売予定価格"
        )
    
    with col2:
        new_product_description = st.text_area(
            "特徴・コンセプト",
            placeholder="例: 水晶を使用した縁結びのお守り。若い女性向け。恋愛成就に特化。SNS映えするデザイン。",
            help="授与品の特徴を詳しく記述するほど、予測精度が向上します",
            height=100
        )
        
        target_audience = st.multiselect(
            "ターゲット層 *",
            ["若い女性", "若い男性", "中高年女性", "中高年男性", "家族連れ", "観光客", "地元の方"],
            default=["若い女性", "観光客"],
            help="主なターゲット層を選んでください（複数選択可）"
        )
    
    # ==========================================================================
    # Step 2: 高度な分析
    # ==========================================================================
    st.markdown('<p class="section-header">② 市場分析・類似商品分析</p>', unsafe_allow_html=True)
    
    if not new_product_name:
        st.info("👆 授与品名を入力すると、分析が開始されます")
        return
    
    # 分析エンジンを初期化
    df_sales = st.session_state.data_loader.load_item_sales()
    
    try:
        df_calendar = st.session_state.data_loader.load_calendar()
    except:
        df_calendar = None
    
    forecast_engine = DemandForecastEngine(df_sales, df_calendar)
    
    # 類似商品を検索
    similar_products = find_similar_products(
        df_sales, new_product_name, new_product_category, new_product_price
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📦 類似商品分析**")
        
        if similar_products:
            st.success(f"{len(similar_products)}件の類似商品を発見")
            
            for i, prod in enumerate(similar_products[:5], 1):
                similarity_bar = "█" * int(prod['similarity'] / 10)
                st.markdown(f"""
                <div class="analysis-card">
                    <strong>{i}. {prod['name'][:25]}...</strong><br>
                    平均: {prod['avg_daily']:.1f}体/日 | 類似度: {similarity_bar} {prod['similarity']:.0f}%
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("類似商品が見つかりませんでした")
    
    with col2:
        st.write("**📊 コンセプト評価**")
        
        # コンセプト評価
        market = MarketAnalyzer(df_sales)
        concept_score = market.score_concept({
            'name': new_product_name,
            'description': new_product_description,
            'target_segments': target_audience,
            'price': new_product_price,
            'category': new_product_category
        })
        
        # スコア表示
        st.markdown(f"""
        <div class="score-card">
            <div style="font-size: 3rem; font-weight: bold;">{concept_score['total_score']}</div>
            <div style="font-size: 1.2rem;">/100点</div>
            <div style="margin-top: 10px; font-size: 1.5rem;">{concept_score['rank']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 強み・弱み
        if concept_score['strengths']:
            st.write("**💪 強み**")
            for s in concept_score['strengths'][:3]:
                st.markdown(f'<div class="strength-item">✓ {s}</div>', unsafe_allow_html=True)
        
        if concept_score['weaknesses']:
            st.write("**⚠️ 改善点**")
            for w in concept_score['weaknesses'][:3]:
                st.markdown(f'<div class="weakness-item">△ {w}</div>', unsafe_allow_html=True)
    
    # ==========================================================================
    # Step 3: 外部環境分析
    # ==========================================================================
    st.markdown('<p class="section-header">③ 外部環境分析</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🔍 検索トレンド**")
        trends = forecast_engine.external.fetch_google_trends(new_product_name)
        
        st.metric("現在の関心度", f"{trends['current_interest']:.0f}/100")
        st.write(f"トレンド: {trends['trend_direction']}")
        st.write(f"ピーク月: {trends['peak_month']}")
        st.caption(trends['note'])
    
    with col2:
        st.write("**🗓️ カレンダー効果**")
        calendar_effect = forecast_engine.external.analyze_calendar_effect()
        
        if calendar_effect['available']:
            st.metric("休日の影響", f"×{calendar_effect['holiday_impact']:.2f}")
            
            if calendar_effect['special_period_impact']:
                top_period = max(calendar_effect['special_period_impact'].items(), 
                                key=lambda x: x[1])
                st.write(f"最大効果: {top_period[0]} (×{top_period[1]:.2f})")
        else:
            st.info("データなし")
    
    with col3:
        st.write("**🌤️ 天気の影響**")
        weather_effect = forecast_engine.external.analyze_weather_correlation()
        
        if weather_effect['available']:
            rain_impact = weather_effect.get('rain_impact', 1.0)
            st.metric("雨天時の影響", f"×{rain_impact:.2f}")
            
            if weather_effect['temperature_correlation'] != 0:
                st.write(f"気温との相関: {weather_effect['temperature_correlation']:.2f}")
        else:
            st.info("データなし")
    
    # ==========================================================================
    # Step 4: 需要予測
    # ==========================================================================
    st.markdown('<p class="section-header">④ 総合需要予測</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_period = st.selectbox(
            "予測期間",
            ["1ヶ月（30日）", "3ヶ月（90日）", "6ヶ月（180日）", "1年（365日）"],
            index=2
        )
        period_days = {"1ヶ月（30日）": 30, "3ヶ月（90日）": 90, 
                      "6ヶ月（180日）": 180, "1年（365日）": 365}[forecast_period]
    
    with col2:
        confidence_level = st.selectbox(
            "予測の保守性",
            ["楽観的", "標準", "保守的"],
            index=1,
            help="保守的を選ぶと、少なめに予測します"
        )
    
    with col3:
        include_learning = st.checkbox(
            "学習データを活用",
            value=True,
            help="過去の予測精度データを活用して予測を補正"
        )
    
    if st.button("🔮 **総合需要予測を実行**", type="primary", use_container_width=True):
        if not target_audience:
            st.error("ターゲット層を1つ以上選択してください")
        else:
            with st.spinner("複数の分析を統合して予測中..."):
                # 総合予測を実行
                result = forecast_engine.forecast_new_product(
                    product_name=new_product_name,
                    category=new_product_category,
                    price=new_product_price,
                    description=new_product_description,
                    target_segments=target_audience,
                    similar_products=similar_products,
                    forecast_days=period_days,
                    confidence_level=confidence_level
                )
                
                display_comprehensive_forecast_result(result, new_product_name, new_product_price)


def find_similar_products(df_sales: pd.DataFrame, name: str, category: str, price: int) -> list:
    """類似商品を検索"""
    
    if df_sales.empty:
        return []
    
    product_col = '商品名'
    qty_col = '販売商品数'
    sales_col = '販売総売上'
    
    # 商品ごとの統計
    product_stats = df_sales.groupby(product_col).agg({
        qty_col: ['sum', 'mean', 'count'],
        sales_col: 'sum'
    }).reset_index()
    
    product_stats.columns = ['name', 'total_qty', 'avg_daily', 'days_count', 'total_sales']
    product_stats['unit_price'] = product_stats['total_sales'] / product_stats['total_qty']
    product_stats['unit_price'] = product_stats['unit_price'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 類似度計算
    similar = []
    keywords = set(re.findall(r'[\u4e00-\u9fff]+', name.lower()))
    
    for _, row in product_stats.iterrows():
        prod_name = row['name']
        
        # 名前の類似度
        name_keywords = set(re.findall(r'[\u4e00-\u9fff]+', prod_name.lower()))
        name_match = len(keywords & name_keywords) / max(len(keywords), 1) * 50
        
        # 価格の類似度
        if row['unit_price'] > 0:
            price_diff = abs(price - row['unit_price']) / price
            price_match = max(0, (1 - price_diff)) * 30
        else:
            price_match = 0
        
        # カテゴリーの類似度
        category_keywords = {
            "お守り": ["守", "お守り", "まもり"],
            "御朱印": ["御朱印", "朱印"],
            "御朱印帳": ["御朱印帳", "朱印帳"],
            "おみくじ": ["おみくじ", "みくじ"],
            "絵馬": ["絵馬"],
            "お札": ["札", "お札"],
        }
        
        cat_match = 0
        for cat, kws in category_keywords.items():
            if cat == category:
                for kw in kws:
                    if kw in prod_name:
                        cat_match = 20
                        break
        
        similarity = name_match + price_match + cat_match
        
        if similarity > 10 and row['total_qty'] > 0:
            similar.append({
                'name': prod_name,
                'total_qty': row['total_qty'],
                'avg_daily': row['avg_daily'],
                'unit_price': row['unit_price'],
                'similarity': similarity
            })
    
    similar.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar[:10]


def display_comprehensive_forecast_result(result: Dict, product_name: str, price: int):
    """総合予測結果を表示"""
    
    st.success("✅ 総合需要予測が完了しました！")
    
    # メイン結果
    st.markdown(f"### 📦 「{product_name}」の需要予測結果")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="score-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <div style="font-size: 0.9rem;">予測販売総数</div>
            <div style="font-size: 2.5rem; font-weight: bold;">{result['total_qty_rounded']:,}</div>
            <div style="font-size: 1rem;">体</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="score-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div style="font-size: 0.9rem;">予測売上</div>
            <div style="font-size: 2rem; font-weight: bold;">¥{result['total_qty_rounded'] * price:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="score-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div style="font-size: 0.9rem;">平均日販</div>
            <div style="font-size: 2.5rem; font-weight: bold;">{result['avg_daily']:.1f}</div>
            <div style="font-size: 1rem;">体/日</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="score-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div style="font-size: 0.9rem;">分析品質</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{result['analysis_quality']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 信頼区間
    ci = result['confidence_interval']
    st.info(f"📊 95%信頼区間: **{ci['lower']:,}体** 〜 **{ci['upper']:,}体**")
    
    # 詳細分析
    with st.expander("📊 **詳細な分析結果を見る**", expanded=True):
        
        tab1, tab2, tab3 = st.tabs(["📈 予測内訳", "🎯 調整係数", "📅 月別予測"])
        
        with tab1:
            analysis = result['analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**ベース予測**")
                st.metric("基本日販", f"{analysis['base_daily']:.1f}体/日")
                st.metric("総合調整係数", f"×{analysis['total_multiplier']:.2f}")
                st.metric("調整後日販", f"{result['avg_daily']:.1f}体/日")
            
            with col2:
                st.write("**コンセプト評価**")
                cs = analysis['concept_score']
                st.metric("総合スコア", f"{cs['total_score']}点")
                st.write(f"ランク: {cs['rank']}")
        
        with tab2:
            adj = analysis['adjustments']
            
            st.write("**適用された調整係数**")
            
            adj_data = [
                {"要因": "ターゲット層", "係数": adj['target_multiplier'], 
                 "説明": "選択したターゲット層に基づく調整"},
                {"要因": "コンセプト評価", "係数": adj['concept_multiplier'], 
                 "説明": "商品コンセプトの評価に基づく調整"},
                {"要因": "検索トレンド", "係数": adj['trend_multiplier'], 
                 "説明": "現在の検索関心度に基づく調整"},
            ]
            
            df_adj = pd.DataFrame(adj_data)
            st.dataframe(df_adj, use_container_width=True, hide_index=True)
        
        with tab3:
            # 月別予測グラフ
            daily = result['daily_forecast']
            df_daily = pd.DataFrame(daily)
            df_daily['month'] = pd.to_datetime(df_daily['date']).dt.to_period('M')
            monthly = df_daily.groupby('month')['predicted'].sum().reset_index()
            monthly['month'] = monthly['month'].astype(str)
            
            fig = px.bar(
                monthly, x='month', y='predicted',
                title='月別予測販売数',
                labels={'month': '月', 'predicted': '予測販売数（体）'},
                color='predicted',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 発注提案
    st.markdown("### 📋 初回発注量の提案")
    
    col1, col2, col3 = st.columns(3)
    
    one_month = round_up_to_50(int(result['avg_daily'] * 30))
    three_months = round_up_to_50(int(result['avg_daily'] * 90))
    six_months = round_up_to_50(int(result['avg_daily'] * 180))
    
    with col1:
        st.metric("少なめ（1ヶ月分）", f"{one_month}体", help="リスクを抑えたい場合")
    
    with col2:
        st.metric("標準（3ヶ月分）", f"{three_months}体", help="おすすめ", delta="推奨")
    
    with col3:
        st.metric("多め（6ヶ月分）", f"{six_months}体", help="在庫切れを避けたい場合")
    
    st.caption("💡 新規授与品は売れ行きが不確実なため、最初は少なめに発注し、様子を見ることをおすすめします。")
    
    # 注意事項
    with st.expander("⚠️ 予測の注意事項"):
        st.markdown(f"""
        **この予測は参考値です。以下の点にご注意ください：**
        
        1. **分析品質**: {result['analysis_quality']}
           - 類似商品: {result['similar_count']}件のデータを参照
           - 信頼区間: {ci['lower']:,}〜{ci['upper']:,}体
        
        2. **新規商品の不確実性**
           - 実際の売れ行きは予測と大きく異なる可能性があります
           - 発売後1ヶ月間の実績を見て、予測を修正してください
        
        3. **外部要因の影響**
           - 天候、社会情勢、競合状況により変動します
           - 特に正月・GW・お盆は予測以上に売れる可能性があります
        
        **おすすめの進め方：**
        1. 初回は少なめ（1〜2ヶ月分）を発注
        2. 発売後2週間の実績を確認
        3. 実績を見て追加発注または在庫調整
        """)


def render_accuracy_dashboard():
    """予測精度ダッシュボード"""
    
    st.markdown('<p class="section-header">📈 予測精度ダッシュボード</p>', unsafe_allow_html=True)
    
    st.info("""
    📊 予測精度ダッシュボードを表示するには、自動学習システムのセットアップが必要です。
    
    **セットアップ手順：**
    1. GitHubにリポジトリをプッシュ
    2. GitHub Secretsを設定
    3. GitHub Actionsが毎日自動実行
    4. 数日後にここにデータが表示されます
    """)


# =============================================================================
# 予測関数
# =============================================================================

def forecast_with_seasonality(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """季節性を考慮した予測"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    overall_mean = df['販売商品数'].mean()
    
    if pd.isna(overall_mean) or overall_mean == 0:
        overall_mean = 1
    
    # 曜日係数
    df['weekday'] = df['date'].dt.dayofweek
    weekday_means = df.groupby('weekday')['販売商品数'].mean()
    weekday_factor = {}
    for wd in range(7):
        if wd in weekday_means.index and weekday_means[wd] > 0:
            weekday_factor[wd] = weekday_means[wd] / overall_mean
        else:
            weekday_factor[wd] = 1.0
    
    # 月係数
    df['month'] = df['date'].dt.month
    month_means = df.groupby('month')['販売商品数'].mean()
    month_factor = {}
    for m in range(1, 13):
        if m in month_means.index and month_means[m] > 0:
            month_factor[m] = month_means[m] / overall_mean
        else:
            month_factor[m] = 1.0
    
    # 予測
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    predictions = []
    for d in future_dates:
        weekday_f = weekday_factor.get(d.dayofweek, 1.0)
        month_f = month_factor.get(d.month, 1.0)
        
        pred = overall_mean * weekday_f * month_f
        pred = max(0.1, pred)
        
        predictions.append({
            'date': d,
            'predicted': round(pred)
        })
    
    return pd.DataFrame(predictions)


# =============================================================================
# メイン関数
# =============================================================================

def main():
    """メイン関数"""
    if not init_data():
        st.stop()
    
    render_header()
    st.divider()
    render_main_tabs()
    
    st.divider()
    st.caption("⛩️ 酒列磯前神社 授与品管理システム v9")


if __name__ == "__main__":
    main()
