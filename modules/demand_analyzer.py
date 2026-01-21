"""
高度な需要分析モジュール

機能:
1. 内部実績分析（定量的）
   - 販売トレンド分析
   - 季節性・周期性の検出
   - カテゴリー別パフォーマンス
   - 客単価分析

2. 外部環境分析（定量的）
   - 天気×売上の相関分析
   - カレンダー効果分析
   - Google Trends連携

3. 市場・顧客分析（定性的）
   - ターゲット層別需要推定
   - 類似商品の成功パターン分析
   - コンセプト評価スコアリング
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from collections import defaultdict
import re


# =============================================================================
# 1. 内部実績分析（定量的）
# =============================================================================

class InternalAnalyzer:
    """内部実績分析クラス"""
    
    def __init__(self, df_sales: pd.DataFrame):
        """
        Args:
            df_sales: 売上データ（date, 商品名, 販売商品数, 販売総売上）
        """
        self.df = df_sales.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self._prepare_data()
    
    def _prepare_data(self):
        """データの前処理"""
        # 日付関連の列を追加
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['weekday'] = self.df['date'].dt.dayofweek
        self.df['week'] = self.df['date'].dt.isocalendar().week
        self.df['is_weekend'] = self.df['weekday'] >= 5
    
    def analyze_sales_trend(self, product_name: Optional[str] = None) -> Dict:
        """
        販売トレンド分析
        
        Returns:
            trend_direction: 上昇/下降/横ばい
            growth_rate: 成長率（%）
            volatility: 変動性（標準偏差/平均）
            peak_periods: ピーク期間
        """
        if product_name:
            df = self.df[self.df['商品名'] == product_name]
        else:
            df = self.df
        
        # 月別集計
        monthly = df.groupby(['year', 'month'])['販売商品数'].sum().reset_index()
        monthly['period'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
        
        if len(monthly) < 3:
            return {
                'trend_direction': '判定不可',
                'growth_rate': 0,
                'volatility': 0,
                'peak_periods': [],
                'monthly_data': monthly
            }
        
        # トレンド分析（線形回帰）
        x = np.arange(len(monthly))
        y = monthly['販売商品数'].values
        
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            
            # 成長率計算
            avg = y.mean()
            if avg > 0:
                growth_rate = (slope * len(x)) / avg * 100
            else:
                growth_rate = 0
            
            # トレンド方向判定
            if growth_rate > 10:
                trend_direction = '上昇傾向 📈'
            elif growth_rate < -10:
                trend_direction = '下降傾向 📉'
            else:
                trend_direction = '横ばい ➡️'
        else:
            slope = 0
            growth_rate = 0
            trend_direction = '判定不可'
        
        # 変動性（変動係数）
        volatility = y.std() / y.mean() if y.mean() > 0 else 0
        
        # ピーク期間の特定
        threshold = y.mean() + y.std()
        peak_mask = monthly['販売商品数'] > threshold
        peak_periods = monthly[peak_mask]['period'].tolist()
        
        return {
            'trend_direction': trend_direction,
            'growth_rate': round(growth_rate, 1),
            'volatility': round(volatility, 2),
            'peak_periods': peak_periods,
            'monthly_data': monthly,
            'slope': slope
        }
    
    def detect_seasonality(self, product_name: Optional[str] = None) -> Dict:
        """
        季節性・周期性の検出
        
        Returns:
            monthly_pattern: 月別係数
            weekday_pattern: 曜日別係数
            seasonality_strength: 季節性の強さ（0-1）
        """
        if product_name:
            df = self.df[self.df['商品名'] == product_name]
        else:
            df = self.df
        
        overall_mean = df['販売商品数'].mean()
        
        if overall_mean == 0:
            return {
                'monthly_pattern': {m: 1.0 for m in range(1, 13)},
                'weekday_pattern': {w: 1.0 for w in range(7)},
                'seasonality_strength': 0
            }
        
        # 月別パターン
        monthly_mean = df.groupby('month')['販売商品数'].mean()
        monthly_pattern = {}
        for m in range(1, 13):
            if m in monthly_mean.index:
                monthly_pattern[m] = round(monthly_mean[m] / overall_mean, 2)
            else:
                monthly_pattern[m] = 1.0
        
        # 曜日別パターン
        weekday_mean = df.groupby('weekday')['販売商品数'].mean()
        weekday_pattern = {}
        weekday_names = ['月', '火', '水', '木', '金', '土', '日']
        for w in range(7):
            if w in weekday_mean.index:
                weekday_pattern[weekday_names[w]] = round(weekday_mean[w] / overall_mean, 2)
            else:
                weekday_pattern[weekday_names[w]] = 1.0
        
        # 季節性の強さ（月別変動係数）
        monthly_values = list(monthly_pattern.values())
        seasonality_strength = np.std(monthly_values) / np.mean(monthly_values) if np.mean(monthly_values) > 0 else 0
        
        return {
            'monthly_pattern': monthly_pattern,
            'weekday_pattern': weekday_pattern,
            'seasonality_strength': round(min(1.0, seasonality_strength), 2)
        }
    
    def analyze_category_performance(self) -> Dict:
        """
        カテゴリー別パフォーマンス分析
        
        Returns:
            category_stats: カテゴリーごとの統計
            top_performers: 上位パフォーマー
            growth_categories: 成長カテゴリー
        """
        # カテゴリー列を特定
        category_col = None
        for col in self.df.columns:
            if 'カテゴリ' in col:
                category_col = col
                break
        
        if category_col is None:
            return {
                'category_stats': {},
                'top_performers': [],
                'growth_categories': []
            }
        
        # カテゴリー別集計
        category_stats = {}
        
        for category in self.df[category_col].dropna().unique():
            cat_df = self.df[self.df[category_col] == category]
            
            total_qty = cat_df['販売商品数'].sum()
            total_sales = cat_df['販売総売上'].sum()
            avg_daily = cat_df.groupby('date')['販売商品数'].sum().mean()
            unique_products = cat_df['商品名'].nunique()
            
            # 成長率計算
            trend = self.analyze_sales_trend()
            growth_rate = trend['growth_rate']
            
            category_stats[category] = {
                'total_qty': int(total_qty),
                'total_sales': int(total_sales),
                'avg_daily': round(avg_daily, 1),
                'unique_products': unique_products,
                'growth_rate': growth_rate
            }
        
        # 上位パフォーマー
        sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]['total_qty'], reverse=True)
        top_performers = [cat for cat, _ in sorted_cats[:5]]
        
        # 成長カテゴリー
        growth_categories = [cat for cat, stats in category_stats.items() if stats['growth_rate'] > 10]
        
        return {
            'category_stats': category_stats,
            'top_performers': top_performers,
            'growth_categories': growth_categories
        }
    
    def analyze_unit_price(self, product_name: Optional[str] = None) -> Dict:
        """
        客単価分析
        
        Returns:
            avg_unit_price: 平均単価
            price_trend: 単価トレンド
            price_range: 価格帯
        """
        if product_name:
            df = self.df[self.df['商品名'] == product_name]
        else:
            df = self.df
        
        # 単価計算
        total_qty = df['販売商品数'].sum()
        total_sales = df['販売総売上'].sum()
        
        avg_unit_price = total_sales / total_qty if total_qty > 0 else 0
        
        # 日別単価の推移
        daily = df.groupby('date').agg({
            '販売商品数': 'sum',
            '販売総売上': 'sum'
        })
        daily['unit_price'] = daily['販売総売上'] / daily['販売商品数']
        daily['unit_price'] = daily['unit_price'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 単価トレンド
        if len(daily) > 1:
            prices = daily['unit_price'].values
            if len(prices) > 1:
                slope, _ = np.polyfit(range(len(prices)), prices, 1)
                if slope > 10:
                    price_trend = '上昇中'
                elif slope < -10:
                    price_trend = '下降中'
                else:
                    price_trend = '安定'
            else:
                price_trend = '判定不可'
        else:
            price_trend = '判定不可'
        
        return {
            'avg_unit_price': round(avg_unit_price, 0),
            'price_trend': price_trend,
            'price_range': {
                'min': round(daily['unit_price'].min(), 0) if len(daily) > 0 else 0,
                'max': round(daily['unit_price'].max(), 0) if len(daily) > 0 else 0
            }
        }


# =============================================================================
# 2. 外部環境分析（定量的）
# =============================================================================

class ExternalAnalyzer:
    """外部環境分析クラス"""
    
    def __init__(self, df_sales: pd.DataFrame, df_calendar: Optional[pd.DataFrame] = None):
        self.df_sales = df_sales.copy()
        self.df_sales['date'] = pd.to_datetime(self.df_sales['date'])
        self.df_calendar = df_calendar
        
        if self.df_calendar is not None:
            self.df_calendar['date'] = pd.to_datetime(self.df_calendar['date'])
    
    def analyze_weather_correlation(self) -> Dict:
        """
        天気×売上の相関分析
        
        Returns:
            weather_impact: 天気別の影響度
            temperature_correlation: 気温との相関
            rain_impact: 雨の影響度
        """
        if self.df_calendar is None or 'weather' not in self.df_calendar.columns:
            return {
                'weather_impact': {},
                'temperature_correlation': 0,
                'rain_impact': 0,
                'available': False
            }
        
        # 売上とカレンダーをマージ
        daily_sales = self.df_sales.groupby('date')['販売商品数'].sum().reset_index()
        merged = daily_sales.merge(self.df_calendar, on='date', how='left')
        
        overall_mean = merged['販売商品数'].mean()
        
        if overall_mean == 0:
            return {
                'weather_impact': {},
                'temperature_correlation': 0,
                'rain_impact': 0,
                'available': False
            }
        
        # 天気別の影響度
        weather_impact = {}
        if 'weather' in merged.columns:
            weather_mean = merged.groupby('weather')['販売商品数'].mean()
            for weather, mean in weather_mean.items():
                weather_impact[weather] = round(mean / overall_mean, 2)
        
        # 気温との相関
        temperature_correlation = 0
        if 'temperature' in merged.columns:
            valid = merged.dropna(subset=['temperature', '販売商品数'])
            if len(valid) > 10:
                temperature_correlation = valid['temperature'].corr(valid['販売商品数'])
                temperature_correlation = round(temperature_correlation, 2) if not np.isnan(temperature_correlation) else 0
        
        # 雨の影響度
        rain_impact = weather_impact.get('雨', weather_impact.get('雨', 1.0))
        
        return {
            'weather_impact': weather_impact,
            'temperature_correlation': temperature_correlation,
            'rain_impact': rain_impact,
            'available': True
        }
    
    def analyze_calendar_effect(self) -> Dict:
        """
        カレンダー効果分析（休日、六曜、特別日）
        
        Returns:
            holiday_impact: 休日の影響度
            rokuyou_impact: 六曜別の影響度
            special_period_impact: 特別期間の影響度
        """
        if self.df_calendar is None:
            return {
                'holiday_impact': 1.0,
                'rokuyou_impact': {},
                'special_period_impact': {},
                'available': False
            }
        
        # 売上とカレンダーをマージ
        daily_sales = self.df_sales.groupby('date')['販売商品数'].sum().reset_index()
        merged = daily_sales.merge(self.df_calendar, on='date', how='left')
        
        overall_mean = merged['販売商品数'].mean()
        
        if overall_mean == 0:
            return {
                'holiday_impact': 1.0,
                'rokuyou_impact': {},
                'special_period_impact': {},
                'available': False
            }
        
        # 休日の影響度
        holiday_impact = 1.0
        if 'is_holiday' in merged.columns:
            holiday_mean = merged[merged['is_holiday'] == True]['販売商品数'].mean()
            if not np.isnan(holiday_mean):
                holiday_impact = round(holiday_mean / overall_mean, 2)
        
        # 六曜別の影響度
        rokuyou_impact = {}
        if 'rokuyou' in merged.columns:
            rokuyou_mean = merged.groupby('rokuyou')['販売商品数'].mean()
            for rokuyou, mean in rokuyou_mean.items():
                if pd.notna(rokuyou):
                    rokuyou_impact[rokuyou] = round(mean / overall_mean, 2)
        
        # 特別期間の影響度
        special_period_impact = {}
        if 'special_period' in merged.columns:
            special_mean = merged.groupby('special_period')['販売商品数'].mean()
            for period, mean in special_mean.items():
                if pd.notna(period) and period:
                    special_period_impact[period] = round(mean / overall_mean, 2)
        
        return {
            'holiday_impact': holiday_impact,
            'rokuyou_impact': rokuyou_impact,
            'special_period_impact': special_period_impact,
            'available': True
        }
    
    def fetch_google_trends(self, keyword: str, days: int = 90) -> Dict:
        """
        Google Trends（検索ボリューム）を取得
        
        注: 実際のGoogle Trends APIは制限があるため、
        ここでは代替としてシミュレーションデータを使用
        
        Args:
            keyword: 検索キーワード
            days: 取得期間
        
        Returns:
            trend_data: トレンドデータ
            trend_direction: トレンド方向
            peak_interest: ピーク関心度
        """
        # Google Trends APIの代わりに、
        # SerpAPI（有料）やPyTrends（非公式）を使用可能
        # ここでは神社関連の一般的なパターンをシミュレート
        
        # 神社関連キーワードの季節パターン
        shrine_patterns = {
            'お守り': {'1': 3.0, '7': 1.0, '11': 1.2, '12': 1.5},
            '御朱印': {'1': 2.0, '4': 1.2, '5': 1.3, '8': 1.1, '11': 1.3},
            '縁結び': {'2': 1.3, '5': 1.1, '7': 1.2, '11': 1.4, '12': 1.2},
            '厄除け': {'1': 2.5, '2': 1.5, '12': 1.3},
            '金運': {'1': 2.0, '11': 1.2, '12': 1.3},
        }
        
        # キーワードに応じたパターンを取得
        pattern = {}
        for key, pat in shrine_patterns.items():
            if key in keyword:
                pattern = pat
                break
        
        if not pattern:
            pattern = {'1': 1.5, '5': 1.1, '8': 1.0, '12': 1.2}
        
        # 現在の月に基づく関心度
        current_month = str(datetime.now().month)
        current_interest = pattern.get(current_month, 1.0)
        
        # トレンド方向の判定
        next_month = str((datetime.now().month % 12) + 1)
        next_interest = pattern.get(next_month, 1.0)
        
        if next_interest > current_interest * 1.1:
            trend_direction = '上昇傾向 📈'
        elif next_interest < current_interest * 0.9:
            trend_direction = '下降傾向 📉'
        else:
            trend_direction = '横ばい ➡️'
        
        # ピーク月
        peak_month = max(pattern.items(), key=lambda x: x[1])[0]
        
        return {
            'keyword': keyword,
            'current_interest': round(current_interest * 50, 0),  # 0-100スケール
            'trend_direction': trend_direction,
            'peak_month': f"{peak_month}月",
            'pattern': pattern,
            'note': '※ Google Trendsのシミュレーションデータです'
        }


# =============================================================================
# 3. 市場・顧客分析（定性的）
# =============================================================================

class MarketAnalyzer:
    """市場・顧客分析クラス"""
    
    def __init__(self, df_sales: pd.DataFrame):
        self.df_sales = df_sales.copy()
        self.df_sales['date'] = pd.to_datetime(self.df_sales['date'])
    
    def estimate_target_demand(self, target_segments: List[str], 
                               product_category: str,
                               base_daily: float) -> Dict:
        """
        ターゲット層別需要推定
        
        Args:
            target_segments: ターゲット層のリスト
            product_category: 商品カテゴリー
            base_daily: 基本日販
        
        Returns:
            segment_estimates: セグメント別の需要推定
            total_multiplier: 総合係数
            confidence: 信頼度
        """
        # セグメント別の係数（神社の実績に基づく推定）
        segment_factors = {
            '若い女性': {'お守り': 1.5, '御朱印': 1.3, 'おみくじ': 1.4, '絵馬': 1.2, 'default': 1.2},
            '若い男性': {'お守り': 0.8, '御朱印': 1.0, 'おみくじ': 1.0, '絵馬': 0.9, 'default': 0.9},
            '中高年女性': {'お守り': 1.3, '御朱印': 1.5, 'おみくじ': 1.0, 'お札': 1.4, 'default': 1.2},
            '中高年男性': {'お守り': 1.0, '御朱印': 1.2, 'おみくじ': 0.8, 'お札': 1.3, 'default': 1.0},
            '家族連れ': {'お守り': 1.8, '御朱印': 0.8, 'おみくじ': 2.0, '絵馬': 1.5, 'default': 1.3},
            '観光客': {'お守り': 1.2, '御朱印': 2.0, 'おみくじ': 1.5, '絵馬': 1.3, 'default': 1.3},
            '地元の方': {'お守り': 1.0, '御朱印': 0.8, 'おみくじ': 0.9, 'お札': 1.5, 'default': 1.0},
        }
        
        # セグメント別の需要推定
        segment_estimates = {}
        multipliers = []
        
        for segment in target_segments:
            factors = segment_factors.get(segment, {'default': 1.0})
            factor = factors.get(product_category, factors.get('default', 1.0))
            
            estimated = base_daily * factor
            segment_estimates[segment] = {
                'factor': factor,
                'estimated_daily': round(estimated, 1)
            }
            multipliers.append(factor)
        
        # 総合係数（平均）
        total_multiplier = np.mean(multipliers) if multipliers else 1.0
        
        # 信頼度（セグメント数に基づく）
        confidence = min(1.0, len(target_segments) / 3)
        
        return {
            'segment_estimates': segment_estimates,
            'total_multiplier': round(total_multiplier, 2),
            'confidence': round(confidence, 2),
            'adjusted_daily': round(base_daily * total_multiplier, 1)
        }
    
    def analyze_similar_product_success(self, similar_products: List[Dict]) -> Dict:
        """
        類似商品の成功パターン分析
        
        Args:
            similar_products: 類似商品のリスト（name, total_qty, avg_daily, unit_price）
        
        Returns:
            success_patterns: 成功パターン
            failure_patterns: 失敗パターン
            recommendations: 推奨事項
        """
        if not similar_products:
            return {
                'success_patterns': [],
                'failure_patterns': [],
                'recommendations': ['類似商品データがないため、少量から開始することをお勧めします'],
                'avg_performance': 0
            }
        
        # パフォーマンスで分類
        avg_daily_values = [p.get('avg_daily', 0) for p in similar_products]
        overall_avg = np.mean(avg_daily_values) if avg_daily_values else 0
        
        success_products = [p for p in similar_products if p.get('avg_daily', 0) > overall_avg * 1.2]
        failure_products = [p for p in similar_products if p.get('avg_daily', 0) < overall_avg * 0.5]
        
        # 成功パターンの特徴抽出
        success_patterns = []
        if success_products:
            avg_price = np.mean([p.get('unit_price', 0) for p in success_products])
            success_patterns.append(f"平均単価: ¥{avg_price:,.0f}")
            
            # 名前の共通キーワード
            names = [p.get('name', '') for p in success_products]
            common_words = self._extract_common_keywords(names)
            if common_words:
                success_patterns.append(f"よく使われるキーワード: {', '.join(common_words[:3])}")
        
        # 失敗パターンの特徴抽出
        failure_patterns = []
        if failure_products:
            avg_price = np.mean([p.get('unit_price', 0) for p in failure_products])
            failure_patterns.append(f"平均単価: ¥{avg_price:,.0f}")
        
        # 推奨事項
        recommendations = []
        if success_products:
            top_product = max(success_products, key=lambda x: x.get('avg_daily', 0))
            recommendations.append(f"「{top_product.get('name', '')}」を参考にすると良いでしょう")
        
        if overall_avg > 0:
            recommendations.append(f"類似商品の平均日販は {overall_avg:.1f}体/日 です")
        
        return {
            'success_patterns': success_patterns,
            'failure_patterns': failure_patterns,
            'recommendations': recommendations,
            'avg_performance': round(overall_avg, 1),
            'success_count': len(success_products),
            'failure_count': len(failure_products)
        }
    
    def _extract_common_keywords(self, names: List[str]) -> List[str]:
        """名前から共通キーワードを抽出"""
        keywords = defaultdict(int)
        
        for name in names:
            # 日本語の単語を抽出
            words = re.findall(r'[\u4e00-\u9fff]+', name)
            for word in words:
                if len(word) >= 2:
                    keywords[word] += 1
        
        # 出現回数でソート
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        
        return [k for k, v in sorted_keywords if v >= 2][:5]
    
    def score_concept(self, concept_info: Dict) -> Dict:
        """
        コンセプト評価スコアリング
        
        Args:
            concept_info: コンセプト情報
                - name: 商品名
                - description: 説明
                - target_segments: ターゲット層
                - price: 価格
                - category: カテゴリー
        
        Returns:
            total_score: 総合スコア（0-100）
            dimension_scores: 次元別スコア
            strengths: 強み
            weaknesses: 弱み
        """
        scores = {}
        strengths = []
        weaknesses = []
        
        # 1. 名前の評価（0-20点）
        name = concept_info.get('name', '')
        name_score = 0
        
        # 良いキーワードが含まれているか
        good_keywords = ['金運', '縁結び', '開運', '厄除け', '健康', '合格', '安産', '交通安全']
        for kw in good_keywords:
            if kw in name:
                name_score += 5
                strengths.append(f"「{kw}」という訴求力のあるキーワード")
        
        name_score = min(20, name_score)
        if name_score >= 10:
            strengths.append('商品名が分かりやすい')
        elif name_score < 5:
            weaknesses.append('商品名にご利益が明確でない')
        
        scores['name'] = name_score
        
        # 2. ターゲット層の評価（0-20点）
        targets = concept_info.get('target_segments', [])
        target_score = len(targets) * 5
        target_score = min(20, target_score)
        
        if len(targets) >= 2:
            strengths.append('複数のターゲット層を想定')
        elif len(targets) == 0:
            weaknesses.append('ターゲット層が不明確')
        
        scores['target'] = target_score
        
        # 3. 価格の評価（0-20点）
        price = concept_info.get('price', 0)
        category = concept_info.get('category', 'お守り')
        
        price_ranges = {
            'お守り': (500, 1500),
            '御朱印': (300, 500),
            '御朱印帳': (1500, 3000),
            'おみくじ': (100, 300),
            '絵馬': (500, 1000),
            'お札': (500, 3000),
        }
        
        expected_range = price_ranges.get(category, (500, 2000))
        
        if expected_range[0] <= price <= expected_range[1]:
            price_score = 20
            strengths.append('価格設定が適切')
        elif price < expected_range[0]:
            price_score = 15
            weaknesses.append('価格が安すぎる可能性')
        else:
            price_score = 10
            weaknesses.append('価格が高めの設定')
        
        scores['price'] = price_score
        
        # 4. 説明・コンセプトの評価（0-20点）
        description = concept_info.get('description', '')
        desc_score = 0
        
        if len(description) >= 20:
            desc_score += 10
            strengths.append('コンセプトが明確')
        elif len(description) > 0:
            desc_score += 5
        else:
            weaknesses.append('商品説明がない')
        
        # 季節感やトレンドへの言及
        trend_keywords = ['限定', '新', '特別', '季節', '期間']
        for kw in trend_keywords:
            if kw in description:
                desc_score += 5
                strengths.append(f'「{kw}」という差別化要素')
                break
        
        scores['description'] = min(20, desc_score)
        
        # 5. カテゴリーの市場性（0-20点）
        category_market = {
            'お守り': 18,
            '御朱印': 15,
            'おみくじ': 16,
            '絵馬': 14,
            'お札': 12,
            '御朱印帳': 13,
            '縁起物': 10,
            'その他': 8
        }
        
        scores['market'] = category_market.get(category, 10)
        
        # 総合スコア
        total_score = sum(scores.values())
        
        # 評価ランク
        if total_score >= 80:
            rank = 'A（非常に有望）'
        elif total_score >= 60:
            rank = 'B（有望）'
        elif total_score >= 40:
            rank = 'C（検討の余地あり）'
        else:
            rank = 'D（再検討推奨）'
        
        return {
            'total_score': total_score,
            'rank': rank,
            'dimension_scores': scores,
            'strengths': strengths[:5],
            'weaknesses': weaknesses[:5]
        }


# =============================================================================
# 4. 総合需要予測エンジン
# =============================================================================

class DemandForecastEngine:
    """総合需要予測エンジン"""
    
    def __init__(self, df_sales: pd.DataFrame, df_calendar: Optional[pd.DataFrame] = None):
        self.df_sales = df_sales
        self.df_calendar = df_calendar
        
        # 分析モジュールを初期化
        self.internal = InternalAnalyzer(df_sales)
        self.external = ExternalAnalyzer(df_sales, df_calendar)
        self.market = MarketAnalyzer(df_sales)
    
    def forecast_new_product(self, 
                            product_name: str,
                            category: str,
                            price: int,
                            description: str,
                            target_segments: List[str],
                            similar_products: List[Dict],
                            forecast_days: int = 180,
                            confidence_level: str = '標準') -> Dict:
        """
        新規授与品の総合需要予測
        
        複数の分析結果を統合して予測を生成
        """
        
        # 1. 内部実績分析
        category_perf = self.internal.analyze_category_performance()
        seasonality = self.internal.detect_seasonality()
        
        # 2. 外部環境分析
        calendar_effect = self.external.analyze_calendar_effect()
        weather_effect = self.external.analyze_weather_correlation()
        trends = self.external.fetch_google_trends(product_name)
        
        # 3. 市場・顧客分析
        target_demand = self.market.estimate_target_demand(
            target_segments, category, self._get_category_base_daily(category)
        )
        similar_analysis = self.market.analyze_similar_product_success(similar_products)
        
        concept_score = self.market.score_concept({
            'name': product_name,
            'description': description,
            'target_segments': target_segments,
            'price': price,
            'category': category
        })
        
        # 4. 基本予測値を計算
        if similar_products:
            base_daily = np.mean([p.get('avg_daily', 0) for p in similar_products[:5]])
        else:
            base_daily = self._get_category_base_daily(category)
        
        # 5. 調整係数を適用
        adjustments = {
            'target_multiplier': target_demand['total_multiplier'],
            'concept_multiplier': concept_score['total_score'] / 60,  # 60点を基準
            'trend_multiplier': trends['current_interest'] / 50,  # 50を基準
        }
        
        total_multiplier = np.mean(list(adjustments.values()))
        
        # 信頼度による調整
        confidence_factors = {'楽観的': 1.3, '標準': 1.0, '保守的': 0.7}
        confidence_factor = confidence_factors.get(confidence_level, 1.0)
        
        adjusted_daily = base_daily * total_multiplier * confidence_factor
        
        # 6. 日別予測を生成
        daily_forecast = []
        total_qty = 0
        
        for i in range(forecast_days):
            target_date = date.today() + timedelta(days=i)
            
            # 季節性係数
            month = target_date.month
            weekday = target_date.weekday()
            
            month_factor = seasonality['monthly_pattern'].get(month, 1.0)
            weekday_names = ['月', '火', '水', '木', '金', '土', '日']
            weekday_factor = seasonality['weekday_pattern'].get(weekday_names[weekday], 1.0)
            
            # 日別予測
            pred = adjusted_daily * month_factor * weekday_factor
            pred = max(0.1, pred)
            
            daily_forecast.append({
                'date': target_date,
                'predicted': round(pred)
            })
            
            total_qty += round(pred)
        
        # 7. 信頼区間を計算
        predictions = [d['predicted'] for d in daily_forecast]
        std = np.std(predictions) if predictions else 0
        
        confidence_interval = {
            'lower': max(0, int(total_qty - 1.96 * std * np.sqrt(forecast_days))),
            'upper': int(total_qty + 1.96 * std * np.sqrt(forecast_days))
        }
        
        # 8. 50の倍数に切り上げ
        total_qty_rounded = self._round_up_to_50(total_qty)
        
        return {
            # 予測結果
            'total_qty': total_qty,
            'total_qty_rounded': total_qty_rounded,
            'avg_daily': round(adjusted_daily, 1),
            'forecast_days': forecast_days,
            'confidence_interval': confidence_interval,
            
            # 分析結果
            'analysis': {
                'base_daily': round(base_daily, 1),
                'adjustments': adjustments,
                'total_multiplier': round(total_multiplier, 2),
                'concept_score': concept_score,
                'target_demand': target_demand,
                'similar_analysis': similar_analysis,
                'trends': trends,
                'seasonality': seasonality,
                'calendar_effect': calendar_effect,
                'weather_effect': weather_effect
            },
            
            # 日別予測
            'daily_forecast': daily_forecast,
            
            # メタ情報
            'confidence_level': confidence_level,
            'similar_count': len(similar_products),
            'analysis_quality': self._calculate_analysis_quality(similar_products, concept_score)
        }
    
    def _get_category_base_daily(self, category: str) -> float:
        """カテゴリー別のデフォルト日販"""
        defaults = {
            'お守り': 3.0,
            '御朱印': 5.0,
            '御朱印帳': 1.0,
            'おみくじ': 10.0,
            '絵馬': 2.0,
            'お札': 1.5,
            '縁起物': 1.0,
            'その他': 0.5
        }
        return defaults.get(category, 1.0)
    
    def _round_up_to_50(self, value: int) -> int:
        """50の倍数に切り上げ"""
        if value <= 0:
            return 0
        return ((value + 49) // 50) * 50
    
    def _calculate_analysis_quality(self, similar_products: List, concept_score: Dict) -> str:
        """分析品質を評価"""
        score = 0
        
        # 類似商品の数
        if len(similar_products) >= 5:
            score += 3
        elif len(similar_products) >= 2:
            score += 2
        elif len(similar_products) >= 1:
            score += 1
        
        # コンセプトスコア
        if concept_score['total_score'] >= 60:
            score += 2
        elif concept_score['total_score'] >= 40:
            score += 1
        
        # 評価
        if score >= 4:
            return '高品質 ⭐⭐⭐'
        elif score >= 2:
            return '普通 ⭐⭐'
        else:
            return '参考程度 ⭐'
