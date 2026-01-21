"""
需要予測モジュール

機能:
- Prophet による時系列予測
- 重回帰分析による外部要因考慮
- 納品スケジュール提案
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import config


class DemandForecaster:
    """需要予測クラス"""
    
    def __init__(self):
        self.prophet_model: Optional[Prophet] = None
        self.regression_model: Optional[Ridge] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self.is_fitted = False
    
    def prepare_features(self, df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """特徴量を準備"""
        # 日付でマージ
        if not df.empty and not calendar_df.empty:
            merged = df.merge(calendar_df, on='date', how='left')
        else:
            merged = df.copy()
        
        # 追加の特徴量を作成
        if 'date' in merged.columns:
            merged['year'] = merged['date'].dt.year
            merged['month'] = merged['date'].dt.month
            merged['day'] = merged['date'].dt.day
            merged['day_of_year'] = merged['date'].dt.dayofyear
            
            # 曜日（0=月曜）
            if 'weekday_num' not in merged.columns:
                merged['weekday_num'] = merged['date'].dt.weekday
            
            # 月末フラグ
            merged['is_month_end'] = merged['date'].dt.is_month_end
            
            # 月初フラグ
            merged['is_month_start'] = merged['date'].dt.is_month_start
            
            # 正月フラグ（1/1-1/3）
            merged['is_new_year'] = (
                (merged['month'] == 1) & (merged['day'] <= 3)
            )
            
            # お盆フラグ（8/13-8/16）
            merged['is_obon'] = (
                (merged['month'] == 8) & 
                (merged['day'] >= 13) & 
                (merged['day'] <= 16)
            )
            
            # GWフラグ（5/3-5/5）
            merged['is_gw'] = (
                (merged['month'] == 5) & 
                (merged['day'] >= 3) & 
                (merged['day'] <= 5)
            )
            
            # 七五三シーズン（11月）
            merged['is_shichigosan'] = (merged['month'] == 11)
            
            # ラグ特徴量
            if '販売商品数' in merged.columns:
                merged = merged.sort_values('date')
                merged['lag_7'] = merged['販売商品数'].shift(7)
                merged['lag_14'] = merged['販売商品数'].shift(14)
                merged['lag_30'] = merged['販売商品数'].shift(30)
                merged['lag_365'] = merged['販売商品数'].shift(365)
                
                # 移動平均
                merged['rolling_mean_7'] = merged['販売商品数'].rolling(7).mean()
                merged['rolling_mean_30'] = merged['販売商品数'].rolling(30).mean()
                
                # 前年同期比
                merged['yoy_ratio'] = merged['販売商品数'] / merged['lag_365'].replace(0, np.nan)
        
        return merged
    
    def fit_prophet(self, df: pd.DataFrame, target_col: str = '販売商品数') -> Dict[str, Any]:
        """Prophetモデルを学習"""
        if df.empty or target_col not in df.columns:
            return {'success': False, 'error': 'データが不足しています'}
        
        # Prophet用のデータ準備
        prophet_df = df[['date', target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 30:
            return {'success': False, 'error': 'データが30日分未満です'}
        
        # モデル構築
        self.prophet_model = Prophet(
            seasonality_mode=config.FORECAST_PARAMS['seasonality_mode'],
            yearly_seasonality=config.FORECAST_PARAMS['yearly_seasonality'],
            weekly_seasonality=config.FORECAST_PARAMS['weekly_seasonality'],
            daily_seasonality=config.FORECAST_PARAMS['daily_seasonality'],
        )
        
        # 日本の祝日を追加
        self.prophet_model.add_country_holidays(country_name='JP')
        
        # 学習
        self.prophet_model.fit(prophet_df)
        self.is_fitted = True
        
        return {'success': True}
    
    def fit_regression(self, df: pd.DataFrame, target_col: str = '販売商品数') -> Dict[str, Any]:
        """重回帰モデルを学習"""
        if df.empty or target_col not in df.columns:
            return {'success': False, 'error': 'データが不足しています'}
        
        # 特徴量列を定義
        self.feature_columns = [
            'weekday_num', 'month', 'day', 'day_of_year',
            'is_weekend', 'is_holiday', 'is_taian', 
            'is_ichiryumanbai', 'is_tensha', 'is_tora_no_hi', 'is_mi_no_hi',
            'is_new_year', 'is_obon', 'is_gw', 'is_shichigosan',
            'is_month_end', 'is_month_start',
        ]
        
        # 天気データがあれば追加
        if 'temp_mean' in df.columns:
            self.feature_columns.extend(['temp_max', 'temp_min', 'temp_mean', 'precipitation'])
        
        # ラグ特徴量があれば追加
        lag_cols = ['lag_7', 'lag_14', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']
        for col in lag_cols:
            if col in df.columns:
                self.feature_columns.append(col)
        
        # 存在する列のみ使用
        available_features = [c for c in self.feature_columns if c in df.columns]
        
        if not available_features:
            return {'success': False, 'error': '特徴量がありません'}
        
        # データ準備
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        # Boolean を数値に変換
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
        
        # 欠損値を除去
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 30:
            return {'success': False, 'error': 'データが30日分未満です'}
        
        # スケーリング
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # モデル学習
        self.regression_model = Ridge(alpha=1.0)
        self.regression_model.fit(X_scaled, y)
        
        # 特徴量重要度
        feature_importance = dict(zip(available_features, self.regression_model.coef_))
        
        self.feature_columns = available_features
        
        return {
            'success': True,
            'feature_importance': feature_importance,
            'r2_score': self.regression_model.score(X_scaled, y),
        }
    
    def predict(self, periods: int = 365, 
                future_calendar: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """需要を予測"""
        if not self.is_fitted or self.prophet_model is None:
            return pd.DataFrame()
        
        # 将来の日付を生成
        future = self.prophet_model.make_future_dataframe(periods=periods)
        
        # Prophet予測
        forecast = self.prophet_model.predict(future)
        
        # 結果を整形
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result.columns = ['date', 'predicted', 'lower', 'upper']
        
        # 負の値を0に
        result['predicted'] = result['predicted'].clip(lower=0)
        result['lower'] = result['lower'].clip(lower=0)
        result['upper'] = result['upper'].clip(lower=0)
        
        # 整数に丸める
        result['predicted'] = result['predicted'].round().astype(int)
        result['lower'] = result['lower'].round().astype(int)
        result['upper'] = result['upper'].round().astype(int)
        
        return result
    
    def get_seasonality_components(self) -> Dict[str, pd.DataFrame]:
        """季節性成分を取得"""
        if not self.is_fitted or self.prophet_model is None:
            return {}
        
        components = {}
        
        # 週次パターン
        weekly = self.prophet_model.plot_components_plotly
        
        return components


class DeliveryPlanner:
    """納品計画クラス"""
    
    def __init__(self, 
                 lead_time: int = None,
                 safety_stock_days: int = None,
                 order_lot: int = None):
        """
        Args:
            lead_time: リードタイム（日）
            safety_stock_days: 安全在庫日数
            order_lot: 発注ロット単位
        """
        self.lead_time = lead_time or config.DELIVERY_PARAMS['default_lead_time']
        self.safety_stock_days = safety_stock_days or config.DELIVERY_PARAMS['default_safety_stock_days']
        self.order_lot = order_lot or config.DELIVERY_PARAMS['default_order_lot']
    
    def calculate_safety_stock(self, forecast_df: pd.DataFrame) -> int:
        """安全在庫を計算"""
        if forecast_df.empty:
            return 0
        
        # 平均日販の safety_stock_days 日分
        avg_daily = forecast_df['predicted'].mean()
        safety_stock = int(avg_daily * self.safety_stock_days)
        
        # ロット単位に切り上げ
        safety_stock = ((safety_stock + self.order_lot - 1) // self.order_lot) * self.order_lot
        
        return safety_stock
    
    def simulate_inventory(self, forecast_df: pd.DataFrame, 
                          initial_stock: int,
                          delivery_schedule: List[Dict]) -> pd.DataFrame:
        """
        在庫シミュレーション
        
        Args:
            forecast_df: 予測データ
            initial_stock: 初期在庫
            delivery_schedule: 納品スケジュール [{'date': date, 'quantity': int}, ...]
        """
        if forecast_df.empty:
            return pd.DataFrame()
        
        # 納品を日付でインデックス化
        deliveries = {d['date']: d['quantity'] for d in delivery_schedule}
        
        # シミュレーション
        simulation = []
        stock = initial_stock
        
        for _, row in forecast_df.iterrows():
            current_date = row['date']
            demand = row['predicted']
            
            # 納品があれば加算
            if current_date in deliveries:
                stock += deliveries[current_date]
            
            # 需要を消費
            stock -= demand
            
            simulation.append({
                'date': current_date,
                'demand': demand,
                'delivery': deliveries.get(current_date, 0),
                'stock': max(0, stock),
                'stockout': stock < 0,
            })
        
        return pd.DataFrame(simulation)
    
    def generate_delivery_plan(self, forecast_df: pd.DataFrame,
                               initial_stock: int,
                               min_stock: int = None) -> List[Dict]:
        """
        最適な納品計画を生成
        
        Args:
            forecast_df: 予測データ
            initial_stock: 初期在庫
            min_stock: 最小在庫（これを下回ったら発注）
        """
        if forecast_df.empty:
            return []
        
        # 安全在庫
        if min_stock is None:
            min_stock = self.calculate_safety_stock(forecast_df)
        
        # 年間予測販売数
        total_demand = forecast_df['predicted'].sum()
        
        # 納品計画を生成
        deliveries = []
        stock = initial_stock
        forecast_df = forecast_df.sort_values('date').reset_index(drop=True)
        
        for i, row in forecast_df.iterrows():
            current_date = row['date']
            demand = row['predicted']
            
            # 在庫を消費
            stock -= demand
            
            # リードタイム後の在庫予測
            future_demand = 0
            for j in range(i, min(i + self.lead_time, len(forecast_df))):
                future_demand += forecast_df.iloc[j]['predicted']
            
            # 発注点に達したら発注
            if stock - future_demand < min_stock:
                # 発注量を計算（次の大きなイベントまでをカバー）
                # デフォルトは3ヶ月分
                cover_days = 90
                cover_demand = 0
                for j in range(i, min(i + cover_days, len(forecast_df))):
                    cover_demand += forecast_df.iloc[j]['predicted']
                
                # 発注量
                order_qty = cover_demand + min_stock - stock
                order_qty = max(order_qty, 0)
                
                # ロット単位に切り上げ
                order_qty = ((order_qty + self.order_lot - 1) // self.order_lot) * self.order_lot
                
                if order_qty > 0:
                    delivery_date = current_date + timedelta(days=self.lead_time)
                    deliveries.append({
                        'order_date': current_date,
                        'delivery_date': delivery_date,
                        'quantity': order_qty,
                        'reason': self._get_delivery_reason(delivery_date),
                    })
                    stock += order_qty
        
        return deliveries
    
    def _get_delivery_reason(self, delivery_date: date) -> str:
        """納品理由を取得"""
        if isinstance(delivery_date, pd.Timestamp):
            delivery_date = delivery_date.date()
        
        month = delivery_date.month
        
        if month == 12:
            return "正月準備"
        elif month in [4, 5]:
            return "GW準備"
        elif month in [7, 8]:
            return "お盆準備"
        elif month in [10, 11]:
            return "七五三準備"
        else:
            return "通常補充"
    
    def optimize_annual_plan(self, forecast_df: pd.DataFrame,
                            initial_stock: int,
                            total_order_quantity: int) -> List[Dict]:
        """
        年間納品計画を最適化
        
        Args:
            forecast_df: 予測データ
            initial_stock: 初期在庫
            total_order_quantity: 年間発注総量
        """
        if forecast_df.empty:
            return []
        
        # キーポイント（大量消費が予想される時期の前）
        key_points = [
            {'month': 12, 'day': 1, 'label': '正月準備', 'weight': 0.35},
            {'month': 4, 'day': 15, 'label': 'GW準備', 'weight': 0.15},
            {'month': 7, 'day': 15, 'label': 'お盆準備', 'weight': 0.20},
            {'month': 10, 'day': 15, 'label': '七五三準備', 'weight': 0.20},
            {'month': 2, 'day': 1, 'label': '通常補充', 'weight': 0.10},
        ]
        
        # 各キーポイントに割り当て
        deliveries = []
        year = forecast_df['date'].dt.year.mode()[0]
        
        for kp in key_points:
            delivery_date = datetime(year, kp['month'], kp['day']).date()
            quantity = int(total_order_quantity * kp['weight'])
            
            # ロット単位に丸める
            quantity = (quantity // self.order_lot) * self.order_lot
            
            if quantity > 0:
                deliveries.append({
                    'delivery_date': delivery_date,
                    'quantity': quantity,
                    'reason': kp['label'],
                })
        
        # 残りを調整
        total_planned = sum(d['quantity'] for d in deliveries)
        if total_planned < total_order_quantity:
            diff = total_order_quantity - total_planned
            # 正月準備に追加
            for d in deliveries:
                if d['reason'] == '正月準備':
                    d['quantity'] += ((diff + self.order_lot - 1) // self.order_lot) * self.order_lot
                    break
        
        return sorted(deliveries, key=lambda x: x['delivery_date'])
