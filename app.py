"""
Airレジ 売上分析・需要予測 Webアプリ（v12: Vertex AI AutoML Forecasting 完全統合版）

v11からの変更点:
1. google.generativeai → google.cloud.aiplatform に変更
2. APIキー認証 → サービスアカウントJSON認証 に変更
3. 統計ベース予測 → Vertex AI AutoML Forecastingエンドポイント呼び出し
4. 共変量（天気、六曜、イベント等）対応
5. エラーハンドリング強化（API制限、接続エラー対応）

v11からの維持機能:
- 複数授与品選択時に「合算」「個別」を選択可能
- 予測期間を「日数指定」「期間指定」で選択可能
- 新規授与品の需要予測（類似商品ベース）
- 予測精度ダッシュボード
- 高度な分析タブ
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import calendar
import re
import os
import json
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# モジュールのインポート
import sys
sys.path.append('.')
from modules.data_loader import SheetsDataLoader, aggregate_by_products, merge_with_calendar
from modules.product_normalizer import ProductNormalizer
import config

# 高度な分析モジュール（オプショナル）
try:
    from modules.demand_analyzer import InternalAnalyzer, ExternalAnalyzer, MarketAnalyzer, DemandForecastEngine
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False


# =============================================================================
# Vertex AI AutoML Forecasting 統合
# =============================================================================

# Vertex AI設定（config.pyまたは環境変数から読み込み）
VERTEX_AI_CONFIG = {
    'project_id': getattr(config, 'VERTEX_AI_PROJECT_ID', os.environ.get('VERTEX_AI_PROJECT_ID', '')),
    'location': getattr(config, 'VERTEX_AI_LOCATION', os.environ.get('VERTEX_AI_LOCATION', 'asia-northeast1')),
    'endpoint_id': getattr(config, 'VERTEX_AI_ENDPOINT_ID', os.environ.get('VERTEX_AI_ENDPOINT_ID', '')),
    'service_account_file': getattr(config, 'VERTEX_AI_SERVICE_ACCOUNT_FILE', 
                                     os.environ.get('VERTEX_AI_SERVICE_ACCOUNT_FILE', 'service_account.json')),
}

# Vertex AI利用可能フラグ
VERTEX_AI_AVAILABLE = False
aiplatform = None
prediction_service_client = None

try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic.schema import predict as predict_schema
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
    from google.oauth2 import service_account
    from google.api_core import exceptions as google_exceptions
    
    # サービスアカウント認証
    if os.path.exists(VERTEX_AI_CONFIG['service_account_file']):
        credentials = service_account.Credentials.from_service_account_file(
            VERTEX_AI_CONFIG['service_account_file'],
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Vertex AI初期化
        if VERTEX_AI_CONFIG['project_id'] and VERTEX_AI_CONFIG['endpoint_id']:
            aiplatform.init(
                project=VERTEX_AI_CONFIG['project_id'],
                location=VERTEX_AI_CONFIG['location'],
                credentials=credentials
            )
            VERTEX_AI_AVAILABLE = True
            logger.info("Vertex AI AutoML Forecasting: 初期化成功")
        else:
            logger.warning("Vertex AI: project_idまたはendpoint_idが設定されていません")
    else:
        logger.warning(f"Vertex AI: サービスアカウントファイルが見つかりません: {VERTEX_AI_CONFIG['service_account_file']}")
        
except ImportError as e:
    logger.warning(f"Vertex AI SDKがインストールされていません: {e}")
except Exception as e:
    logger.error(f"Vertex AI初期化エラー: {e}")


class VertexAIForecaster:
    """Vertex AI AutoML Forecastingエンドポイントを呼び出すクラス"""
    
    def __init__(self):
        self.project_id = VERTEX_AI_CONFIG['project_id']
        self.location = VERTEX_AI_CONFIG['location']
        self.endpoint_id = VERTEX_AI_CONFIG['endpoint_id']
        self.endpoint_name = f"projects/{self.project_id}/locations/{self.location}/endpoints/{self.endpoint_id}"
        self._client = None
    
    @property
    def client(self):
        """Prediction Service Clientを取得（遅延初期化）"""
        if self._client is None:
            from google.cloud.aiplatform_v1.services.prediction_service import PredictionServiceClient
            from google.cloud.aiplatform_v1.types import PredictRequest
            
            client_options = {"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
            credentials = service_account.Credentials.from_service_account_file(
                VERTEX_AI_CONFIG['service_account_file'],
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            self._client = PredictionServiceClient(
                credentials=credentials,
                client_options=client_options
            )
        return self._client
    
    def prepare_forecast_instances(
        self,
        historical_data: pd.DataFrame,
        forecast_horizon: int,
        product_id: str,
        covariates: Optional[Dict[str, List]] = None
    ) -> List[Dict[str, Any]]:
        """
        Vertex AI Forecasting APIが期待するインスタンス形式を準備
        
        Args:
            historical_data: 過去の売上データ（date, 販売商品数）
            forecast_horizon: 予測日数
            product_id: 商品識別子
            covariates: 将来利用可能な共変量（天気、六曜、イベント等）
        
        Returns:
            APIリクエスト用のインスタンスリスト
        """
        df = historical_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 時系列データの準備
        time_series = []
        for _, row in df.iterrows():
            time_series.append({
                'timestamp': row['date'].strftime('%Y-%m-%dT00:00:00Z'),
                'target': float(row['販売商品数'])
            })
        
        # 予測期間の準備
        last_date = df['date'].max()
        forecast_timestamps = []
        for i in range(1, forecast_horizon + 1):
            future_date = last_date + timedelta(days=i)
            forecast_timestamps.append(future_date.strftime('%Y-%m-%dT00:00:00Z'))
        
        # インスタンス構造の構築
        instance = {
            'time_series_identifier': product_id,
            'time_column': 'timestamp',
            'target_column': 'target',
            'historical_data': time_series,
            'forecast_horizon': forecast_horizon,
            'forecast_timestamps': forecast_timestamps,
        }
        
        # 共変量の追加（天気、六曜、イベント等）
        if covariates:
            instance['available_at_forecast_columns'] = list(covariates.keys())
            
            # 過去データの共変量
            if 'historical_covariates' in covariates:
                instance['historical_covariates'] = covariates['historical_covariates']
            
            # 将来データの共変量
            if 'future_covariates' in covariates:
                instance['future_covariates'] = covariates['future_covariates']
        
        return [instance]
    
    def predict(
        self,
        historical_data: pd.DataFrame,
        forecast_horizon: int,
        product_id: str = "default",
        covariates: Optional[Dict[str, List]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Vertex AI AutoML Forecastingエンドポイントに予測リクエストを送信
        
        Args:
            historical_data: 過去の売上データ
            forecast_horizon: 予測日数
            product_id: 商品識別子
            covariates: 共変量データ
        
        Returns:
            予測結果のDataFrameとメタデータ
        """
        if not VERTEX_AI_AVAILABLE:
            raise RuntimeError("Vertex AIが利用できません。設定を確認してください。")
        
        try:
            # インスタンス準備
            instances = self.prepare_forecast_instances(
                historical_data, forecast_horizon, product_id, covariates
            )
            
            # Protobuf形式に変換
            instances_pb = [json_format.ParseDict(inst, Value()) for inst in instances]
            
            # 予測リクエスト送信
            response = self.client.predict(
                endpoint=self.endpoint_name,
                instances=instances_pb,
            )
            
            # レスポンス解析
            predictions = []
            metadata = {
                'model_version': getattr(response, 'model_version_id', 'unknown'),
                'deployed_model_id': getattr(response, 'deployed_model_id', 'unknown'),
            }
            
            last_date = pd.to_datetime(historical_data['date']).max()
            
            for i, prediction in enumerate(response.predictions):
                pred_dict = json_format.MessageToDict(prediction)
                
                # 予測値の取得（AutoML Forecastingのレスポンス形式に応じて調整）
                if 'value' in pred_dict:
                    pred_value = pred_dict['value']
                elif 'predicted_target' in pred_dict:
                    pred_value = pred_dict['predicted_target']
                else:
                    # フォールバック: リスト形式の場合
                    pred_value = list(pred_dict.values())[0] if pred_dict else 0
                
                # 予測値が配列の場合の処理
                if isinstance(pred_value, list):
                    for j, val in enumerate(pred_value):
                        predictions.append({
                            'date': last_date + timedelta(days=j+1),
                            'predicted': max(0, round(float(val))),
                            'confidence_lower': pred_dict.get('lower_bound', [None])[j] if isinstance(pred_dict.get('lower_bound'), list) else None,
                            'confidence_upper': pred_dict.get('upper_bound', [None])[j] if isinstance(pred_dict.get('upper_bound'), list) else None,
                        })
                else:
                    predictions.append({
                        'date': last_date + timedelta(days=i+1),
                        'predicted': max(0, round(float(pred_value))),
                    })
            
            return pd.DataFrame(predictions), metadata
            
        except google_exceptions.ResourceExhausted as e:
            logger.error(f"Vertex AI クォータ制限: {e}")
            raise RuntimeError(f"APIクォータ制限に達しました。しばらく待ってから再試行してください。\n詳細: {e}")
        
        except google_exceptions.InvalidArgument as e:
            logger.error(f"Vertex AI リクエストエラー: {e}")
            raise RuntimeError(f"リクエスト形式が不正です。\n詳細: {e}")
        
        except google_exceptions.NotFound as e:
            logger.error(f"Vertex AI エンドポイント未発見: {e}")
            raise RuntimeError(f"指定されたエンドポイントが見つかりません。endpoint_idを確認してください。\n詳細: {e}")
        
        except google_exceptions.PermissionDenied as e:
            logger.error(f"Vertex AI 権限エラー: {e}")
            raise RuntimeError(f"アクセス権限がありません。サービスアカウントの権限を確認してください。\n詳細: {e}")
        
        except Exception as e:
            logger.error(f"Vertex AI 予測エラー: {e}")
            raise RuntimeError(f"予測中にエラーが発生しました。\n詳細: {e}")


# Vertex AIフォアキャスターのシングルトンインスタンス
_vertex_ai_forecaster = None

def get_vertex_ai_forecaster() -> Optional[VertexAIForecaster]:
    """Vertex AIフォアキャスターを取得"""
    global _vertex_ai_forecaster
    if VERTEX_AI_AVAILABLE and _vertex_ai_forecaster is None:
        _vertex_ai_forecaster = VertexAIForecaster()
    return _vertex_ai_forecaster


# =============================================================================
# 共変量データ生成（天気、六曜、イベント）
# =============================================================================

def generate_covariates(start_date: date, end_date: date, location: str = "hitachinaka") -> Dict[str, List]:
    """
    将来利用可能な共変量データを生成
    
    Args:
        start_date: 開始日
        end_date: 終了日
        location: 地域（天気予報用）
    
    Returns:
        共変量データの辞書
    """
    covariates = {
        'future_covariates': []
    }
    
    current_date = start_date
    while current_date <= end_date:
        covariate_entry = {
            'timestamp': current_date.strftime('%Y-%m-%dT00:00:00Z'),
            'weekday': current_date.weekday(),  # 0=月曜, 6=日曜
            'is_weekend': 1 if current_date.weekday() >= 5 else 0,
            'month': current_date.month,
            'day_of_month': current_date.day,
        }
        
        # 六曜（簡易計算）
        rokuyou_list = ['大安', '赤口', '先勝', '友引', '先負', '仏滅']
        rokuyou_idx = (current_date.year + current_date.month + current_date.day) % 6
        covariate_entry['rokuyou'] = rokuyou_idx
        covariate_entry['is_taian'] = 1 if rokuyou_list[rokuyou_idx] == '大安' else 0
        
        # 特別期間フラグ
        covariate_entry['is_new_year'] = 1 if (current_date.month == 1 and current_date.day <= 7) else 0
        covariate_entry['is_obon'] = 1 if (current_date.month == 8 and 13 <= current_date.day <= 16) else 0
        covariate_entry['is_shichigosan'] = 1 if (current_date.month == 11 and 10 <= current_date.day <= 20) else 0
        covariate_entry['is_golden_week'] = 1 if (current_date.month == 5 and 3 <= current_date.day <= 5) else 0
        
        covariates['future_covariates'].append(covariate_entry)
        current_date += timedelta(days=1)
    
    return covariates


# =============================================================================
# 予測関数（Vertex AI + フォールバック）
# =============================================================================

def get_vertex_ai_prediction(
    df: pd.DataFrame,
    periods: int,
    product_id: str = "default",
    use_covariates: bool = True
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Vertex AI AutoML Forecastingによる予測（フォールバック付き）
    
    Args:
        df: 売上データ（date, 販売商品数）
        periods: 予測日数
        product_id: 商品識別子
        use_covariates: 共変量を使用するか
    
    Returns:
        予測DataFrame, Vertex AI使用フラグ, メッセージ
    """
    forecaster = get_vertex_ai_forecaster()
    
    if forecaster is None:
        # Vertex AIが利用不可の場合はフォールバック
        return forecast_with_seasonality_fallback(df, periods), False, "Vertex AI未設定のため、統計モデルで予測"
    
    try:
        # 共変量の準備
        covariates = None
        if use_covariates:
            last_date = pd.to_datetime(df['date']).max()
            start_date = (last_date + timedelta(days=1)).date()
            end_date = (last_date + timedelta(days=periods)).date()
            covariates = generate_covariates(start_date, end_date)
        
        # Vertex AI予測
        predictions, metadata = forecaster.predict(
            historical_data=df,
            forecast_horizon=periods,
            product_id=product_id,
            covariates=covariates
        )
        
        return predictions, True, f"Vertex AI AutoML Forecasting (モデル: {metadata.get('deployed_model_id', 'N/A')})"
        
    except RuntimeError as e:
        # エラー時はフォールバック
        logger.warning(f"Vertex AI予測失敗、フォールバック実行: {e}")
        return forecast_with_seasonality_fallback(df, periods), False, f"Vertex AIエラー: {str(e)[:100]}... 統計モデルで予測"
    except Exception as e:
        logger.error(f"予測エラー: {e}")
        return forecast_with_seasonality_fallback(df, periods), False, f"エラー: {str(e)[:100]}... 統計モデルで予測"


def forecast_with_seasonality_fallback(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    フォールバック用の季節性考慮予測（統計ベース）
    
    Vertex AIが利用できない場合に使用
    """
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
        
        # 特別期間の調整
        special_factor = 1.0
        if d.month == 1 and d.day <= 7:  # 正月
            special_factor = 3.0
        elif d.month == 8 and 13 <= d.day <= 16:  # お盆
            special_factor = 1.5
        elif d.month == 11 and 10 <= d.day <= 20:  # 七五三
            special_factor = 1.3
        
        pred = overall_mean * weekday_f * month_f * special_factor
        pred = max(0.1, pred)
        
        predictions.append({
            'date': d,
            'predicted': round(pred)
        })
    
    return pd.DataFrame(predictions)


# =============================================================================
# 予測方法の統合（Vertex AI対応）
# =============================================================================

def forecast_with_vertex_ai(
    df: pd.DataFrame,
    periods: int,
    method: str = "Vertex AI",
    product_id: str = "default"
) -> Tuple[pd.DataFrame, str]:
    """
    予測方法に応じた予測を実行
    
    Args:
        df: 売上データ
        periods: 予測日数
        method: 予測方法
        product_id: 商品識別子
    
    Returns:
        予測DataFrame, 使用した予測方法の説明
    """
    if method == "🚀 Vertex AI（推奨）":
        predictions, used_vertex_ai, message = get_vertex_ai_prediction(df, periods, product_id, use_covariates=True)
        return predictions, message
    
    elif method == "移動平均法（シンプル）":
        return forecast_moving_average(df, periods), "移動平均法（統計モデル）"
    
    elif method == "季節性考慮（統計）":
        return forecast_with_seasonality_fallback(df, periods), "季節性考慮（統計モデル）"
    
    elif method == "指数平滑法":
        return forecast_exponential_smoothing(df, periods), "指数平滑法（統計モデル）"
    
    else:
        # デフォルトはVertex AI
        predictions, used_vertex_ai, message = get_vertex_ai_prediction(df, periods, product_id, use_covariates=True)
        return predictions, message


def forecast_moving_average(df: pd.DataFrame, periods: int, window: int = 30) -> pd.DataFrame:
    """移動平均法による予測"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    recent_data = df.tail(window)
    base_mean = recent_data['販売商品数'].mean()
    
    if pd.isna(base_mean) or base_mean <= 0:
        base_mean = 1.0
    
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    predictions = []
    for d in future_dates:
        pred = max(0.1, base_mean)
        predictions.append({
            'date': d,
            'predicted': round(pred)
        })
    
    return pd.DataFrame(predictions)


def forecast_exponential_smoothing(df: pd.DataFrame, periods: int, alpha: float = 0.3) -> pd.DataFrame:
    """指数平滑法による予測"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    values = df['販売商品数'].values
    
    if len(values) == 0:
        return pd.DataFrame({'date': [], 'predicted': []})
    
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed_value = alpha * values[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(smoothed_value)
    
    base_prediction = smoothed[-1] if smoothed else 1.0
    
    if pd.isna(base_prediction) or base_prediction <= 0:
        base_prediction = 1.0
    
    if len(smoothed) >= 7:
        recent_trend = (smoothed[-1] - smoothed[-7]) / 7
    else:
        recent_trend = 0
    
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    predictions = []
    for i, d in enumerate(future_dates):
        decay_factor = 0.95 ** i
        pred = base_prediction + (recent_trend * i * decay_factor)
        pred = max(0.1, pred)
        
        predictions.append({
            'date': d,
            'predicted': round(pred)
        })
    
    return pd.DataFrame(predictions)


def forecast_all_methods_with_vertex_ai(df: pd.DataFrame, periods: int, product_id: str = "default") -> Dict[str, Tuple[pd.DataFrame, str]]:
    """
    すべての予測方法で予測を実行（Vertex AI含む）
    """
    results = {}
    
    # Vertex AI予測
    if VERTEX_AI_AVAILABLE:
        predictions, used_vertex_ai, message = get_vertex_ai_prediction(df, periods, product_id)
        results['Vertex AI'] = (predictions, message)
    
    # 統計モデル予測
    results['季節性考慮'] = (forecast_with_seasonality_fallback(df, periods), "季節性考慮（統計モデル）")
    results['移動平均法'] = (forecast_moving_average(df, periods), "移動平均法（統計モデル）")
    results['指数平滑法'] = (forecast_exponential_smoothing(df, periods), "指数平滑法（統計モデル）")
    
    return results


# =============================================================================
# 予測方法の定義（v12更新）
# =============================================================================

FORECAST_METHODS = {
    "🚀 Vertex AI（推奨）": {
        "description": "Google Cloud AutoML Forecastingによる高精度予測。天気・六曜・イベントを考慮。",
        "icon": "🚀",
        "color": "#4285F4",
        "requires_vertex_ai": True
    },
    "季節性考慮（統計）": {
        "description": "月別・曜日別の傾向と特別期間を考慮した統計モデル。Vertex AI未設定時の推奨。",
        "icon": "📈",
        "color": "#4CAF50",
        "requires_vertex_ai": False
    },
    "移動平均法（シンプル）": {
        "description": "過去30日間の平均値をベースに予測。安定した商品向け。",
        "icon": "📊",
        "color": "#1E88E5",
        "requires_vertex_ai": False
    },
    "指数平滑法": {
        "description": "直近のデータを重視した予測。トレンドの変化に敏感。",
        "icon": "📉",
        "color": "#FF9800",
        "requires_vertex_ai": False
    },
    "🔄 すべての方法で比較": {
        "description": "Vertex AIと統計モデルすべてで予測し、結果を比較します。",
        "icon": "🔄",
        "color": "#9C27B0",
        "requires_vertex_ai": False
    }
}

# カテゴリー別の特性（新規授与品予測用）
CATEGORY_CHARACTERISTICS = {
    "お守り": {"seasonality": "high", "base_daily": 3.0, "price_range": (500, 1500)},
    "御朱印": {"seasonality": "medium", "base_daily": 5.0, "price_range": (300, 500)},
    "御朱印帳": {"seasonality": "low", "base_daily": 1.0, "price_range": (1500, 3000)},
    "おみくじ": {"seasonality": "high", "base_daily": 10.0, "price_range": (100, 300)},
    "絵馬": {"seasonality": "high", "base_daily": 2.0, "price_range": (500, 1000)},
    "お札": {"seasonality": "high", "base_daily": 1.5, "price_range": (500, 3000)},
    "縁起物": {"seasonality": "medium", "base_daily": 1.0, "price_range": (500, 5000)},
    "その他": {"seasonality": "low", "base_daily": 0.5, "price_range": (500, 2000)},
}


# =============================================================================
# ページ設定
# =============================================================================

st.set_page_config(
    page_title="Airレジ 売上分析（Vertex AI版）",
    page_icon="⛩️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# カスタムCSS
st.markdown("""
<style>
    /* ============================================
       基本スタイル
       ============================================ */
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
    .accuracy-good { color: #4CAF50; font-weight: bold; }
    .accuracy-medium { color: #FF9800; font-weight: bold; }
    .accuracy-poor { color: #F44336; font-weight: bold; }
    .new-product-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 10px 0;
    }
    .analysis-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
    }
    .method-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid;
    }
    .method-vertex-ai { border-left-color: #4285F4; background: #e8f0fe; }
    .method-seasonality { border-left-color: #4CAF50; }
    .method-moving-avg { border-left-color: #1E88E5; }
    .method-exponential { border-left-color: #FF9800; }
    .vertex-ai-status {
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .vertex-ai-available {
        background: #e8f5e9;
        border: 1px solid #4CAF50;
        color: #2e7d32;
    }
    .vertex-ai-unavailable {
        background: #fff3e0;
        border: 1px solid #FF9800;
        color: #e65100;
    }
    .individual-product-box {
        background: #f0f8ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #1E88E5;
    }
    
    /* ============================================
       グローバル設定（横スクロール防止）
       ============================================ */
    .main .block-container {
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
        overflow-x: hidden;
    }
    
    /* データフレームの横スクロール対応 */
    [data-testid="stDataFrame"] {
        width: 100%;
    }
    [data-testid="stDataFrame"] > div {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    
    /* ============================================
       スマホ対応（768px以下）
       ============================================ */
    @media screen and (max-width: 768px) {
        /* ヘッダー */
        .main-header {
            font-size: 1.4rem;
            text-align: center;
        }
        .section-header {
            font-size: 1.1rem;
        }
        
        /* カラムを縦並びに */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        
        /* メトリクス（数値表示）をコンパクトに */
        [data-testid="metric-container"] {
            padding: 8px 5px;
            background: #f8f9fa;
            border-radius: 8px;
            margin: 4px 0;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        
        /* メトリクスを2列表示に */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: 8px;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 45% !important;
            min-width: 45% !important;
            max-width: 48% !important;
        }
        
        /* タブ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.75rem;
            padding: 8px 10px;
            white-space: nowrap;
        }
        
        /* ボタン */
        .stButton > button {
            width: 100%;
            padding: 12px 16px;
            font-size: 0.9rem;
        }
        
        /* 入力フィールド */
        .stSelectbox, .stNumberInput, .stTextInput {
            margin-bottom: 8px;
        }
        .stSelectbox label, .stNumberInput label, .stTextInput label {
            font-size: 0.8rem;
        }
        
        /* ラジオボタンを縦並びに */
        [data-testid="stRadio"] > div {
            flex-direction: column;
            gap: 8px;
        }
        [data-testid="stRadio"] label {
            font-size: 0.85rem;
        }
        
        /* カード */
        .analysis-card, .method-card {
            padding: 10px;
            font-size: 0.85rem;
        }
        .new-product-card {
            padding: 15px;
        }
        .new-product-card h2 {
            font-size: 1.1rem;
        }
        .new-product-card p {
            font-size: 0.85rem;
        }
        
        /* グラフ */
        .js-plotly-plot {
            margin: 0 -10px;
        }
        .js-plotly-plot .plotly .modebar {
            display: none !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
            padding: 10px;
        }
        
        /* Info/Warning/Errorボックス */
        [data-testid="stAlert"] {
            padding: 10px;
            font-size: 0.85rem;
        }
        
        /* Divider */
        hr {
            margin: 15px 0;
        }
        
        /* 選択中の授与品 */
        .product-tag {
            font-size: 0.8rem;
            padding: 4px 10px;
        }
    }
    
    /* ============================================
       タブレット対応（769px〜1024px）
       ============================================ */
    @media screen and (min-width: 769px) and (max-width: 1024px) {
        .main-header {
            font-size: 2rem;
        }
        [data-testid="column"] {
            min-width: 45% !important;
        }
    }
    
    /* ============================================
       選択中の授与品の削除ボタン
       ============================================ */
    .product-tag {
        display: inline-flex;
        align-items: center;
        background: #e3f2fd;
        border-radius: 20px;
        padding: 5px 12px;
        margin: 3px;
        font-size: 0.9rem;
    }
    .product-tag-remove {
        margin-left: 8px;
        cursor: pointer;
        color: #666;
        font-weight: bold;
    }
    .product-tag-remove:hover {
        color: #f44336;
    }
    
    /* ============================================
       パフォーマンス最適化
       ============================================ */
    /* アニメーションを軽量化 */
    * {
        -webkit-tap-highlight-color: transparent;
    }
    .stApp {
        -webkit-font-smoothing: antialiased;
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
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = {}
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "合算"
if 'individual_sales_data' not in st.session_state:
    st.session_state.individual_sales_data = {}
if 'last_forecast_method' not in st.session_state:
    st.session_state.last_forecast_method = ""
if 'product_to_remove' not in st.session_state:
    st.session_state.product_to_remove = None
if 'clear_all_flag' not in st.session_state:
    st.session_state.clear_all_flag = False
if 'individual_forecast_results' not in st.session_state:
    st.session_state.individual_forecast_results = []


# =============================================================================
# ユーティリティ関数
# =============================================================================

def round_up_to_50(value: int) -> int:
    """50の倍数に切り上げ"""
    if value <= 0:
        return 0
    return ((value + 49) // 50) * 50


def get_available_forecast_methods() -> List[str]:
    """利用可能な予測方法のリストを取得"""
    methods = []
    for method_name, method_info in FORECAST_METHODS.items():
        if method_info.get('requires_vertex_ai', False) and not VERTEX_AI_AVAILABLE:
            continue
        methods.append(method_name)
    return methods


def get_mobile_chart_config() -> dict:
    """スマホ最適化されたPlotlyチャート設定を取得"""
    return {
        'displayModeBar': False,  # ツールバー非表示
        'staticPlot': False,      # 操作は可能
        'responsive': True,       # レスポンシブ
        'scrollZoom': False,      # スクロールズーム無効
    }


def get_mobile_chart_layout(title: str = '', height: int = 300) -> dict:
    """スマホ最適化されたPlotlyレイアウト設定を取得"""
    return {
        'title': dict(text=title, font=dict(size=14)),
        'height': height,
        'margin': dict(l=40, r=20, t=40, b=40),
        'legend': dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        'xaxis': dict(
            tickfont=dict(size=10),
            title=dict(font=dict(size=11))
        ),
        'yaxis': dict(
            tickfont=dict(size=10),
            title=dict(font=dict(size=11))
        ),
        'dragmode': False,
        'hovermode': 'x unified',
    }


# =============================================================================
# データ初期化
# =============================================================================

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
    
    if category_col is None:
        return
    
    product_col = None
    for col in df_items.columns:
        if '商品名' in col or col == '商品' or col == 'product':
            product_col = col
            break
    
    if product_col is None and len(df_items.columns) >= 3:
        product_col = df_items.columns[2]
    
    if product_col is None:
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


# =============================================================================
# ヘッダー
# =============================================================================

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
            st.session_state.forecast_results = {}
            st.session_state.individual_sales_data = {}
            st.rerun()
    
    # Vertex AIステータス表示
    if VERTEX_AI_AVAILABLE:
        st.markdown(f"""
        <div class="vertex-ai-status vertex-ai-available">
            ✅ <strong>Vertex AI AutoML Forecasting:</strong> 接続済み
            （プロジェクト: {VERTEX_AI_CONFIG['project_id']}, リージョン: {VERTEX_AI_CONFIG['location']}）
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="vertex-ai-status vertex-ai-unavailable">
            ⚠️ <strong>Vertex AI:</strong> 未設定（統計モデルで予測します）
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.data_loader:
        min_date, max_date = st.session_state.data_loader.get_date_range()
        if min_date and max_date:
            st.caption(f"📅 データ期間: {min_date.strftime('%Y年%m月%d日')} 〜 {max_date.strftime('%Y年%m月%d日')}")


# =============================================================================
# メインナビゲーション
# =============================================================================

def render_main_tabs():
    """メインタブを描画"""
    tab_labels = [
        "📊 既存授与品の分析・予測",
        "✨ 新規授与品の需要予測",
        "⚙️ Vertex AI設定",
    ]
    
    if ADVANCED_ANALYSIS_AVAILABLE:
        tab_labels.append("🔬 高度な分析")
    
    tab_labels.append("📈 予測精度ダッシュボード")
    
    tabs = st.tabs(tab_labels)
    
    tab_idx = 0
    
    with tabs[tab_idx]:
        render_existing_product_analysis()
    tab_idx += 1
    
    with tabs[tab_idx]:
        render_new_product_forecast()
    tab_idx += 1
    
    with tabs[tab_idx]:
        render_vertex_ai_settings()
    tab_idx += 1
    
    if ADVANCED_ANALYSIS_AVAILABLE:
        with tabs[tab_idx]:
            render_advanced_analysis()
        tab_idx += 1
    
    with tabs[tab_idx]:
        render_accuracy_dashboard()


# =============================================================================
# Vertex AI設定タブ
# =============================================================================

def render_vertex_ai_settings():
    """Vertex AI設定タブ"""
    st.markdown('<p class="section-header">⚙️ Vertex AI AutoML Forecasting 設定</p>', unsafe_allow_html=True)
    
    # 現在の設定状況
    st.write("### 📋 現在の設定状況")
    
    config_status = {
        'プロジェクトID': VERTEX_AI_CONFIG['project_id'] or '未設定',
        'リージョン': VERTEX_AI_CONFIG['location'] or '未設定',
        'エンドポイントID': VERTEX_AI_CONFIG['endpoint_id'] or '未設定',
        'サービスアカウントファイル': VERTEX_AI_CONFIG['service_account_file'],
        'ファイル存在': '✅ あり' if os.path.exists(VERTEX_AI_CONFIG['service_account_file']) else '❌ なし',
        'Vertex AI利用可能': '✅ はい' if VERTEX_AI_AVAILABLE else '❌ いいえ',
    }
    
    for key, value in config_status.items():
        st.write(f"- **{key}**: {value}")
    
    st.divider()
    
    # 設定方法の説明
    st.write("### 🔧 設定方法")
    
    st.markdown("""
    **方法1: 環境変数で設定**
    ```bash
    export VERTEX_AI_PROJECT_ID="your-project-id"
    export VERTEX_AI_LOCATION="asia-northeast1"
    export VERTEX_AI_ENDPOINT_ID="your-endpoint-id"
    export VERTEX_AI_SERVICE_ACCOUNT_FILE="path/to/service_account.json"
    ```
    
    **方法2: config.pyで設定**
    ```python
    # config.py
    VERTEX_AI_PROJECT_ID = "your-project-id"
    VERTEX_AI_LOCATION = "asia-northeast1"
    VERTEX_AI_ENDPOINT_ID = "your-endpoint-id"
    VERTEX_AI_SERVICE_ACCOUNT_FILE = "service_account.json"
    ```
    """)
    
    st.divider()
    
    # AutoML Forecastingモデルの作成手順
    st.write("### 📚 AutoML Forecastingモデルの作成手順")
    
    with st.expander("1️⃣ データの準備", expanded=False):
        st.markdown("""
        Vertex AI AutoML Forecastingに必要なデータ形式：
        
        | カラム | 説明 | 例 |
        |--------|------|-----|
        | timestamp | 時間列（ISO形式） | 2025-01-01T00:00:00Z |
        | target | 予測対象（販売数） | 15 |
        | time_series_identifier | 系列識別子（商品ID等） | product_001 |
        | weekday | 曜日（共変量） | 0-6 |
        | is_holiday | 休日フラグ（共変量） | 0 or 1 |
        | weather | 天気（共変量） | sunny, rainy, etc. |
        """)
    
    with st.expander("2️⃣ モデルのトレーニング", expanded=False):
        st.markdown("""
        1. [Google Cloud Console](https://console.cloud.google.com/vertex-ai) にアクセス
        2. 「データセット」→「作成」→「時系列予測」を選択
        3. CSVをアップロードし、カラムを設定
        4. 「トレーニング」→「AutoML」を選択
        5. トレーニング完了を待つ（数時間〜）
        """)
    
    with st.expander("3️⃣ エンドポイントのデプロイ", expanded=False):
        st.markdown("""
        1. トレーニング済みモデルを選択
        2. 「デプロイとテスト」→「エンドポイントにデプロイ」
        3. エンドポイント名を設定してデプロイ
        4. デプロイ完了後、エンドポイントIDをコピー
        """)
    
    with st.expander("4️⃣ サービスアカウントの設定", expanded=False):
        st.markdown("""
        1. 「IAMと管理」→「サービスアカウント」
        2. 「サービスアカウントを作成」
        3. 以下のロールを付与：
           - Vertex AI ユーザー
           - Vertex AI 予測ユーザー
        4. 「鍵を作成」→ JSON形式でダウンロード
        5. ダウンロードしたファイルをプロジェクトフォルダに配置
        """)
    
    # 接続テスト
    st.divider()
    st.write("### 🧪 接続テスト")
    
    if st.button("🔍 Vertex AI接続をテスト", type="primary"):
        if not VERTEX_AI_AVAILABLE:
            st.error("Vertex AIが設定されていません。上記の設定を完了してください。")
        else:
            with st.spinner("接続テスト中..."):
                try:
                    forecaster = get_vertex_ai_forecaster()
                    # 簡単なテストデータで接続確認
                    test_df = pd.DataFrame({
                        'date': pd.date_range(start='2025-01-01', periods=30, freq='D'),
                        '販売商品数': np.random.randint(1, 10, 30)
                    })
                    predictions, metadata = forecaster.predict(test_df, 7, "test_product")
                    st.success(f"✅ 接続成功！モデルID: {metadata.get('deployed_model_id', 'N/A')}")
                    st.write("テスト予測結果:")
                    st.dataframe(predictions.head())
                except Exception as e:
                    st.error(f"❌ 接続エラー: {e}")


# =============================================================================
# 既存授与品の分析
# =============================================================================

def render_existing_product_analysis():
    """既存授与品の分析・予測"""
    render_product_selection()
    start_date, end_date = render_period_selection()
    
    if len(st.session_state.selected_products) > 1:
        render_analysis_mode_selection()
    
    if st.session_state.analysis_mode == "個別":
        render_individual_analysis(start_date, end_date)
    else:
        sales_data = render_sales_analysis(start_date, end_date)
        render_forecast_section(sales_data)
        render_delivery_section()


def render_analysis_mode_selection():
    """合算/個別モードの選択"""
    st.markdown('<p class="section-header">📊 分析モード</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio(
            "複数授与品の分析方法",
            ["合算", "個別"],
            index=0 if st.session_state.analysis_mode == "合算" else 1,
            horizontal=True,
            help="合算：選択した授与品の合計を分析\n個別：授与品ごとに別々に分析"
        )
        st.session_state.analysis_mode = mode
    
    with col2:
        if mode == "合算":
            st.info(f"📊 {len(st.session_state.selected_products)}件の授与品を**合計**して分析します")
        else:
            st.info(f"📊 {len(st.session_state.selected_products)}件の授与品を**個別**に分析します")


def render_product_selection():
    """授与品選択セクション"""
    st.markdown('<p class="section-header">① 授与品を選ぶ</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔍 名前で検索", "📁 カテゴリーから選ぶ"])
    
    with tab1:
        render_search_tab()
    
    with tab2:
        render_category_tab()
    
    render_selected_products()


def render_search_tab():
    """名前検索タブ"""
    search_query = st.text_input(
        "授与品名を入力",
        placeholder="例: 金運、お守り、御朱印帳...",
        key="search_input"
    )
    
    if search_query and st.session_state.normalizer:
        results = st.session_state.normalizer.search(search_query, limit=20)
        
        if results:
            st.write(f"**{len(results)}件** 見つかりました")
            
            cols = st.columns(3)
            for i, result in enumerate(results):
                name = result['normalized_name']
                bracket = result.get('bracket_content', '')
                
                with cols[i % 3]:
                    is_selected = name in st.session_state.selected_products
                    label = f"{name}"
                    if bracket:
                        label += f" ({bracket})"
                    
                    if st.checkbox(label, value=is_selected, key=f"search_{name}"):
                        if name not in st.session_state.selected_products:
                            st.session_state.selected_products.append(name)
                    else:
                        if name in st.session_state.selected_products:
                            st.session_state.selected_products.remove(name)
        else:
            st.info("該当する授与品が見つかりませんでした")


def render_category_tab():
    """カテゴリー選択タブ"""
    if not st.session_state.categories:
        st.info("カテゴリー情報がありません")
        return
    
    st.write("**カテゴリーを選択して一括追加：**")
    
    cols = st.columns(4)
    
    sorted_categories = sorted(
        st.session_state.categories.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    for i, (category, products) in enumerate(sorted_categories[:12]):
        with cols[i % 4]:
            if st.button(f"📁 {category} ({len(products)}件)", key=f"cat_{category}"):
                for p in products:
                    if p not in st.session_state.selected_products:
                        st.session_state.selected_products.append(p)
                st.rerun()


def render_selected_products():
    """選択中の授与品を表示（個別削除機能付き）"""
    st.divider()
    
    if st.session_state.selected_products:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**✅ 選択中の授与品（{len(st.session_state.selected_products)}件）**")
        with col2:
            if st.button("🗑️ すべてクリア", key="clear_all_btn_main"):
                st.session_state.selected_products = []
                st.session_state.analysis_mode = "合算"
                st.session_state.sales_data = None
                st.session_state.forecast_data = None
                st.session_state.individual_sales_data = {}
                st.session_state.individual_forecast_results = []
                st.rerun()
        
        # 選択中の商品を表示
        st.markdown('<div style="background: #e3f2fd; border-radius: 10px; padding: 15px; margin: 10px 0;">', unsafe_allow_html=True)
        
        for product in st.session_state.selected_products:
            st.markdown(f"📦 **{product}**")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 削除用セレクトボックス
        st.write("**🗑️ 商品を個別に削除**")
        product_to_delete = st.selectbox(
            "削除する商品を選択",
            options=["（選択してください）"] + st.session_state.selected_products,
            key="product_delete_select",
            label_visibility="collapsed"
        )
        
        if product_to_delete != "（選択してください）":
            if st.button(f"「{product_to_delete}」を削除", key="delete_selected_product_btn", type="secondary"):
                st.session_state.selected_products.remove(product_to_delete)
                st.session_state.sales_data = None
                st.session_state.forecast_data = None
                st.session_state.individual_sales_data = {}
                st.session_state.individual_forecast_results = []
                st.rerun()
    else:
        st.warning("👆 上から授与品を選んでください")


def render_period_selection():
    """期間選択セクション"""
    st.markdown('<p class="section-header">② 期間を選ぶ</p>', unsafe_allow_html=True)
    
    today = date.today()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        preset = st.selectbox(
            "プリセット",
            ["カスタム", "過去1ヶ月", "過去3ヶ月", "過去6ヶ月", "過去1年", "過去2年", "全期間"],
            index=4
        )
    
    if preset == "過去1ヶ月":
        default_start = today - timedelta(days=30)
        default_end = today
    elif preset == "過去3ヶ月":
        default_start = today - timedelta(days=90)
        default_end = today
    elif preset == "過去6ヶ月":
        default_start = today - timedelta(days=180)
        default_end = today
    elif preset == "過去1年":
        default_start = today - timedelta(days=365)
        default_end = today
    elif preset == "過去2年":
        default_start = today - timedelta(days=730)
        default_end = today
    elif preset == "全期間":
        default_start = date(2022, 8, 1)
        default_end = today
    else:
        default_start = today - timedelta(days=365)
        default_end = today
    
    with col2:
        # 開始日
        st.write("**開始日**")
        col_sy, col_sm, col_sd = st.columns(3)
        
        years = list(range(2022, today.year + 2))
        months_jp = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
        
        with col_sy:
            start_year = st.selectbox(
                "年",
                years,
                index=years.index(default_start.year) if default_start.year in years else 0,
                key="start_year",
                label_visibility="collapsed"
            )
            st.caption("年")
        with col_sm:
            start_month = st.selectbox(
                "月",
                list(range(1, 13)),
                index=default_start.month - 1,
                format_func=lambda x: months_jp[x-1],
                key="start_month",
                label_visibility="collapsed"
            )
            st.caption("月")
        with col_sd:
            max_day_start = calendar.monthrange(start_year, start_month)[1]
            start_day = st.number_input(
                "日",
                min_value=1,
                max_value=max_day_start,
                value=min(default_start.day, max_day_start),
                key="start_day",
                label_visibility="collapsed"
            )
            st.caption("日")
        
        # 終了日
        st.write("**終了日**")
        col_ey, col_em, col_ed = st.columns(3)
        
        with col_ey:
            end_year = st.selectbox(
                "年",
                years,
                index=years.index(default_end.year) if default_end.year in years else 0,
                key="end_year",
                label_visibility="collapsed"
            )
            st.caption("年")
        with col_em:
            end_month = st.selectbox(
                "月",
                list(range(1, 13)),
                index=default_end.month - 1,
                format_func=lambda x: months_jp[x-1],
                key="end_month",
                label_visibility="collapsed"
            )
            st.caption("月")
        with col_ed:
            max_day_end = calendar.monthrange(end_year, end_month)[1]
            end_day = st.number_input(
                "日",
                min_value=1,
                max_value=max_day_end,
                value=min(default_end.day, max_day_end),
                key="end_day",
                label_visibility="collapsed"
            )
            st.caption("日")
    
    start_date = date(start_year, start_month, start_day)
    end_date = date(end_year, end_month, end_day)
    
    if start_date > end_date:
        st.error("⚠️ 開始日が終了日より後になっています")
        end_date = start_date
    
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
    
    # 平日・休日の平均を計算
    df_agg['weekday'] = pd.to_datetime(df_agg['date']).dt.dayofweek
    df_weekday = df_agg[df_agg['weekday'] < 5]  # 月〜金
    df_weekend = df_agg[df_agg['weekday'] >= 5]  # 土日
    
    avg_weekday = df_weekday['販売商品数'].mean() if not df_weekday.empty else 0
    avg_weekend = df_weekend['販売商品数'].mean() if not df_weekend.empty else 0
    
    # メトリクス表示（2行に分ける）
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🛒 販売数量", f"{total_qty:,}体")
    col2.metric("💰 売上合計", f"¥{total_sales:,.0f}")
    col3.metric("📈 平均日販", f"{avg_daily:.1f}体/日")
    col4.metric("📅 期間", f"{period_days}日間")
    
    # 平日・休日の平均を表示
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("📅 平日平均", f"{avg_weekday:.1f}体/日", help="月〜金曜日の平均")
    col6.metric("🎌 休日平均", f"{avg_weekend:.1f}体/日", help="土・日曜日の平均")
    
    # 休日/平日比率
    if avg_weekday > 0:
        ratio = avg_weekend / avg_weekday
        col7.metric("📊 休日/平日比", f"{ratio:.2f}倍")
    
    st.session_state.sales_data = df_agg
    
    return df_agg


def render_forecast_section(sales_data: pd.DataFrame):
    """需要予測セクション（Vertex AI対応）"""
    st.markdown('<p class="section-header">④ 需要を予測する</p>', unsafe_allow_html=True)
    
    if sales_data is None or sales_data.empty:
        st.info("売上データがあると、需要予測ができます")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_mode = st.radio(
            "予測期間の指定方法",
            ["日数で指定", "期間で指定"],
            horizontal=True,
            key="forecast_mode_existing",
            help="「期間で指定」は期間限定品の予測に便利です"
        )
    
    with col2:
        available_methods = get_available_forecast_methods()
        default_idx = 0  # Vertex AIがあれば0、なければ季節性考慮
        if "🚀 Vertex AI（推奨）" not in available_methods:
            default_idx = available_methods.index("季節性考慮（統計）") if "季節性考慮（統計）" in available_methods else 0
        
        method = st.selectbox(
            "予測方法",
            available_methods,
            index=default_idx,
            key="forecast_method_existing"
        )
    
    # 予測期間の設定
    if forecast_mode == "日数で指定":
        forecast_days = st.slider("予測日数", 30, 365, 180, key="forecast_days_existing")
        forecast_start_date = None
        forecast_end_date = None
    else:
        # 期間指定UI（分析期間と同じスタイル）
        today = date.today()
        default_start = today + timedelta(days=1)
        default_end = today + timedelta(days=180)
        
        st.write("**予測期間指定**")
        col_s1, col_s2, col_s3, col_e1, col_e2, col_e3 = st.columns([1, 1, 1, 1, 1, 1])
        
        with col_s1:
            start_year = st.selectbox(
                "予測開始年",
                list(range(2025, 2028)),
                index=list(range(2025, 2028)).index(default_start.year) if default_start.year in range(2025, 2028) else 0,
                key="forecast_start_year"
            )
        with col_s2:
            start_month = st.selectbox(
                "予測開始月",
                list(range(1, 13)),
                index=default_start.month - 1,
                format_func=lambda x: f"{x}月",
                key="forecast_start_month"
            )
        with col_s3:
            max_day_start = calendar.monthrange(start_year, start_month)[1]
            start_day = st.selectbox(
                "予測開始日",
                list(range(1, max_day_start + 1)),
                index=min(default_start.day - 1, max_day_start - 1),
                format_func=lambda x: f"{x}日",
                key="forecast_start_day"
            )
        
        with col_e1:
            end_year = st.selectbox(
                "予測終了年",
                list(range(2025, 2028)),
                index=list(range(2025, 2028)).index(default_end.year) if default_end.year in range(2025, 2028) else 0,
                key="forecast_end_year"
            )
        with col_e2:
            end_month = st.selectbox(
                "予測終了月",
                list(range(1, 13)),
                index=default_end.month - 1,
                format_func=lambda x: f"{x}月",
                key="forecast_end_month"
            )
        with col_e3:
            max_day_end = calendar.monthrange(end_year, end_month)[1]
            end_day = st.selectbox(
                "予測終了日",
                list(range(1, max_day_end + 1)),
                index=min(default_end.day - 1, max_day_end - 1),
                format_func=lambda x: f"{x}日",
                key="forecast_end_day"
            )
        
        forecast_start_date = date(start_year, start_month, start_day)
        forecast_end_date = date(end_year, end_month, end_day)
        
        if forecast_end_date <= forecast_start_date:
            st.error("⚠️ 終了日は開始日より後にしてください")
            return
        
        forecast_days = (forecast_end_date - forecast_start_date).days + 1
        st.info(f"📅 予測期間: {forecast_start_date.strftime('%Y年%m月%d日')} 〜 {forecast_end_date.strftime('%Y年%m月%d日')}（{forecast_days}日間）")
    
    # 予測方法の説明を表示
    method_info = FORECAST_METHODS[method]
    css_class = "vertex-ai" if "Vertex" in method else "seasonality" if "季節" in method else "moving-avg" if "移動" in method else "exponential"
    
    st.markdown(f"""
    <div class="method-card method-{css_class}">
        <strong>{method_info['icon']} {method}</strong><br>
        {method_info['description']}
    </div>
    """, unsafe_allow_html=True)
    
    # 共変量オプション（Vertex AI選択時）
    use_covariates = False
    if "Vertex AI" in method and VERTEX_AI_AVAILABLE:
        use_covariates = st.checkbox(
            "共変量を使用（天気・六曜・イベント）",
            value=True,
            help="予測精度が向上しますが、処理時間が長くなる場合があります"
        )
    
    if st.button("🔮 需要を予測", type="primary", use_container_width=True, key="forecast_btn_existing"):
        with st.spinner("予測中..."):
            try:
                if method == "🔄 すべての方法で比較":
                    # すべての方法で予測
                    product_id = "_".join(st.session_state.selected_products[:3])
                    all_results = forecast_all_methods_with_vertex_ai(sales_data, forecast_days, product_id)
                    display_comparison_results_v12(all_results, forecast_days, sales_data)
                else:
                    # 単一の予測方法
                    product_id = "_".join(st.session_state.selected_products[:3])
                    forecast, method_message = forecast_with_vertex_ai(sales_data, forecast_days, method, product_id)
                    
                    if forecast is not None and not forecast.empty:
                        display_single_forecast_result_v12(forecast, forecast_days, method, method_message, sales_data)
                    else:
                        st.error("予測結果が空です。データを確認してください。")
            except Exception as e:
                st.error(f"予測エラー: {e}")
                logger.error(f"予測エラー: {e}")


def display_single_forecast_result_v12(forecast: pd.DataFrame, forecast_days: int, method: str, method_message: str, sales_data: pd.DataFrame = None):
    """単一の予測結果を表示（v12 スマホ最適化 + ロジック説明）"""
    raw_total = int(forecast['predicted'].sum())
    rounded_total = round_up_to_50(raw_total)
    avg_predicted = forecast['predicted'].mean()
    
    # Vertex AI使用時は特別表示
    if "Vertex AI" in method_message:
        st.success(f"✅ 予測完了！（🚀 {method_message}）")
    else:
        st.success(f"✅ 予測完了！（{method_message}）")
    
    st.session_state.last_forecast_method = method_message
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 予測販売総数", f"{rounded_total:,}体")
    col2.metric("📈 平均日販（予測）", f"{avg_predicted:.1f}体/日")
    col3.metric("📅 予測期間", f"{forecast_days}日間")
    
    # 予測ロジックの説明を追加
    with st.expander("📊 予測ロジックの詳細", expanded=False):
        display_forecast_logic_explanation(method, sales_data, forecast, forecast_days, avg_predicted)
    
    # グラフ表示（スマホ最適化）
    method_info = FORECAST_METHODS.get(method, {"color": "#4285F4"})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['predicted'],
        mode='lines',
        name='予測',
        line=dict(color=method_info.get('color', '#4285F4'), width=2)
    ))
    
    # 信頼区間があれば表示
    if 'confidence_lower' in forecast.columns and forecast['confidence_lower'].notna().any():
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['confidence_upper'],
            mode='lines',
            name='上限',
            line=dict(color='rgba(66, 133, 244, 0.3)', dash='dash'),
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['confidence_lower'],
            mode='lines',
            name='下限',
            line=dict(color='rgba(66, 133, 244, 0.3)', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(66, 133, 244, 0.1)',
            showlegend=True
        ))
    
    # スマホ最適化レイアウト
    layout = get_mobile_chart_layout(f'{method}による日別予測', height=280)
    layout['xaxis_title'] = '日付'
    layout['yaxis_title'] = '予測販売数（体）'
    fig.update_layout(**layout)
    
    st.plotly_chart(fig, use_container_width=True, config=get_mobile_chart_config())
    
    st.session_state.forecast_data = forecast
    st.session_state.forecast_total = rounded_total


def display_forecast_logic_explanation(method: str, sales_data: pd.DataFrame, forecast: pd.DataFrame, forecast_days: int, avg_predicted: float):
    """予測ロジックの詳細説明を表示"""
    
    if sales_data is None or sales_data.empty:
        st.write("入力データがありません")
        return
    
    # 入力データの統計
    total_days = len(sales_data)
    total_qty = int(sales_data['販売商品数'].sum())
    avg_daily = sales_data['販売商品数'].mean()
    max_daily = sales_data['販売商品数'].max()
    min_daily = sales_data['販売商品数'].min()
    
    st.write("#### 📥 入力データ（過去の実績）")
    st.write(f"""
    - **分析期間**: {total_days}日間
    - **総販売数**: {total_qty:,}体
    - **平均日販**: {avg_daily:.1f}体/日
    - **最大日販**: {max_daily:.0f}体/日
    - **最小日販**: {min_daily:.0f}体/日
    """)
    
    st.write("#### 🔮 予測ロジック")
    
    if "Vertex AI" in method:
        st.write(f"""
        **Vertex AI AutoML Forecasting**
        1. 過去{total_days}日間のデータを機械学習モデルに入力
        2. 時系列パターン（トレンド・周期性）を自動検出
        3. 天気・六曜・イベント情報も考慮（共変量）
        4. {forecast_days}日間の日別予測を生成
        
        **計算結果**:
        - 予測平均日販: {avg_predicted:.1f}体/日
        - 実績平均との差: {avg_predicted - avg_daily:+.1f}体/日 ({((avg_predicted/avg_daily)-1)*100:+.1f}%)
        """)
    
    elif "季節性" in method:
        # 曜日別平均を計算
        if 'date' in sales_data.columns:
            sales_data_copy = sales_data.copy()
            sales_data_copy['weekday'] = pd.to_datetime(sales_data_copy['date']).dt.dayofweek
            weekday_avg = sales_data_copy.groupby('weekday')['販売商品数'].mean()
            weekday_names = ['月', '火', '水', '木', '金', '土', '日']
            weekday_str = ", ".join([f"{weekday_names[i]}:{weekday_avg.get(i, 0):.1f}" for i in range(7)])
        else:
            weekday_str = "データなし"
        
        st.write(f"""
        **季節性考慮予測**
        1. 曜日別の平均販売数を計算
        2. 月別の季節係数を算出
        3. 曜日パターン × 季節係数で日別予測
        
        **曜日別平均**: {weekday_str}
        
        **計算結果**:
        - 予測平均日販: {avg_predicted:.1f}体/日
        - 実績平均との差: {avg_predicted - avg_daily:+.1f}体/日 ({((avg_predicted/avg_daily)-1)*100:+.1f}%)
        """)
    
    elif "移動平均" in method:
        # 直近30日の平均
        recent_30 = sales_data.tail(30)['販売商品数'].mean() if len(sales_data) >= 30 else avg_daily
        
        st.write(f"""
        **移動平均法**
        1. 直近30日間の販売データを使用
        2. 30日間の平均値を基準として予測
        
        **直近30日平均**: {recent_30:.1f}体/日
        
        **計算式**: 予測日販 = 直近30日平均 = {recent_30:.1f}体/日
        **予測総数**: {recent_30:.1f} × {forecast_days}日 = {recent_30 * forecast_days:.0f}体
        """)
    
    elif "指数平滑" in method:
        alpha = 0.3  # 平滑化係数
        recent_7 = sales_data.tail(7)['販売商品数'].mean() if len(sales_data) >= 7 else avg_daily
        
        st.write(f"""
        **指数平滑法**
        1. 直近のデータを重視（平滑化係数 α={alpha}）
        2. 新しいデータほど高い重みで計算
        
        **直近7日平均**: {recent_7:.1f}体/日
        **全期間平均**: {avg_daily:.1f}体/日
        
        **計算式**: 予測 = α×直近 + (1-α)×全体 = {alpha}×{recent_7:.1f} + {1-alpha}×{avg_daily:.1f} = {alpha*recent_7 + (1-alpha)*avg_daily:.1f}体/日
        """)
    
    else:
        st.write(f"""
        **予測方法**: {method}
        - 入力データ: {total_days}日間の実績
        - 予測期間: {forecast_days}日間
        - 予測平均日販: {avg_predicted:.1f}体/日
        """)


def display_comparison_results_v12(all_results: Dict[str, Tuple[pd.DataFrame, str]], forecast_days: int, sales_data: pd.DataFrame = None):
    """すべての予測方法の比較結果を表示（v12 スマホ最適化 + 予測総数一覧）"""
    st.success("✅ すべての予測方法で比較完了！")
    
    # 各予測方法の予測総数を計算
    method_totals = {}
    for method_name, (forecast, message) in all_results.items():
        raw_total = int(forecast['predicted'].sum())
        rounded_total = round_up_to_50(raw_total)
        avg_predicted = forecast['predicted'].mean()
        method_totals[method_name] = {
            'raw': raw_total,
            'rounded': rounded_total,
            'avg': avg_predicted
        }
    
    # ========== 予測総数サマリー表 ==========
    st.write("### 📊 予測方法別 予測総数サマリー")
    
    # 表形式で表示
    summary_rows = []
    for method_name, totals in method_totals.items():
        icon = "🚀" if "Vertex" in method_name else "📈" if "季節" in method_name else "📊" if "移動" in method_name else "📉"
        summary_rows.append({
            '予測方法': f"{icon} {method_name}",
            '予測総数（生値）': f"{totals['raw']:,}体",
            '発注推奨数（50倍数）': f"{totals['rounded']:,}体",
            '平均日販': f"{totals['avg']:.1f}体/日"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # 4つの方法の統計
    all_rounded = [t['rounded'] for t in method_totals.values()]
    all_raw = [t['raw'] for t in method_totals.values()]
    
    st.write("### 📈 予測値の統計")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 最小値", f"{min(all_rounded):,}体")
    col2.metric("📊 最大値", f"{max(all_rounded):,}体")
    col3.metric("📊 平均値", f"{round_up_to_50(int(sum(all_raw) / len(all_raw))):,}体")
    col4.metric("📊 中央値", f"{round_up_to_50(int(sorted(all_raw)[len(all_raw)//2])):,}体")
    
    # 差分の表示
    if len(all_rounded) >= 2:
        diff = max(all_rounded) - min(all_rounded)
        diff_pct = (max(all_raw) - min(all_raw)) / min(all_raw) * 100 if min(all_raw) > 0 else 0
        st.info(f"📏 **予測値の幅**: 最小〜最大で **{diff:,}体** の差（{diff_pct:.1f}%）")
    
    method_colors = {
        'Vertex AI': '#4285F4',
        '季節性考慮': '#4CAF50',
        '移動平均法': '#1E88E5',
        '指数平滑法': '#FF9800'
    }
    
    # 比較グラフ（スマホ最適化）
    st.write("### 📈 日別予測比較グラフ")
    
    fig = go.Figure()
    
    for method_name, (forecast, message) in all_results.items():
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['predicted'],
            mode='lines',
            name=method_name,
            line=dict(color=method_colors.get(method_name, '#666666'), width=2)
        ))
    
    layout = get_mobile_chart_layout('予測方法別の日別予測比較', height=300)
    layout['xaxis_title'] = '日付'
    layout['yaxis_title'] = '予測販売数（体）'
    fig.update_layout(**layout)
    
    st.plotly_chart(fig, use_container_width=True, config=get_mobile_chart_config())
    
    # 推奨
    if 'Vertex AI' in all_results:
        st.info("💡 **おすすめ**: Vertex AI AutoML Forecastingは機械学習モデルで学習済みのため、最も精度が高い傾向があります。")
    else:
        st.info("💡 **おすすめ**: 季節性考慮は月別・曜日別の傾向を考慮するため、統計モデルの中では最も精度が高い傾向があります。")
    
    # セッション状態に保存（Vertex AIがあればそれ、なければ季節性考慮）
    if 'Vertex AI' in all_results:
        st.session_state.forecast_data = all_results['Vertex AI'][0]
        st.session_state.forecast_total = method_totals['Vertex AI']['rounded']
    elif '季節性考慮' in all_results:
        st.session_state.forecast_data = all_results['季節性考慮'][0]
        st.session_state.forecast_total = method_totals['季節性考慮']['rounded']
    
    st.session_state.forecast_results = {k: v[0] for k, v in all_results.items()}


def render_individual_analysis(start_date: date, end_date: date):
    """個別分析モード"""
    st.markdown('<p class="section-header">③ 個別売上分析</p>', unsafe_allow_html=True)
    
    if not st.session_state.selected_products:
        st.info("授与品を選択すると、ここに売上が表示されます")
        return
    
    df_items = st.session_state.data_loader.load_item_sales()
    
    if df_items.empty:
        st.warning("データがありません")
        return
    
    mask = (df_items['date'] >= pd.Timestamp(start_date)) & (df_items['date'] <= pd.Timestamp(end_date))
    df_filtered = df_items[mask]
    
    individual_data = {}
    
    for product in st.session_state.selected_products:
        original_names = st.session_state.normalizer.get_all_original_names([product])
        df_agg = aggregate_by_products(df_filtered, original_names, aggregate=True)
        
        if not df_agg.empty:
            df_agg = df_agg.sort_values('date').reset_index(drop=True)
            individual_data[product] = df_agg
    
    st.session_state.individual_sales_data = individual_data
    
    for product, df_agg in individual_data.items():
        with st.expander(f"📦 **{product}**", expanded=True):
            total_qty = int(df_agg['販売商品数'].sum())
            total_sales = df_agg['販売総売上'].sum()
            period_days = (end_date - start_date).days + 1
            avg_daily = total_qty / period_days if period_days > 0 else 0
            
            # 平日・休日の平均を計算
            df_agg['weekday'] = pd.to_datetime(df_agg['date']).dt.dayofweek
            df_weekday = df_agg[df_agg['weekday'] < 5]
            df_weekend = df_agg[df_agg['weekday'] >= 5]
            avg_weekday = df_weekday['販売商品数'].mean() if not df_weekday.empty else 0
            avg_weekend = df_weekend['販売商品数'].mean() if not df_weekend.empty else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🛒 販売数量", f"{total_qty:,}体")
            col2.metric("💰 売上合計", f"¥{total_sales:,.0f}")
            col3.metric("📅 平日平均", f"{avg_weekday:.1f}体/日")
            col4.metric("🎌 休日平均", f"{avg_weekend:.1f}体/日")
    
    render_individual_forecast_section()
    
    # 個別モードでも納品計画を表示
    render_delivery_section()


def render_individual_forecast_section():
    """個別予測セクション（期間指定対応）"""
    st.markdown('<p class="section-header">④ 個別需要予測</p>', unsafe_allow_html=True)
    
    if not st.session_state.individual_sales_data:
        st.info("売上データがあると、需要予測ができます")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_mode = st.radio(
            "予測期間の指定方法",
            ["日数で指定", "期間で指定"],
            horizontal=True,
            key="individual_forecast_mode",
            help="「期間で指定」は期間限定品の予測に便利です"
        )
    
    with col2:
        available_methods = get_available_forecast_methods()
        # 個別モードでも「すべての方法で比較」を使用可能に
        
        method = st.selectbox(
            "予測方法",
            available_methods,
            index=0,
            key="individual_forecast_method"
        )
    
    # 予測期間の設定
    if forecast_mode == "日数で指定":
        forecast_days = st.slider("予測日数", 30, 365, 180, key="individual_forecast_days")
        forecast_start_date = None
        forecast_end_date = None
    else:
        # 期間指定UI（分析期間と同じスタイル）
        today = date.today()
        default_start = today + timedelta(days=1)
        default_end = today + timedelta(days=180)
        
        st.write("**予測期間指定**")
        col_s1, col_s2, col_s3, col_e1, col_e2, col_e3 = st.columns([1, 1, 1, 1, 1, 1])
        
        with col_s1:
            start_year = st.selectbox(
                "予測開始年",
                list(range(2025, 2028)),
                index=list(range(2025, 2028)).index(default_start.year) if default_start.year in range(2025, 2028) else 0,
                key="ind_forecast_start_year"
            )
        with col_s2:
            start_month = st.selectbox(
                "予測開始月",
                list(range(1, 13)),
                index=default_start.month - 1,
                format_func=lambda x: f"{x}月",
                key="ind_forecast_start_month"
            )
        with col_s3:
            max_day_start = calendar.monthrange(start_year, start_month)[1]
            start_day = st.selectbox(
                "予測開始日",
                list(range(1, max_day_start + 1)),
                index=min(default_start.day - 1, max_day_start - 1),
                format_func=lambda x: f"{x}日",
                key="ind_forecast_start_day"
            )
        
        with col_e1:
            end_year = st.selectbox(
                "予測終了年",
                list(range(2025, 2028)),
                index=list(range(2025, 2028)).index(default_end.year) if default_end.year in range(2025, 2028) else 0,
                key="ind_forecast_end_year"
            )
        with col_e2:
            end_month = st.selectbox(
                "予測終了月",
                list(range(1, 13)),
                index=default_end.month - 1,
                format_func=lambda x: f"{x}月",
                key="ind_forecast_end_month"
            )
        with col_e3:
            max_day_end = calendar.monthrange(end_year, end_month)[1]
            end_day = st.selectbox(
                "予測終了日",
                list(range(1, max_day_end + 1)),
                index=min(default_end.day - 1, max_day_end - 1),
                format_func=lambda x: f"{x}日",
                key="ind_forecast_end_day"
            )
        
        forecast_start_date = date(start_year, start_month, start_day)
        forecast_end_date = date(end_year, end_month, end_day)
        
        if forecast_end_date <= forecast_start_date:
            st.error("⚠️ 終了日は開始日より後にしてください")
            return
        
        forecast_days = (forecast_end_date - forecast_start_date).days + 1
        st.info(f"📅 予測期間: {forecast_start_date.strftime('%Y年%m月%d日')} 〜 {forecast_end_date.strftime('%Y年%m月%d日')}（{forecast_days}日間）")
    
    method_info = FORECAST_METHODS[method]
    st.markdown(f"""
    <div class="analysis-card">
        <strong>{method_info['icon']} {method}</strong><br>
        {method_info['description']}
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔮 個別に需要予測を実行", type="primary", use_container_width=True, key="individual_forecast_btn"):
        with st.spinner("予測中..."):
            results = []
            
            for product, sales_data in st.session_state.individual_sales_data.items():
                try:
                    forecast, method_message = forecast_with_vertex_ai(sales_data, forecast_days, method, product)
                    
                    if forecast is not None and not forecast.empty:
                        raw_total = int(forecast['predicted'].sum())
                        rounded_total = round_up_to_50(raw_total)
                        avg_predicted = forecast['predicted'].mean()
                        
                        results.append({
                            'product': product,
                            'forecast': forecast,
                            'raw_total': raw_total,
                            'rounded_total': rounded_total,
                            'avg_predicted': avg_predicted,
                            'method_message': method_message
                        })
                except Exception as e:
                    st.warning(f"{product}の予測に失敗: {e}")
            
            if results:
                # 納品計画で使えるようにsession_stateに保存
                if len(results) == 1:
                    st.session_state.forecast_data = results[0]['forecast']
                else:
                    # 複数商品の場合は日付ごとに合算
                    combined_forecast = results[0]['forecast'].copy()
                    combined_forecast = combined_forecast.rename(columns={'predicted': 'predicted_sum'})
                    
                    for r in results[1:]:
                        merged = combined_forecast.merge(
                            r['forecast'][['date', 'predicted']], 
                            on='date', 
                            how='outer'
                        )
                        merged['predicted_sum'] = merged['predicted_sum'].fillna(0) + merged['predicted'].fillna(0)
                        merged = merged.drop(columns=['predicted'])
                        combined_forecast = merged
                    
                    combined_forecast = combined_forecast.rename(columns={'predicted_sum': 'predicted'})
                    st.session_state.forecast_data = combined_forecast
                
                total_all = sum(r['rounded_total'] for r in results)
                st.session_state.forecast_total = total_all
                st.session_state.last_forecast_method = results[0]['method_message'] if results else ""
                st.session_state.individual_forecast_results = results  # 結果を保存
                st.rerun()  # 納品セクションを更新するため再描画
    
    # 予測結果の表示（session_stateから）
    if 'individual_forecast_results' in st.session_state and st.session_state.individual_forecast_results:
        results = st.session_state.individual_forecast_results
        st.success(f"✅ {len(results)}件の授与品の予測が完了しました！")
        
        summary_df = pd.DataFrame([
            {
                '授与品': r['product'],
                '予測総数': f"{r['rounded_total']:,}体",
                '平均日販': f"{r['avg_predicted']:.1f}体/日",
                '発注推奨数（50倍数）': r['rounded_total']
            }
            for r in results
        ])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        total_all = sum(r['rounded_total'] for r in results)
        st.metric("📦 全体の予測総数", f"{total_all:,}体")


def render_delivery_section():
    """納品計画セクション（個別モード対応）"""
    st.markdown('<p class="section-header">⑤ 納品計画を立てる</p>', unsafe_allow_html=True)
    
    # 個別予測結果があるかチェック
    individual_results = st.session_state.get('individual_forecast_results', [])
    forecast = st.session_state.get('forecast_data')
    
    if (not individual_results) and (forecast is None or (isinstance(forecast, pd.DataFrame) and forecast.empty)):
        st.info("需要予測を実行すると、納品計画を立てられます")
        return
    
    # 複数商品の個別予測結果がある場合
    if individual_results and len(individual_results) >= 1:
        if len(individual_results) > 1:
            st.success(f"📦 **{len(individual_results)}件の商品**の予測結果があります")
            
            delivery_view = st.radio(
                "納品計画の表示方法",
                ["📊 全商品を合算して計画", "📦 商品ごとに個別計画"],
                horizontal=True,
                key="delivery_view_mode_main"
            )
            
            if delivery_view == "📦 商品ごとに個別計画":
                st.divider()
                for idx, r in enumerate(individual_results):
                    product = r['product']
                    forecast_df = r['forecast']
                    rounded_total = r['rounded_total']
                    avg_predicted = r['avg_predicted']
                    
                    with st.expander(f"📦 **{product}**（予測: {rounded_total:,}体、日販: {avg_predicted:.1f}体）", expanded=(idx==0)):
                        render_delivery_inputs_and_schedule(
                            total_demand=rounded_total,
                            forecast_data=forecast_df,
                            product_name=product,
                            avg_daily=avg_predicted
                        )
                return
        else:
            # 1商品のみの場合
            r = individual_results[0]
            st.success(f"📦 **{r['product']}** の予測結果")
    
    # 合算モード
    total_demand = st.session_state.get('forecast_total', 0)
    method_used = st.session_state.get('last_forecast_method', '')
    forecast_data = forecast
    
    # 平均日販を計算
    forecast_days = len(forecast_data) if forecast_data is not None and not forecast_data.empty else 180
    avg_daily = total_demand / forecast_days if forecast_days > 0 else 0
    
    if method_used:
        st.info(f"📦 予測された需要数: **{total_demand:,}体**（{forecast_days}日間、日販{avg_daily:.1f}体） - {method_used}")
    else:
        st.info(f"📦 予測された需要数: **{total_demand:,}体**（{forecast_days}日間、日販{avg_daily:.1f}体）")
    
    render_delivery_inputs_and_schedule(total_demand, forecast_data, "合算", avg_daily)


def render_individual_delivery_plans(results: list):
    """個別商品ごとの納品計画を表示"""
    for idx, r in enumerate(results):
        product = r['product']
        forecast = r['forecast']
        rounded_total = r['rounded_total']
        avg_predicted = r['avg_predicted']
        
        with st.expander(f"📦 **{product}** の納品計画（予測: {rounded_total:,}体）", expanded=(idx==0)):
            render_delivery_inputs_and_schedule(
                total_demand=rounded_total,
                forecast_data=forecast,
                product_name=product,
                avg_daily=avg_predicted
            )


def render_delivery_inputs_and_schedule(total_demand: int, forecast_data: pd.DataFrame, product_name: str, avg_daily: float = 0):
    """納品計画の入力と計算を表示"""
    
    key_suffix = f"{product_name.replace(' ', '_')[:8]}_{hash(product_name) % 999}"
    
    # 予測期間（日数）を取得
    forecast_days = len(forecast_data) if forecast_data is not None and not forecast_data.empty else 180
    if avg_daily == 0:
        avg_daily = total_demand / forecast_days if forecast_days > 0 else 0
    
    # 入力セクション
    st.write("**📝 在庫・発注情報を入力**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_stock = st.number_input(
            "🏠 現在の在庫数", 
            min_value=0, 
            value=500, 
            step=50, 
            key=f"stk_{key_suffix}"
        )
    
    with col2:
        min_stock = st.number_input(
            "⚠️ 安全在庫数", 
            min_value=0, 
            value=100, 
            step=50, 
            key=f"minstk_{key_suffix}"
        )
    
    with col3:
        lead_time = st.number_input(
            "🚚 リードタイム(日)", 
            min_value=1, 
            value=14, 
            step=1, 
            key=f"lt_{key_suffix}",
            help="発注から納品までの日数"
        )
    
    # 発注数の計算
    needed = total_demand + min_stock - current_stock
    recommended_order = round_up_to_50(max(0, needed))
    
    # 推奨発注数と計算ロジックを常に表示
    st.divider()
    st.write("**🧮 発注推奨数の計算**")
    
    # 計算過程を表形式で表示
    col_calc1, col_calc2 = st.columns([2, 1])
    
    with col_calc1:
        st.markdown(f"""
        | 計算項目 | 数値 | 説明 |
        |:---------|-----:|:-----|
        | ① 予測需要 | **{total_demand:,}体** | {forecast_days}日間 × {avg_daily:.1f}体/日 |
        | ② 安全在庫 | **+{min_stock:,}体** | 欠品防止の余裕分 |
        | ③ 現在在庫 | **-{current_stock:,}体** | 既にある在庫 |
        | **必要数量** | **{needed:,}体** | ① + ② - ③ |
        | **発注推奨数** | **{recommended_order:,}体** | 50の倍数に切り上げ |
        """)
    
    with col_calc2:
        if needed <= 0:
            st.success(f"✅ 発注不要\n\n在庫で{forecast_days}日間カバー可能")
        else:
            days_until_stockout = int(current_stock / avg_daily) if avg_daily > 0 else 999
            st.warning(f"⚠️ 要発注\n\n約{days_until_stockout}日で在庫切れ")
    
    # 発注数入力方法
    order_mode = st.radio(
        "発注数の決め方",
        ["🔮 予測から自動計算", "✏️ 手入力で指定"],
        horizontal=True,
        key=f"ordmode_{key_suffix}"
    )
    
    if order_mode == "🔮 予測から自動計算":
        order_quantity = recommended_order
        st.metric("🛒 発注数（自動計算）", f"{recommended_order:,}体")
    else:
        order_quantity = st.number_input(
            "✏️ 発注数を入力",
            min_value=0,
            value=recommended_order,
            step=50,
            key=f"manord_{key_suffix}"
        )
    
    # 納品スケジュール提案
    st.divider()
    st.write("**📅 納品スケジュール提案**")
    
    delivery_mode = st.radio(
        "納品方法",
        ["一括納品", "分割納品（月別）", "分割納品（カスタム）"],
        horizontal=True,
        key=f"delivery_mode_{key_suffix}"
    )
    
    if st.button("📊 納品スケジュールを作成", type="primary", use_container_width=True, key=f"create_schedule_btn_{key_suffix}"):
        if order_quantity <= 0:
            st.warning("発注数が0です。発注の必要がありません。")
        else:
            schedule = create_delivery_schedule(
                order_quantity=order_quantity,
                current_stock=current_stock,
                min_stock=min_stock,
                lead_time=lead_time,
                forecast_data=forecast_data,
                delivery_mode=delivery_mode
            )
            
            display_delivery_schedule(schedule, current_stock, min_stock, forecast_data)


def create_delivery_schedule(
    order_quantity: int,
    current_stock: int,
    min_stock: int,
    lead_time: int,
    forecast_data: pd.DataFrame,
    delivery_mode: str
) -> List[Dict]:
    """納品スケジュールを作成"""
    
    today = date.today()
    
    if delivery_mode == "一括納品":
        delivery_date = today + timedelta(days=lead_time)
        return [{
            'date': delivery_date,
            'quantity': order_quantity,
            'type': '一括納品'
        }]
    
    elif delivery_mode == "分割納品（月別）":
        if forecast_data is None or forecast_data.empty:
            months = 3
        else:
            forecast_days = len(forecast_data)
            months = max(1, forecast_days // 30)
            months = min(months, 6)
        
        schedule = []
        qty_per_delivery = round_up_to_50(order_quantity // months)
        remaining = order_quantity
        
        for i in range(months):
            delivery_date = today + timedelta(days=lead_time + (i * 30))
            qty = min(qty_per_delivery, remaining)
            if qty > 0:
                schedule.append({
                    'date': delivery_date,
                    'quantity': qty,
                    'type': f'{i+1}回目'
                })
                remaining -= qty
        
        if remaining > 0 and schedule:
            schedule[-1]['quantity'] += remaining
        
        return schedule
    
    else:  # カスタム分割
        schedule = []
        stock = current_stock
        
        if forecast_data is not None and not forecast_data.empty:
            daily_demands = forecast_data['predicted'].tolist()
        else:
            daily_demands = [5] * 180
        
        delivery_qty = round_up_to_50(order_quantity // 3)
        remaining = order_quantity
        last_delivery_date = today
        
        for i, daily_demand in enumerate(daily_demands):
            target_date = today + timedelta(days=i)
            stock -= daily_demand
            
            if stock <= min_stock and remaining > 0:
                order_date = target_date - timedelta(days=lead_time)
                if order_date < last_delivery_date:
                    order_date = last_delivery_date + timedelta(days=1)
                
                delivery_date = order_date + timedelta(days=lead_time)
                qty = min(delivery_qty, remaining)
                
                schedule.append({
                    'date': delivery_date,
                    'quantity': qty,
                    'type': f'{len(schedule)+1}回目'
                })
                
                stock += qty
                remaining -= qty
                last_delivery_date = delivery_date
        
        if not schedule and remaining > 0:
            schedule.append({
                'date': today + timedelta(days=lead_time),
                'quantity': remaining,
                'type': '一括納品'
            })
        
        return schedule


def display_delivery_schedule(schedule: List[Dict], current_stock: int, min_stock: int, forecast_data: pd.DataFrame):
    """納品スケジュールを表示"""
    
    st.success(f"✅ 納品スケジュールを作成しました（{len(schedule)}回納品）")
    
    st.write("**📋 納品スケジュール**")
    
    schedule_df = pd.DataFrame([
        {
            '納品日': s['date'].strftime('%Y/%m/%d'),
            '曜日': ['月','火','水','木','金','土','日'][s['date'].weekday()],
            '数量': f"{s['quantity']:,}体",
            '備考': s['type']
        }
        for s in schedule
    ])
    
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)
    
    total_delivery = sum(s['quantity'] for s in schedule)
    st.metric("📦 納品合計", f"{total_delivery:,}体")
    
    with st.expander("📈 在庫推移シミュレーション", expanded=True):
        sim_data = simulate_inventory(
            schedule=schedule,
            current_stock=current_stock,
            min_stock=min_stock,
            forecast_data=forecast_data
        )
        
        if sim_data:
            display_inventory_chart(sim_data, min_stock)


def simulate_inventory(schedule: List[Dict], current_stock: int, min_stock: int, forecast_data: pd.DataFrame) -> List[Dict]:
    """在庫シミュレーションを実行"""
    
    today = date.today()
    
    if forecast_data is None or forecast_data.empty:
        return []
    
    sim_data = []
    stock = current_stock
    
    delivery_dict = {}
    for s in schedule:
        d = s['date']
        if d not in delivery_dict:
            delivery_dict[d] = 0
        delivery_dict[d] += s['quantity']
    
    sim_days = min(len(forecast_data), 90)
    
    for i in range(sim_days):
        target_date = today + timedelta(days=i)
        
        if target_date in delivery_dict:
            stock += delivery_dict[target_date]
        
        if i < len(forecast_data):
            daily_demand = forecast_data.iloc[i]['predicted']
        else:
            daily_demand = forecast_data['predicted'].mean()
        
        stock -= daily_demand
        stock = max(0, stock)
        
        sim_data.append({
            'date': target_date,
            'stock': stock,
            'demand': daily_demand,
            'delivery': delivery_dict.get(target_date, 0)
        })
    
    return sim_data


def display_inventory_chart(sim_data: List[Dict], min_stock: int):
    """在庫推移グラフを表示（スマホ最適化）"""
    
    df = pd.DataFrame(sim_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['stock'],
        mode='lines',
        name='在庫数',
        line=dict(color='#1E88E5', width=2),
        fill='tozeroy',
        fillcolor='rgba(30, 136, 229, 0.1)'
    ))
    
    fig.add_hline(
        y=min_stock, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"安全在庫 {min_stock}",
        annotation_position="right"
    )
    
    deliveries = df[df['delivery'] > 0]
    if not deliveries.empty:
        fig.add_trace(go.Scatter(
            x=deliveries['date'],
            y=deliveries['stock'],
            mode='markers',
            name='納品',
            marker=dict(color='green', size=12, symbol='triangle-up')
        ))
    
    fig.update_layout(
        title='在庫推移シミュレーション',
        xaxis_title='日付',
        yaxis_title='在庫数（体）',
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        dragmode=False,
    )
    
    config = {
        'displayModeBar': False,
        'staticPlot': False,
        'responsive': True
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    stock_below_min = df[df['stock'] < min_stock]
    if not stock_below_min.empty:
        first_danger = stock_below_min.iloc[0]['date']
        st.warning(f"⚠️ {first_danger.strftime('%Y/%m/%d')}頃に在庫が安全在庫を下回る可能性があります")


# =============================================================================
# 新規授与品の需要予測
# =============================================================================

def render_new_product_forecast():
    """新規授与品の需要予測"""
    
    st.markdown("""
    <div class="new-product-card">
        <h2>✨ 新規授与品の需要予測</h2>
        <p>まだ販売実績のない新しい授与品の需要を、類似商品のデータから予測します。</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">① 新規授与品の情報を入力</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_product_name = st.text_input(
            "授与品名",
            placeholder="例: 縁結び水晶守",
            help="新しく作る授与品の名前"
        )
        
        new_product_category = st.selectbox(
            "カテゴリー",
            list(CATEGORY_CHARACTERISTICS.keys()),
            help="最も近いカテゴリーを選んでください"
        )
        
        new_product_price = st.number_input(
            "価格（円）",
            min_value=100,
            max_value=50000,
            value=1000,
            step=100,
            help="販売予定価格"
        )
    
    with col2:
        new_product_description = st.text_area(
            "特徴・コンセプト",
            placeholder="例: 水晶を使用した縁結びのお守り。若い女性向け。",
            help="授与品の特徴を記述"
        )
        
        target_audience = st.multiselect(
            "ターゲット層",
            ["若い女性", "若い男性", "中高年女性", "中高年男性", "家族連れ", "観光客", "地元の方"],
            default=["若い女性", "観光客"]
        )
    
    st.markdown('<p class="section-header">② 類似商品を分析</p>', unsafe_allow_html=True)
    
    if new_product_name and new_product_name.strip():
        similar_products = find_similar_products(
            new_product_name, 
            new_product_category, 
            new_product_price,
            new_product_description
        )
        
        if similar_products:
            st.write(f"**類似商品が {len(similar_products)} 件見つかりました**")
            
            for i, prod in enumerate(similar_products[:5], 1):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"{i}. {prod['name']}")
                with col2:
                    st.write(f"平均 {prod['avg_daily']:.1f}体/日")
                with col3:
                    st.write(f"類似度 {prod['similarity']:.0f}%")
        else:
            st.info("類似商品が見つかりませんでした。カテゴリーの平均値から予測します。")
    else:
        similar_products = []
        st.info("👆 授与品名を入力すると、類似商品を検索します")
    
    st.markdown('<p class="section-header">③ 需要予測</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_period = st.selectbox(
            "予測期間",
            ["1ヶ月", "3ヶ月", "6ヶ月", "1年"],
            index=2
        )
    
    with col2:
        confidence_level = st.selectbox(
            "予測の保守性",
            ["楽観的", "標準", "保守的"],
            index=1
        )
    
    if st.button("🔮 新規授与品の需要を予測", type="primary", use_container_width=True):
        if not new_product_name or not new_product_name.strip():
            st.error("授与品名を入力してください")
        else:
            with st.spinner("予測中..."):
                forecast_result = forecast_new_product(
                    new_product_name,
                    new_product_category,
                    new_product_price,
                    similar_products,
                    forecast_period,
                    confidence_level
                )
                
                display_new_product_forecast(forecast_result, new_product_name, new_product_price)


def find_similar_products(name: str, category: str, price: int, description: str) -> list:
    """類似商品を探す"""
    
    if not name or not name.strip():
        return []
    
    if st.session_state.data_loader is None:
        return []
    
    df_items = st.session_state.data_loader.load_item_sales()
    
    if df_items.empty:
        return []
    
    product_col = '商品名'
    qty_col = '販売商品数'
    sales_col = '販売総売上'
    
    product_stats = df_items.groupby(product_col).agg({
        qty_col: ['sum', 'mean', 'count'],
        sales_col: 'sum'
    }).reset_index()
    
    product_stats.columns = ['name', 'total_qty', 'avg_daily', 'days_count', 'total_sales']
    
    product_stats['unit_price'] = product_stats['total_sales'] / product_stats['total_qty']
    product_stats['unit_price'] = product_stats['unit_price'].fillna(0)
    
    similar = []
    
    keywords = set(re.findall(r'[\u4e00-\u9fff]+', name.lower()))
    if description:
        keywords.update(re.findall(r'[\u4e00-\u9fff]+', description.lower()))
    
    for _, row in product_stats.iterrows():
        prod_name = row['name']
        
        name_keywords = set(re.findall(r'[\u4e00-\u9fff]+', prod_name.lower()))
        name_match = len(keywords & name_keywords) / max(len(keywords), 1) * 50
        
        if row['unit_price'] > 0:
            price_diff = abs(price - row['unit_price']) / price
            price_match = max(0, (1 - price_diff)) * 30
        else:
            price_match = 0
        
        category_keywords = {
            "お守り": ["守", "お守り", "まもり"],
            "御朱印": ["御朱印", "朱印"],
            "御朱印帳": ["御朱印帳", "朱印帳"],
            "おみくじ": ["おみくじ", "みくじ"],
            "絵馬": ["絵馬"],
            "お札": ["札", "お札"],
            "縁起物": ["縁起", "だるま", "招き猫"],
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


def forecast_new_product(name: str, category: str, price: int, 
                         similar_products: list, period: str, confidence: str) -> dict:
    """新規授与品の需要を予測"""
    
    period_days = {"1ヶ月": 30, "3ヶ月": 90, "6ヶ月": 180, "1年": 365}[period]
    confidence_factor = {"楽観的": 1.2, "標準": 1.0, "保守的": 0.7}[confidence]
    
    if similar_products:
        weighted_sum = sum(p['avg_daily'] * p['similarity'] for p in similar_products[:5])
        weight_total = sum(p['similarity'] for p in similar_products[:5])
        base_daily = weighted_sum / weight_total if weight_total > 0 else 1.0
    else:
        base_daily = CATEGORY_CHARACTERISTICS.get(category, {}).get('base_daily', 1.0)
    
    cat_char = CATEGORY_CHARACTERISTICS.get(category, {})
    seasonality = cat_char.get('seasonality', 'medium')
    
    if seasonality == 'high':
        month_factors = {1: 3.0, 2: 0.7, 3: 0.9, 4: 0.9, 5: 1.0, 6: 0.8,
                        7: 0.9, 8: 1.1, 9: 0.9, 10: 1.0, 11: 1.2, 12: 1.5}
    elif seasonality == 'medium':
        month_factors = {1: 1.5, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.1, 6: 0.9,
                        7: 1.0, 8: 1.1, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.2}
    else:
        month_factors = {i: 1.0 for i in range(1, 13)}
    
    daily_forecast = []
    total_qty = 0
    
    for i in range(period_days):
        target_date = date.today() + timedelta(days=i)
        month = target_date.month
        weekday = target_date.weekday()
        
        weekday_factor = 1.5 if weekday >= 5 else 1.0
        month_factor = month_factors.get(month, 1.0)
        
        pred = base_daily * weekday_factor * month_factor * confidence_factor
        pred = max(0, round(pred))
        
        daily_forecast.append({
            'date': target_date,
            'predicted': pred
        })
        
        total_qty += pred
    
    total_qty_rounded = round_up_to_50(total_qty)
    
    df_forecast = pd.DataFrame(daily_forecast)
    df_forecast['month'] = pd.to_datetime(df_forecast['date']).dt.to_period('M')
    monthly = df_forecast.groupby('month')['predicted'].sum().to_dict()
    
    return {
        'daily_forecast': daily_forecast,
        'total_qty': total_qty,
        'total_qty_rounded': total_qty_rounded,
        'avg_daily': total_qty / period_days,
        'period_days': period_days,
        'monthly': monthly,
        'base_daily': base_daily,
        'confidence': confidence,
        'similar_count': len(similar_products)
    }


def display_new_product_forecast(result: dict, product_name: str, price: int):
    """新規授与品の予測結果を表示"""
    
    st.success("✅ 予測完了！")
    
    st.write(f"### 📦 「{product_name}」の需要予測")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("予測販売総数", f"{result['total_qty_rounded']:,}体")
    col2.metric("予測売上", f"¥{result['total_qty_rounded'] * price:,.0f}")
    col3.metric("平均日販", f"{result['avg_daily']:.1f}体/日")
    col4.metric("予測期間", f"{result['period_days']}日間")
    
    if result['similar_count'] >= 3:
        st.info(f"📊 類似商品 {result['similar_count']} 件のデータを基に予測しました。信頼度: ⭐⭐⭐")
    elif result['similar_count'] >= 1:
        st.warning(f"📊 類似商品 {result['similar_count']} 件のデータを基に予測しました。信頼度: ⭐⭐")
    else:
        st.warning("📊 類似商品がなかったため、カテゴリーの平均値から予測しました。信頼度: ⭐")
    
    monthly_data = []
    for period, qty in result['monthly'].items():
        monthly_data.append({'月': str(period), '予測販売数': qty})
    
    df_monthly = pd.DataFrame(monthly_data)
    
    fig = px.bar(
        df_monthly, x='月', y='予測販売数',
        title='月別予測販売数',
        color='予測販売数',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### 📋 初回発注量の提案")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("少なめ（1ヶ月分）", f"{round_up_to_50(int(result['avg_daily'] * 30))}体")
    col2.metric("標準（3ヶ月分）", f"{round_up_to_50(int(result['avg_daily'] * 90))}体")
    col3.metric("多め（6ヶ月分）", f"{round_up_to_50(int(result['avg_daily'] * 180))}体")


# =============================================================================
# 高度な分析
# =============================================================================

def render_advanced_analysis():
    """高度な分析タブ"""
    st.markdown('<p class="section-header">🔬 高度な分析</p>', unsafe_allow_html=True)
    
    if not ADVANCED_ANALYSIS_AVAILABLE:
        st.warning("demand_analyzer.pyモジュールが見つかりません。")
        return
    
    sales_data = st.session_state.get('sales_data')
    
    if sales_data is None or sales_data.empty:
        st.info("「既存授与品の分析・予測」タブで授与品を選択してください。")
        return
    
    try:
        df_items = st.session_state.data_loader.load_item_sales()
        internal = InternalAnalyzer(df_items)
        external = ExternalAnalyzer(df_items, None)
    except Exception as e:
        st.error(f"分析モジュールの初期化に失敗しました: {e}")
        return
    
    with st.expander("📊 **高度な分析を見る**", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📈 トレンド分析", "🗓️ 季節性分析", "🌤️ 外部要因分析"])
        
        with tab1:
            render_trend_analysis(internal)
        
        with tab2:
            render_seasonality_analysis(internal)
        
        with tab3:
            render_external_analysis(external)


def render_trend_analysis(internal):
    """トレンド分析を表示"""
    st.write("### 📈 販売トレンド分析")
    
    try:
        trend = internal.analyze_sales_trend()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("トレンド方向", trend['trend_direction'])
        col2.metric("成長率", f"{trend['growth_rate']}%")
        col3.metric("変動性", f"{trend['volatility']:.2f}")
        
        if 'monthly_data' in trend and not trend['monthly_data'].empty:
            fig = px.line(
                trend['monthly_data'], 
                x='period', 
                y='販売商品数',
                title='月別販売推移'
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"トレンド分析を実行できませんでした: {e}")


def render_seasonality_analysis(internal):
    """季節性分析を表示"""
    st.write("### 🗓️ 季節性分析")
    
    try:
        seasonality = internal.detect_seasonality()
        
        st.metric("季節性の強さ", f"{seasonality['seasonality_strength']:.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**月別係数**")
            monthly = seasonality['monthly_pattern']
            df_monthly = pd.DataFrame({
                '月': list(monthly.keys()),
                '係数': list(monthly.values())
            })
            fig = px.bar(df_monthly, x='月', y='係数', title='月別販売係数')
            fig.add_hline(y=1.0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**曜日別係数**")
            weekday = seasonality['weekday_pattern']
            df_weekday = pd.DataFrame({
                '曜日': list(weekday.keys()),
                '係数': list(weekday.values())
            })
            fig = px.bar(df_weekday, x='曜日', y='係数', title='曜日別販売係数')
            fig.add_hline(y=1.0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"季節性分析を実行できませんでした: {e}")


def render_external_analysis(external):
    """外部要因分析を表示"""
    st.write("### 🌤️ 外部要因分析")
    
    try:
        calendar_effect = external.analyze_calendar_effect()
        
        if calendar_effect.get('available', False):
            st.metric("休日の影響度", f"{calendar_effect['holiday_impact']:.2f}x")
        else:
            st.info("カレンダーデータがないため、外部要因分析は利用できません。")
    except Exception as e:
        st.warning(f"外部要因分析を実行できませんでした: {e}")


# =============================================================================
# 予測精度ダッシュボード
# =============================================================================

def render_accuracy_dashboard():
    """予測精度ダッシュボード"""
    
    st.markdown('<p class="section-header">📈 予測精度ダッシュボード</p>', unsafe_allow_html=True)
    
    try:
        service = st.session_state.data_loader.service
        result = service.spreadsheets().values().get(
            spreadsheetId=st.session_state.data_loader.spreadsheet_id,
            range="'forecast_accuracy'!A:H"
        ).execute()
        
        values = result.get('values', [])
        
        if len(values) <= 1:
            st.info("""
            📊 まだ予測精度データがありません。
            
            自動学習システムが稼働すると、ここに予測精度が表示されます。
            """)
            return
        
        headers = values[0]
        df = pd.DataFrame(values[1:], columns=headers)
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['predicted_qty'] = pd.to_numeric(df['predicted_qty'], errors='coerce')
        df['actual_qty'] = pd.to_numeric(df['actual_qty'], errors='coerce')
        df['diff_pct'] = pd.to_numeric(df['diff_pct'], errors='coerce')
        
        st.write("### 過去30日間の予測精度")
        
        recent = df[df['date'] >= (datetime.now() - timedelta(days=30))]
        
        if not recent.empty:
            avg_error = recent['diff_pct'].abs().mean()
            total_predicted = recent['predicted_qty'].sum()
            total_actual = recent['actual_qty'].sum()
            accuracy = 100 - avg_error
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("平均誤差率", f"{avg_error:.1f}%")
            col2.metric("予測精度", f"{accuracy:.1f}%")
            col3.metric("予測合計", f"{total_predicted:.0f}体")
            col4.metric("実績合計", f"{total_actual:.0f}体")
            
            fig = go.Figure()
            
            daily = recent.groupby('date').agg({
                'predicted_qty': 'sum',
                'actual_qty': 'sum'
            }).reset_index()
            
            fig.add_trace(go.Scatter(
                x=daily['date'],
                y=daily['predicted_qty'],
                mode='lines+markers',
                name='予測',
                line=dict(color='#4285F4')
            ))
            
            fig.add_trace(go.Scatter(
                x=daily['date'],
                y=daily['actual_qty'],
                mode='lines+markers',
                name='実績',
                line=dict(color='#4CAF50')
            ))
            
            fig.update_layout(
                title='予測 vs 実績（日別）',
                xaxis_title='日付',
                yaxis_title='販売数（体）'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("過去30日間のデータがありません")
    
    except Exception as e:
        st.info("""
        📊 予測精度ダッシュボードを表示するには、自動学習システムのセットアップが必要です。
        """)


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
    
    # バージョン情報
    version_info = "v16 (個別納品計画・発注ロジック強化版)"
    if VERTEX_AI_AVAILABLE:
        version_info += " | 🚀 Vertex AI: 有効"
    else:
        version_info += " | ⚠️ Vertex AI: 未設定"
    
    st.caption(f"⛩️ 酒列磯前神社 授与品管理システム {version_info}")


if __name__ == "__main__":
    main()
