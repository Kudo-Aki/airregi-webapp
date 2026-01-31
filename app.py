"""
Airレジ 売上分析・需要予測 Webアプリ（v20: 精度向上版 - 0埋め・トレンド・正月日別対応）

v19からの変更点（v20新機能）:
1. 【予測精度の大幅向上】
   - 0埋め処理: 売上がない日を0で補完し、正確な曜日・季節係数を計算
   - 欠品期間除外: 在庫切れ期間を学習から除外（需要と供給制約を分離）
   - トレンド係数: 前年同期比の成長率を予測に反映
   - 正月日別係数: 1/1〜1/7を日別に係数設定（元日ピーク対応）
   
2. 【発注支援機能】
   - 発注点（リオーダーポイント）の自動計算
   - 安全在庫の算出（サービスレベル別）
   - 推奨発注量の表示

3. 【UI改善】
   - v20精度向上オプションのexpander追加
   - 欠品期間の入力UI
   - トレンド・正月日別のON/OFF切り替え

v19からの維持機能:
- ベースライン計算（中央値/トリム平均）
- 特別期間係数の自動計算
- 分位点予測（P50/P80/P90）
- 発注モード選択（滞留回避/バランス/欠品回避）
- 簡易バックテスト機能（MAPE表示）
- st.secrets対応（Streamlit Cloud推奨）
- HTML注入対策
- st.formによる予測パラメータ設定

v18以前からの維持機能:
- ファクトチェック用プロンプト生成機能
- 複数授与品選択時に「合算」「個別」を選択可能
- 予測期間を「日数指定」「期間指定」で選択可能
- 新規授与品の需要予測（類似商品ベース）
- 予測精度ダッシュボード
- 高度な分析タブ
- グループ機能（合算/単独）
- 年次比較
- Airレジ・郵送の内訳表示
- すべての方法で比較（マトリックス形式）
- 納品計画
- Vertex AI AutoML Forecasting連携
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
import hashlib
import html  # XSS対策用

# scipy（統計処理用）- オプショナル
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("scipyがインストールされていません。一部の統計機能が制限されます。")

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

# =============================================================================
# セキュリティ強化: GCP認証のモダン化
# =============================================================================

def get_gcp_credentials():
    """
    GCP認証情報を取得（優先順位: st.secrets > 環境変数 > ファイル）
    
    Streamlit Cloud推奨の認証方式に対応
    """
    from google.oauth2 import service_account
    
    credentials = None
    auth_method = None
    
    try:
        # 1. st.secretsから取得（Streamlit Cloud推奨）
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            try:
                service_account_info = dict(st.secrets['gcp_service_account'])
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                auth_method = "st.secrets"
                logger.info("GCP認証: st.secretsから取得成功")
                return credentials, auth_method
            except Exception as e:
                logger.warning(f"st.secretsからの認証取得失敗: {e}")
        
        # 2. 環境変数からJSON文字列を取得
        env_json = os.environ.get('VERTEX_AI_SERVICE_ACCOUNT_JSON')
        if env_json:
            try:
                service_account_info = json.loads(env_json)
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                auth_method = "環境変数(JSON)"
                logger.info("GCP認証: 環境変数から取得成功")
                return credentials, auth_method
            except Exception as e:
                logger.warning(f"環境変数からの認証取得失敗: {e}")
        
        # 3. サービスアカウントファイルから取得（従来方式）
        sa_file = VERTEX_AI_CONFIG['service_account_file']
        if os.path.exists(sa_file):
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    sa_file,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                auth_method = "ファイル"
                logger.info(f"GCP認証: ファイルから取得成功 ({sa_file})")
                return credentials, auth_method
            except Exception as e:
                logger.warning(f"ファイルからの認証取得失敗: {e}")
        
        # 4. Application Default Credentials（ADC）を試行
        try:
            import google.auth
            credentials, project = google.auth.default(
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            auth_method = "ADC"
            logger.info("GCP認証: ADCから取得成功")
            return credentials, auth_method
        except Exception as e:
            logger.warning(f"ADCからの認証取得失敗: {e}")
    
    except Exception as e:
        logger.error(f"GCP認証取得中の予期しないエラー: {e}")
    
    return None, None


def safe_html(text: str) -> str:
    """
    HTML注入対策: テキストをエスケープ
    
    Args:
        text: エスケープするテキスト
    
    Returns:
        エスケープ済みテキスト
    """
    if text is None:
        return ""
    return html.escape(str(text))


def mask_sensitive_value(value: str, visible_chars: int = 4) -> str:
    """
    機密情報をマスク表示
    
    Args:
        value: マスクする値
        visible_chars: 表示する文字数
    
    Returns:
        マスク済みの値
    """
    if not value or len(value) <= visible_chars:
        return "***"
    return value[:visible_chars] + "***"


try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic.schema import predict as predict_schema
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
    from google.oauth2 import service_account
    from google.api_core import exceptions as google_exceptions
    
    # 認証情報を取得
    credentials, auth_method = get_gcp_credentials()
    
    if credentials and VERTEX_AI_CONFIG['project_id'] and VERTEX_AI_CONFIG['endpoint_id']:
        # Vertex AI初期化
        aiplatform.init(
            project=VERTEX_AI_CONFIG['project_id'],
            location=VERTEX_AI_CONFIG['location'],
            credentials=credentials
        )
        VERTEX_AI_AVAILABLE = True
        logger.info(f"Vertex AI AutoML Forecasting: 初期化成功 (認証方式: {auth_method})")
    else:
        if not credentials:
            logger.warning("Vertex AI: 認証情報が見つかりません")
        if not VERTEX_AI_CONFIG['project_id']:
            logger.warning("Vertex AI: project_idが設定されていません")
        if not VERTEX_AI_CONFIG['endpoint_id']:
            logger.warning("Vertex AI: endpoint_idが設定されていません")
        
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
            
            # 新しい認証方式を使用
            credentials, _ = get_gcp_credentials()
            if credentials is None:
                raise RuntimeError("GCP認証情報が取得できませんでした")
            
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
    
    ★v19改善: 中央値ベースの頑健な予測
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
# 【v19新機能】予測精度強化のための統計関数群
# =============================================================================

def calculate_robust_baseline(values: np.ndarray, method: str = 'median') -> float:
    """
    外れ値に強いベースライン計算
    
    Args:
        values: 販売数の配列
        method: 計算方法 ('median', 'trimmed_mean', 'iqr_mean')
    
    Returns:
        頑健なベースライン値
    """
    if len(values) == 0:
        return 1.0
    
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return 1.0
    
    if method == 'median':
        # 中央値（外れ値に最も強い）
        return float(np.median(values))
    
    elif method == 'trimmed_mean':
        # トリム平均（上下10%を除外）
        if SCIPY_AVAILABLE:
            try:
                return float(stats.trim_mean(values, proportiontocut=0.1))
            except:
                pass
        # scipyがない場合の代替実装
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        trim_count = int(n * 0.1)
        if trim_count > 0 and n > trim_count * 2:
            trimmed = sorted_vals[trim_count:-trim_count]
            return float(np.mean(trimmed))
        return float(np.median(values))
    
    elif method == 'iqr_mean':
        # IQR内の値のみで平均（四分位範囲内）
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered = values[(values >= lower_bound) & (values <= upper_bound)]
        if len(filtered) > 0:
            return float(np.mean(filtered))
        return float(np.median(values))
    
    else:
        return float(np.mean(values))


def calculate_robust_factor(group_values: np.ndarray, overall_baseline: float, 
                           min_samples: int = 5) -> float:
    """
    頑健な係数計算（サンプル数が少ない場合は1.0に縮退）
    
    Args:
        group_values: グループの値
        overall_baseline: 全体のベースライン
        min_samples: 最小サンプル数
    
    Returns:
        係数（1.0に近づくスムージング付き）
    """
    if len(group_values) < min_samples or overall_baseline <= 0:
        return 1.0
    
    # 中央値ベースで係数を計算
    group_median = np.median(group_values)
    raw_factor = group_median / overall_baseline if overall_baseline > 0 else 1.0
    
    # サンプル数に応じてスムージング（少ないほど1.0に近づける）
    confidence = min(1.0, len(group_values) / (min_samples * 2))
    smoothed_factor = confidence * raw_factor + (1 - confidence) * 1.0
    
    # 極端な値を抑制（0.3〜3.0の範囲に収める）
    return float(max(0.3, min(3.0, smoothed_factor)))


def identify_special_periods(df: pd.DataFrame) -> Dict[str, List[date]]:
    """
    特別期間（正月、お盆、七五三等）の日付を特定
    
    Args:
        df: 日付を含むDataFrame
    
    Returns:
        特別期間名をキー、日付リストを値とする辞書
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    special_periods = {
        'new_year': [],     # 正月（1/1〜1/7）
        'obon': [],         # お盆（8/13〜8/16）
        'shichigosan': [],  # 七五三（11/10〜11/20）
        'golden_week': [],  # ゴールデンウィーク（5/3〜5/5）
        'year_end': [],     # 年末（12/28〜12/31）
    }
    
    for d in df['date'].unique():
        d = pd.Timestamp(d)
        if d.month == 1 and d.day <= 7:
            special_periods['new_year'].append(d.date())
        elif d.month == 8 and 13 <= d.day <= 16:
            special_periods['obon'].append(d.date())
        elif d.month == 11 and 10 <= d.day <= 20:
            special_periods['shichigosan'].append(d.date())
        elif d.month == 5 and 3 <= d.day <= 5:
            special_periods['golden_week'].append(d.date())
        elif d.month == 12 and d.day >= 28:
            special_periods['year_end'].append(d.date())
    
    return special_periods


def calculate_special_period_factors(df: pd.DataFrame, overall_baseline: float,
                                     auto_calculate: bool = True) -> Dict[str, float]:
    """
    特別期間の係数を計算（過去データから自動算出 or 固定値）
    
    Args:
        df: 売上データ
        overall_baseline: 全体のベースライン
        auto_calculate: Trueなら過去データから計算、Falseなら固定値
    
    Returns:
        特別期間名をキー、係数を値とする辞書
    """
    # デフォルト（固定）係数
    default_factors = {
        'new_year': 3.0,      # 正月
        'obon': 1.5,          # お盆
        'shichigosan': 1.3,   # 七五三
        'golden_week': 1.3,   # ゴールデンウィーク
        'year_end': 1.5,      # 年末
        'normal': 1.0         # 通常日
    }
    
    if not auto_calculate or overall_baseline <= 0:
        return default_factors
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 特別期間フラグを付与
    def get_period_type(d):
        if d.month == 1 and d.day <= 7:
            return 'new_year'
        elif d.month == 8 and 13 <= d.day <= 16:
            return 'obon'
        elif d.month == 11 and 10 <= d.day <= 20:
            return 'shichigosan'
        elif d.month == 5 and 3 <= d.day <= 5:
            return 'golden_week'
        elif d.month == 12 and d.day >= 28:
            return 'year_end'
        return 'normal'
    
    df['period_type'] = df['date'].apply(get_period_type)
    
    # 各期間の係数を計算
    calculated_factors = {}
    for period_name in default_factors.keys():
        period_data = df[df['period_type'] == period_name]['販売商品数'].values
        
        if len(period_data) >= 3:  # 最低3サンプル必要
            period_median = np.median(period_data)
            factor = period_median / overall_baseline if overall_baseline > 0 else 1.0
            # 極端な値を抑制（0.5〜5.0の範囲）
            factor = max(0.5, min(5.0, factor))
            calculated_factors[period_name] = float(factor)
        else:
            # サンプル不足時はデフォルト値を使用
            calculated_factors[period_name] = default_factors[period_name]
    
    return calculated_factors


def calculate_prediction_quantiles(predictions: np.ndarray, residuals: np.ndarray,
                                   quantiles: List[float] = [0.5, 0.8, 0.9]) -> Dict[str, np.ndarray]:
    """
    予測の分位点を計算
    
    Args:
        predictions: 点予測値の配列
        residuals: 過去の残差（実績-予測）
        quantiles: 計算する分位点のリスト
    
    Returns:
        分位点名をキー、予測値配列を値とする辞書
    """
    result = {'predicted': predictions}
    
    if len(residuals) < 5:
        # 残差データ不足時は点予測に基づく簡易計算
        std_estimate = np.std(predictions) * 0.2  # 予測値の20%を標準偏差と仮定
        
        # 正規分布の分位点を計算（scipyがない場合の代替）
        # Z値の近似: P50=0, P80≈0.84, P90≈1.28
        z_values = {0.5: 0.0, 0.8: 0.84, 0.9: 1.28}
        
        for q in quantiles:
            if SCIPY_AVAILABLE:
                z_score = stats.norm.ppf(q)
            else:
                z_score = z_values.get(q, 0.0)
            result[f'p{int(q*100)}'] = predictions + z_score * std_estimate
    else:
        # 残差の分布から分位点を計算
        residual_quantiles = {q: np.percentile(residuals, q * 100) for q in quantiles}
        for q in quantiles:
            result[f'p{int(q*100)}'] = predictions + residual_quantiles[q]
    
    # 負の値を0に補正
    for key in result:
        result[key] = np.maximum(0, result[key])
    
    return result


def run_simple_backtest(df: pd.DataFrame, holdout_days: int = 14,
                       forecast_func=None) -> Dict[str, Any]:
    """
    簡易バックテストを実行してMAPEを計算
    
    Args:
        df: 売上データ
        holdout_days: ホールドアウト日数
        forecast_func: 予測関数（Noneなら改善版フォールバックを使用）
    
    Returns:
        バックテスト結果（MAPE、MAE、詳細データ）
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if len(df) < holdout_days + 30:  # 最低30日の学習データが必要
        return {
            'mape': None,
            'mae': None,
            'message': 'データ不足のためバックテスト不可',
            'holdout_days': holdout_days,
            'available': False,
            'residuals': []
        }
    
    # データを分割
    train_df = df.iloc[:-holdout_days].copy()
    test_df = df.iloc[-holdout_days:].copy()
    
    # 予測を実行（backtest_days=0で再帰を防止）
    if forecast_func is None:
        forecast_result = forecast_with_seasonality_enhanced(
            train_df, holdout_days, 
            baseline_method='median',
            auto_special_factors=True,
            include_quantiles=False,  # バックテスト内では分位点不要
            backtest_days=0  # ★重要: 再帰を防止
        )
    else:
        forecast_result = forecast_func(train_df, holdout_days)
    
    # 予測と実績を比較
    test_df = test_df.reset_index(drop=True)
    forecast_result = forecast_result.reset_index(drop=True)
    
    actual = test_df['販売商品数'].values
    predicted = forecast_result['predicted'].values[:len(actual)]
    
    # MAPE計算（0除算対策）
    non_zero_mask = actual > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs(actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask]) * 100
    else:
        mape = None
    
    # MAE計算
    mae = np.mean(np.abs(actual - predicted))
    
    # 残差を保存（分位点計算用）
    residuals = actual - predicted
    
    return {
        'mape': float(mape) if mape is not None else None,
        'mae': float(mae),
        'residuals': residuals.tolist(),
        'holdout_days': holdout_days,
        'actual': actual.tolist(),
        'predicted': predicted.tolist(),
        'message': f'直近{holdout_days}日間でバックテスト実施',
        'available': True
    }


# =============================================================================
# 【v20新機能】精度向上のための追加機能群
# =============================================================================

def fill_missing_dates(df: pd.DataFrame, fill_value: int = 0) -> Tuple[pd.DataFrame, List[date]]:
    """
    【v20新機能】日付の欠損を0で埋める（0埋め処理）
    
    Airレジでは売上がない日は日付が抜けるため、
    全日付を埋めて正確な曜日・季節係数を計算する。
    
    Args:
        df: 売上データ（date, 販売商品数を含む）
        fill_value: 欠損日に埋める値（デフォルト0）
    
    Returns:
        (0埋め後のDataFrame, 埋めた日付のリスト)
    """
    if df is None or df.empty:
        return df, []
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 日付の範囲を取得
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    # 全日付の連続シーケンスを作成
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # 元データに存在する日付
    existing_dates = set(df['date'].dt.date)
    
    # 欠損日を特定
    missing_dates = [d.date() for d in all_dates if d.date() not in existing_dates]
    
    if not missing_dates:
        return df, []
    
    # 欠損日のデータを作成
    missing_data = []
    for d in missing_dates:
        missing_data.append({
            'date': pd.Timestamp(d),
            '販売商品数': fill_value
        })
    
    missing_df = pd.DataFrame(missing_data)
    
    # 元データと結合
    filled_df = pd.concat([df, missing_df], ignore_index=True)
    filled_df = filled_df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"0埋め処理: {len(missing_dates)}日分の欠損を補完しました")
    
    return filled_df, missing_dates


def exclude_stockout_periods(df: pd.DataFrame, stockout_periods: List[Tuple[date, date]]) -> pd.DataFrame:
    """
    【v20新機能】欠品期間を学習データから除外
    
    欠品（在庫切れ）期間は「需要がなかった」のではなく「供給できなかった」ため、
    学習データから除外して正確な需要を推定する。
    
    Args:
        df: 売上データ
        stockout_periods: 欠品期間のリスト [(開始日, 終了日), ...]
    
    Returns:
        欠品期間を除外したDataFrame（除外行はNaN扱い）
    """
    if not stockout_periods:
        return df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 欠品フラグを初期化
    df['is_stockout'] = False
    
    for start_date, end_date in stockout_periods:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
        df.loc[mask, 'is_stockout'] = True
    
    # 欠品日の販売数をNaNに（学習から除外）
    excluded_count = df['is_stockout'].sum()
    df.loc[df['is_stockout'], '販売商品数'] = np.nan
    
    if excluded_count > 0:
        logger.info(f"欠品期間除外: {excluded_count}日分を学習対象外にしました")
    
    return df


def calculate_trend_factor(df: pd.DataFrame, comparison_window_days: int = 60) -> Tuple[float, Dict[str, Any]]:
    """
    【v20新機能】トレンド係数の計算（前年同期比）
    
    直近の売上傾向と前年同期を比較し、成長率を算出する。
    これにより、年々の増減トレンドを予測に反映できる。
    
    Args:
        df: 売上データ
        comparison_window_days: 比較期間（日数）
    
    Returns:
        (トレンド係数, 詳細情報の辞書)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # NaN（欠品日）を除外して計算
    df_valid = df.dropna(subset=['販売商品数'])
    
    # データが1年+比較期間未満の場合はトレンドなし
    if len(df_valid) < 365 + comparison_window_days:
        return 1.0, {
            'available': False,
            'message': 'データ不足（1年以上必要）',
            'trend_factor': 1.0,
            'current_mean': None,
            'last_year_mean': None
        }
    
    last_date = df_valid['date'].max()
    
    # 直近N日間
    current_period_start = last_date - pd.Timedelta(days=comparison_window_days)
    current_vals = df_valid[df_valid['date'] > current_period_start]['販売商品数']
    
    # 前年同期間
    last_year_end = last_date - pd.Timedelta(days=365)
    last_year_start = last_year_end - pd.Timedelta(days=comparison_window_days)
    last_year_vals = df_valid[
        (df_valid['date'] > last_year_start) & 
        (df_valid['date'] <= last_year_end)
    ]['販売商品数']
    
    if len(current_vals) < 10 or len(last_year_vals) < 10:
        return 1.0, {
            'available': False,
            'message': '比較期間のデータ不足',
            'trend_factor': 1.0,
            'current_mean': None,
            'last_year_mean': None
        }
    
    current_mean = float(current_vals.mean())
    last_year_mean = float(last_year_vals.mean())
    
    if last_year_mean <= 0:
        return 1.0, {
            'available': False,
            'message': '前年同期の売上が0',
            'trend_factor': 1.0,
            'current_mean': current_mean,
            'last_year_mean': 0
        }
    
    # トレンド係数（0.7〜1.5にクリップして極端な変動を抑制）
    raw_trend = current_mean / last_year_mean
    trend_factor = max(0.7, min(1.5, raw_trend))
    
    # 変化率（%）
    change_rate = (trend_factor - 1.0) * 100
    
    return trend_factor, {
        'available': True,
        'message': f'前年比 {trend_factor:.1%}（{change_rate:+.1f}%）',
        'trend_factor': trend_factor,
        'raw_trend': raw_trend,
        'current_mean': current_mean,
        'last_year_mean': last_year_mean,
        'comparison_window_days': comparison_window_days
    }


def get_period_type_v2(d: pd.Timestamp) -> str:
    """
    【v20新機能】特別期間の判定（正月は日別）
    
    正月（1/1〜1/7）は日によって需要が大きく異なるため、
    日別に係数を持つ。
    
    Args:
        d: 日付
    
    Returns:
        特別期間タイプ（'new_year_d1'〜'new_year_d7'、'obon'、'normal'など）
    """
    d = pd.Timestamp(d)
    
    # 正月は日別（1/1〜1/7）
    if d.month == 1 and 1 <= d.day <= 7:
        return f'new_year_d{d.day}'
    
    # お盆（8/13〜8/16）
    elif d.month == 8 and 13 <= d.day <= 16:
        return 'obon'
    
    # 七五三（11/10〜11/20）
    elif d.month == 11 and 10 <= d.day <= 20:
        return 'shichigosan'
    
    # ゴールデンウィーク（5/3〜5/5）
    elif d.month == 5 and 3 <= d.day <= 5:
        return 'golden_week'
    
    # 年末（12/28〜12/31）
    elif d.month == 12 and d.day >= 28:
        return 'year_end'
    
    return 'normal'


def calculate_special_period_factors_v2(
    df: pd.DataFrame, 
    overall_baseline: float,
    auto_calculate: bool = True
) -> Dict[str, float]:
    """
    【v20新機能】特別期間係数の計算（正月日別対応版）
    
    正月（1/1〜1/7）を日別に分けて係数を計算することで、
    元日のピークから徐々に下がる需要パターンを捉える。
    
    Args:
        df: 売上データ
        overall_baseline: 全体のベースライン
        auto_calculate: 自動計算するか
    
    Returns:
        特別期間係数の辞書（正月は日別）
    """
    # デフォルト係数（正月は日別プロファイル）
    default_factors = {
        'new_year_d1': 5.0,   # 元日（ピーク）
        'new_year_d2': 4.5,   # 1/2
        'new_year_d3': 4.0,   # 1/3（三が日最終日）
        'new_year_d4': 2.5,   # 1/4
        'new_year_d5': 2.0,   # 1/5
        'new_year_d6': 1.8,   # 1/6
        'new_year_d7': 1.5,   # 1/7
        'obon': 1.5,          # お盆
        'shichigosan': 1.3,   # 七五三
        'golden_week': 1.3,   # ゴールデンウィーク
        'year_end': 1.5,      # 年末
        'normal': 1.0         # 通常日
    }
    
    if not auto_calculate or overall_baseline <= 0:
        return default_factors
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # NaN（欠品日）を除外
    df = df.dropna(subset=['販売商品数'])
    
    # 特別期間フラグを付与（v2版：正月日別）
    df['period_type'] = df['date'].apply(get_period_type_v2)
    
    # 各期間の係数を計算
    calculated_factors = {}
    
    for period_name in default_factors.keys():
        period_data = df[df['period_type'] == period_name]['販売商品数'].values
        
        # 正月の日別は1サンプル以上あれば計算（年数分しかない）
        min_samples = 1 if period_name.startswith('new_year_d') else 3
        
        if len(period_data) >= min_samples:
            period_median = np.median(period_data)
            factor = period_median / overall_baseline if overall_baseline > 0 else 1.0
            
            # 極端な値を抑制（0.5〜8.0の範囲、正月は上限を高く）
            max_factor = 8.0 if period_name.startswith('new_year') else 5.0
            factor = max(0.5, min(max_factor, factor))
            calculated_factors[period_name] = float(factor)
        else:
            # サンプル不足時はデフォルト値を使用
            calculated_factors[period_name] = default_factors[period_name]
    
    return calculated_factors


def calculate_reorder_point(
    prediction_df: pd.DataFrame,
    residuals: List[float],
    lead_time_days: int = 14,
    service_level: float = 0.95
) -> Dict[str, Any]:
    """
    【v20新機能】発注点（リオーダーポイント）と安全在庫の計算
    
    発注点 = リードタイム中の予測需要 + 安全在庫
    安全在庫 = 安全係数(Z) × RMSE × √リードタイム
    
    Args:
        prediction_df: 予測結果のDataFrame
        residuals: バックテストの残差リスト
        lead_time_days: リードタイム（発注から納品までの日数）
        service_level: サービスレベル（0.90, 0.95, 0.99）
    
    Returns:
        発注点計算結果の辞書
    """
    # サービスレベルに応じた安全係数（Z値）
    z_scores = {
        0.90: 1.28,   # 90%サービスレベル
        0.95: 1.65,   # 95%サービスレベル（推奨）
        0.99: 2.33    # 99%サービスレベル
    }
    z = z_scores.get(service_level, 1.65)
    
    # リードタイム期間の予測需要合計
    if len(prediction_df) < lead_time_days:
        lead_time_demand = prediction_df['predicted'].sum()
        actual_days = len(prediction_df)
    else:
        lead_time_demand = prediction_df.iloc[:lead_time_days]['predicted'].sum()
        actual_days = lead_time_days
    
    # 予測誤差（RMSE）の計算
    if not residuals or len(residuals) < 7:
        # データがない場合は予測の20%を誤差と仮定
        daily_demand = lead_time_demand / actual_days if actual_days > 0 else 1
        rmse = daily_demand * 0.2
        rmse_source = 'estimated'
    else:
        residuals_array = np.array(residuals)
        rmse = float(np.sqrt(np.mean(residuals_array ** 2)))
        rmse_source = 'calculated'
    
    # 安全在庫
    safety_stock = z * rmse * np.sqrt(actual_days)
    
    # 発注点
    reorder_point = lead_time_demand + safety_stock
    
    # 日別予測平均
    daily_avg = lead_time_demand / actual_days if actual_days > 0 else 0
    
    return {
        'lead_time_days': actual_days,
        'lead_time_demand': int(round(lead_time_demand)),
        'safety_stock': int(round(safety_stock)),
        'reorder_point': int(round(reorder_point)),
        'service_level': service_level,
        'service_level_pct': f'{service_level*100:.0f}%',
        'z_score': z,
        'rmse': float(rmse),
        'rmse_source': rmse_source,
        'daily_avg': float(daily_avg),
        'message': (
            f"📦 発注推奨\n"
            f"├─ 予測需要（{actual_days}日間）: {int(lead_time_demand):,}個\n"
            f"├─ 安全在庫（{service_level*100:.0f}%SL）: +{int(safety_stock):,}個\n"
            f"└─ 推奨発注点: {int(reorder_point):,}個"
        )
    }


def forecast_with_seasonality_enhanced(
    df: pd.DataFrame, 
    periods: int,
    baseline_method: str = 'median',
    auto_special_factors: bool = True,
    include_quantiles: bool = False,
    order_mode: str = 'balanced',
    backtest_days: int = 14,
    # v20新規オプション（既存互換のためデフォルトはFalse/None）
    enable_zero_fill: bool = True,
    stockout_periods: Optional[List[Tuple[date, date]]] = None,
    enable_trend: bool = True,
    use_daily_new_year: bool = True,
    trend_window_days: int = 60
) -> pd.DataFrame:
    """
    【v20精度強化版】季節性考慮予測（0埋め・欠品除外・トレンド・正月日別対応）
    
    v19からの追加機能:
    - 0埋め処理: 売上0の日を自動補完して正確な係数計算
    - 欠品期間除外: 在庫切れ期間を学習から除外
    - トレンド係数: 前年同期比の成長率を反映
    - 正月日別係数: 1/1〜1/7を日別に係数設定（元日ピーク対応）
    
    Args:
        df: 売上データ（date, 販売商品数を含む）
        periods: 予測日数
        baseline_method: ベースライン計算方法 ('median', 'trimmed_mean', 'iqr_mean', 'mean')
        auto_special_factors: 特別期間係数を過去データから自動計算するか
        include_quantiles: 分位点予測を含めるか
        order_mode: 発注モード ('conservative'=P50, 'balanced'=P80, 'aggressive'=P90)
        backtest_days: バックテスト日数（0なら実行しない）
        enable_zero_fill: 【v20】日付欠損を0で埋めるか（推奨: True）
        stockout_periods: 【v20】欠品期間のリスト [(開始日, 終了日), ...]
        enable_trend: 【v20】トレンド係数を適用するか（推奨: True）
        use_daily_new_year: 【v20】正月を日別係数にするか（推奨: True）
        trend_window_days: 【v20】トレンド計算の比較期間（日数）
    
    Returns:
        予測結果のDataFrame
    """
    df = df.copy()
    
    # データフレームの検証
    if df is None or df.empty:
        raise ValueError("売上データが空です")
    
    if 'date' not in df.columns:
        raise ValueError("'date'列が見つかりません")
    
    if '販売商品数' not in df.columns:
        raise ValueError("'販売商品数'列が見つかりません")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # ========== v20新機能1: 0埋め処理 ==========
    missing_dates = []
    if enable_zero_fill:
        df, missing_dates = fill_missing_dates(df, fill_value=0)
    
    # ========== v20新機能2: 欠品期間の除外 ==========
    if stockout_periods:
        df = exclude_stockout_periods(df, stockout_periods)
    
    # データが少なすぎる場合のフォールバック
    # NaNを除外してカウント
    valid_count = df['販売商品数'].notna().sum()
    if valid_count < 7:
        logger.warning(f"有効データが少なすぎます（{valid_count}件）。シンプルな予測にフォールバックします。")
        # 単純平均で予測（NaN除外）
        avg_value = df['販売商品数'].dropna().mean() if valid_count > 0 else 1.0
        avg_value = max(1.0, avg_value)
        
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        predictions = [{'date': d, 'predicted': round(avg_value)} for d in future_dates]
        result_df = pd.DataFrame(predictions)
        result_df.attrs['backtest'] = {'mape': None, 'available': False, 'message': 'データ不足'}
        result_df.attrs['v20_features'] = {
            'zero_fill': enable_zero_fill,
            'missing_dates_count': len(missing_dates),
            'stockout_excluded': stockout_periods is not None,
            'trend_applied': False,
            'daily_new_year': use_daily_new_year
        }
        return result_df
    
    # NaNを除外した値で計算
    valid_values = df['販売商品数'].dropna().values
    
    # ========== 1. 頑健なベースライン計算 ==========
    overall_baseline = calculate_robust_baseline(valid_values, method=baseline_method)
    
    if overall_baseline <= 0:
        overall_baseline = 1.0
    
    # ========== v20新機能3: トレンド係数の計算 ==========
    trend_factor = 1.0
    trend_info = {'available': False, 'trend_factor': 1.0}
    if enable_trend:
        trend_factor, trend_info = calculate_trend_factor(df, comparison_window_days=trend_window_days)
    
    # ========== 2. 曜日・月列を先に追加 ==========
    df['weekday'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # ========== 3. 特別期間を除外した通常日のベースライン ==========
    def is_special_day(d):
        d = pd.Timestamp(d)
        if d.month == 1 and d.day <= 7:  # 正月
            return True
        if d.month == 8 and 13 <= d.day <= 16:  # お盆
            return True
        if d.month == 11 and 10 <= d.day <= 20:  # 七五三
            return True
        if d.month == 5 and 3 <= d.day <= 5:  # GW
            return True
        if d.month == 12 and d.day >= 28:  # 年末
            return True
        return False
    
    df['is_special'] = df['date'].apply(is_special_day)
    # NaN（欠品日）と特別期間を除外
    normal_df = df[(~df['is_special']) & (df['販売商品数'].notna())].copy()
    
    if len(normal_df) > 10:
        # 通常日のみでベースラインを再計算（二重計上対策）
        normal_baseline = calculate_robust_baseline(
            normal_df['販売商品数'].values, 
            method=baseline_method
        )
    else:
        normal_baseline = overall_baseline
    
    # ========== 4. 曜日係数（頑健版） ==========
    weekday_factor = {}
    
    for wd in range(7):
        wd_values = normal_df[normal_df['weekday'] == wd]['販売商品数'].values
        weekday_factor[wd] = calculate_robust_factor(wd_values, normal_baseline, min_samples=3)
    
    # ========== 5. 月係数（頑健版） ==========
    month_factor = {}
    
    for m in range(1, 13):
        # 通常日のみで月係数を計算
        m_values = normal_df[normal_df['month'] == m]['販売商品数'].values
        month_factor[m] = calculate_robust_factor(m_values, normal_baseline, min_samples=5)
    
    # ========== 6. 特別期間係数（v20: 正月日別対応） ==========
    if use_daily_new_year:
        # v20版: 正月を日別に計算
        special_factors = calculate_special_period_factors_v2(
            df, normal_baseline, auto_calculate=auto_special_factors
        )
    else:
        # 従来版: 正月を1係数で計算
        special_factors = calculate_special_period_factors(
            df, normal_baseline, auto_calculate=auto_special_factors
        )
    
    # ========== 7. バックテスト（残差取得用） ==========
    residuals = np.array([])
    backtest_result = None
    
    # バックテスト用にNaNを除外したデータを使用
    df_for_backtest = df[df['販売商品数'].notna()].copy()
    if backtest_days > 0 and len(df_for_backtest) > backtest_days + 30:
        backtest_result = run_simple_backtest(df_for_backtest, backtest_days)
        if backtest_result['available']:
            residuals = np.array(backtest_result['residuals'])
    
    # ========== 8. 予測生成 ==========
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    predictions = []
    point_predictions = []
    
    for d in future_dates:
        weekday_f = weekday_factor.get(d.dayofweek, 1.0)
        month_f = month_factor.get(d.month, 1.0)
        
        # 特別期間の判定（v20: 正月日別対応）
        if use_daily_new_year:
            # v20版: 正月は日別
            period_type = get_period_type_v2(d)
            special_f = special_factors.get(period_type, special_factors.get('normal', 1.0))
        else:
            # 従来版: 正月は1係数
            special_f = special_factors.get('normal', 1.0)
            if d.month == 1 and d.day <= 7:
                special_f = special_factors.get('new_year', 3.0)
            elif d.month == 8 and 13 <= d.day <= 16:
                special_f = special_factors.get('obon', 1.5)
            elif d.month == 11 and 10 <= d.day <= 20:
                special_f = special_factors.get('shichigosan', 1.3)
            elif d.month == 5 and 3 <= d.day <= 5:
                special_f = special_factors.get('golden_week', 1.3)
            elif d.month == 12 and d.day >= 28:
                special_f = special_factors.get('year_end', 1.5)
        
        # 予測値計算（通常日ベースライン × 曜日係数 × 月係数 × 特別期間係数 × トレンド係数）
        pred = normal_baseline * weekday_f * month_f * special_f * trend_factor
        pred = max(0.1, pred)
        point_predictions.append(pred)
        
        predictions.append({
            'date': d,
            'predicted': round(pred),
            'weekday_factor': weekday_f,
            'month_factor': month_f,
            'special_factor': special_f,
            'trend_factor': trend_factor  # v20追加
        })
    
    result_df = pd.DataFrame(predictions)
    
    # ========== 9. 分位点予測の追加 ==========
    if include_quantiles:
        point_array = np.array(point_predictions)
        quantile_results = calculate_prediction_quantiles(
            point_array, residuals, quantiles=[0.5, 0.8, 0.9]
        )
        
        result_df['p50'] = quantile_results['p50'].round().astype(int)
        result_df['p80'] = quantile_results['p80'].round().astype(int)
        result_df['p90'] = quantile_results['p90'].round().astype(int)
        
        # 発注モードに応じた推奨値
        if order_mode == 'conservative':  # 滞留回避
            result_df['recommended'] = result_df['p50']
        elif order_mode == 'aggressive':  # 欠品回避
            result_df['recommended'] = result_df['p90']
        else:  # balanced
            result_df['recommended'] = result_df['p80']
    
    # ========== 10. メタデータの保存 ==========
    # バックテスト結果
    if backtest_result is not None:
        result_df.attrs['backtest'] = backtest_result
    else:
        result_df.attrs['backtest'] = {'mape': None, 'available': False, 'message': 'バックテスト未実行'}
    
    result_df.attrs['special_factors'] = special_factors
    result_df.attrs['baseline_method'] = baseline_method
    result_df.attrs['normal_baseline'] = normal_baseline
    
    # v20追加メタデータ
    result_df.attrs['v20_features'] = {
        'zero_fill': enable_zero_fill,
        'missing_dates_count': len(missing_dates),
        'stockout_excluded': stockout_periods is not None,
        'stockout_periods_count': len(stockout_periods) if stockout_periods else 0,
        'trend_applied': enable_trend and trend_info['available'],
        'trend_factor': trend_factor,
        'trend_info': trend_info,
        'daily_new_year': use_daily_new_year
    }
    
    # 発注点計算（残差がある場合）
    if backtest_result and backtest_result.get('available') and backtest_result.get('residuals'):
        reorder_info = calculate_reorder_point(
            result_df, 
            backtest_result['residuals'],
            lead_time_days=min(14, periods),
            service_level=0.95
        )
        result_df.attrs['reorder_point'] = reorder_info
    
    return result_df


# =============================================================================
# 予測方法の統合（Vertex AI対応）
# =============================================================================

def forecast_with_vertex_ai(
    df: pd.DataFrame,
    periods: int,
    method: str = "Vertex AI",
    product_id: str = "default",
    # v19新パラメータ
    baseline_method: str = 'median',
    auto_special_factors: bool = True,
    include_quantiles: bool = False,
    order_mode: str = 'balanced',
    backtest_days: int = 14,
    # v20新パラメータ
    enable_zero_fill: bool = True,
    stockout_periods: Optional[List[Tuple[date, date]]] = None,
    enable_trend: bool = True,
    use_daily_new_year: bool = True,
    trend_window_days: int = 60
) -> Tuple[pd.DataFrame, str]:
    """
    予測方法に応じた予測を実行
    
    Args:
        df: 売上データ
        periods: 予測日数
        method: 予測方法
        product_id: 商品識別子
        baseline_method: ベースライン計算方法（v19新規）
        auto_special_factors: 特別期間係数の自動計算（v19新規）
        include_quantiles: 分位点予測を含める（v19新規）
        order_mode: 発注モード（v19新規）
        backtest_days: バックテスト日数（v19新規）
        enable_zero_fill: 【v20】0埋め処理
        stockout_periods: 【v20】欠品期間リスト
        enable_trend: 【v20】トレンド係数
        use_daily_new_year: 【v20】正月日別係数
        trend_window_days: 【v20】トレンド比較期間
    
    Returns:
        予測DataFrame, 使用した予測方法の説明
    """
    if method == "🚀 Vertex AI（推奨）":
        predictions, used_vertex_ai, message = get_vertex_ai_prediction(df, periods, product_id, use_covariates=True)
        return predictions, message
    
    elif method == "移動平均法（シンプル）":
        return forecast_moving_average(df, periods), "移動平均法（統計モデル）"
    
    elif method == "季節性考慮（統計）":
        # 従来版（互換性維持）
        return forecast_with_seasonality_fallback(df, periods), "季節性考慮（統計モデル）"
    
    elif method == "🎯 季節性考慮（精度強化版）":
        # v20対応：精度強化版
        forecast = forecast_with_seasonality_enhanced(
            df, periods,
            baseline_method=baseline_method,
            auto_special_factors=auto_special_factors,
            include_quantiles=include_quantiles,
            order_mode=order_mode,
            backtest_days=backtest_days,
            # v20新パラメータ
            enable_zero_fill=enable_zero_fill,
            stockout_periods=stockout_periods,
            enable_trend=enable_trend,
            use_daily_new_year=use_daily_new_year,
            trend_window_days=trend_window_days
        )
        
        # メッセージ生成
        method_desc = f"季節性考慮（精度強化版・{baseline_method}ベース）"
        if auto_special_factors:
            method_desc += "・特別期間係数自動計算"
        if include_quantiles:
            mode_name = {'conservative': '滞留回避', 'balanced': 'バランス', 'aggressive': '欠品回避'}
            method_desc += f"・{mode_name.get(order_mode, order_mode)}モード"
        
        # v20機能の表示
        v20_features = []
        if enable_zero_fill:
            v20_features.append("0埋め")
        if enable_trend:
            # トレンド情報を取得
            if hasattr(forecast, 'attrs') and 'v20_features' in forecast.attrs:
                v20_info = forecast.attrs['v20_features']
                if v20_info.get('trend_applied'):
                    trend_factor = v20_info.get('trend_factor', 1.0)
                    if trend_factor != 1.0:
                        v20_features.append(f"トレンド{trend_factor:.1%}")
        if use_daily_new_year:
            v20_features.append("正月日別")
        
        if v20_features:
            method_desc += "・" + "・".join(v20_features)
        
        # バックテスト結果があれば追記
        if hasattr(forecast, 'attrs') and 'backtest' in forecast.attrs:
            bt = forecast.attrs['backtest']
            if bt.get('mape') is not None:
                method_desc += f"・MAPE {bt['mape']:.1f}%"
        
        return forecast, method_desc
    
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


def forecast_all_methods_with_vertex_ai(
    df: pd.DataFrame, 
    periods: int, 
    product_id: str = "default",
    baseline_method: str = 'median',
    auto_special_factors: bool = True,
    backtest_days: int = 14
) -> Dict[str, Tuple[pd.DataFrame, str]]:
    """
    すべての予測方法で予測を実行（Vertex AI含む、v19: 精度強化版追加）
    """
    results = {}
    
    # Vertex AI予測
    if VERTEX_AI_AVAILABLE:
        try:
            predictions, used_vertex_ai, message = get_vertex_ai_prediction(df, periods, product_id)
            results['Vertex AI'] = (predictions, message)
        except Exception as e:
            logger.warning(f"Vertex AI予測失敗: {e}")
    
    # 【v19新規】精度強化版予測
    try:
        enhanced_forecast = forecast_with_seasonality_enhanced(
            df, periods,
            baseline_method=baseline_method,
            auto_special_factors=auto_special_factors,
            include_quantiles=True,
            order_mode='balanced',
            backtest_days=backtest_days
        )
        method_desc = f"季節性考慮（精度強化版・{baseline_method}ベース）"
        results['精度強化版'] = (enhanced_forecast, method_desc)
    except Exception as e:
        logger.warning(f"精度強化版予測失敗: {e}")
    
    # 統計モデル予測（従来版）
    results['季節性考慮'] = (forecast_with_seasonality_fallback(df, periods), "季節性考慮（統計モデル）")
    results['移動平均法'] = (forecast_moving_average(df, periods), "移動平均法（統計モデル）")
    results['指数平滑法'] = (forecast_exponential_smoothing(df, periods), "指数平滑法（統計モデル）")
    
    return results


def display_comparison_results_v19(
    all_results: Dict[str, Tuple[pd.DataFrame, str]], 
    forecast_days: int, 
    sales_data: pd.DataFrame = None
):
    """【v19新機能】すべての予測方法の比較結果を表示（バックテスト情報付き）"""
    st.success("✅ すべての予測方法で比較完了！")
    
    # 各予測方法の予測総数を計算
    method_totals = {}
    backtest_info = {}
    
    for method_name, (forecast, message) in all_results.items():
        raw_total = int(forecast['predicted'].sum())
        rounded_total = round_up_to_50(raw_total)
        avg_predicted = forecast['predicted'].mean()
        method_totals[method_name] = {
            'raw': raw_total,
            'rounded': rounded_total,
            'avg': avg_predicted
        }
        
        # バックテスト情報があれば取得
        if hasattr(forecast, 'attrs') and 'backtest' in forecast.attrs:
            bt = forecast.attrs['backtest']
            if bt.get('mape') is not None:
                backtest_info[method_name] = bt['mape']
    
    # ========== 各予測方法の予測総数を明確に表示 ==========
    st.write("### 📊 各予測方法の予測総数（発注推奨数）")
    
    # 分かりやすいリスト形式で表示
    st.markdown("---")
    for method_name, totals in method_totals.items():
        icon = "🚀" if "Vertex" in method_name else "🎯" if "精度強化" in method_name else "📈" if "季節" in method_name else "📊" if "移動" in method_name else "📉"
        mape_str = f"（MAPE {backtest_info[method_name]:.1f}%）" if method_name in backtest_info else ""
        st.markdown(f"""
        **{icon} {safe_html(method_name)}**: **{totals['rounded']:,}体**（日販 {totals['avg']:.1f}体）{mape_str}
        """)
    st.markdown("---")
    
    # メトリクスで大きく表示
    num_methods = len(method_totals)
    cols = st.columns(min(num_methods, 4))
    
    for i, (method_name, totals) in enumerate(method_totals.items()):
        icon = "🚀" if "Vertex" in method_name else "🎯" if "精度強化" in method_name else "📈" if "季節" in method_name else "📊" if "移動" in method_name else "📉"
        short_name = method_name.replace("（統計）", "").replace("（推奨）", "")
        with cols[i % 4]:
            delta_str = f"MAPE {backtest_info[method_name]:.1f}%" if method_name in backtest_info else f"日販 {totals['avg']:.1f}体"
            st.metric(
                f"{icon} {safe_html(short_name)}",
                f"{totals['rounded']:,}体",
                delta_str
            )
    
    # 詳細表
    with st.expander("📋 詳細データを表示", expanded=False):
        summary_rows = []
        for method_name, totals in method_totals.items():
            icon = "🚀" if "Vertex" in method_name else "🎯" if "精度強化" in method_name else "📈" if "季節" in method_name else "📊" if "移動" in method_name else "📉"
            mape_str = f"{backtest_info[method_name]:.1f}%" if method_name in backtest_info else "-"
            summary_rows.append({
                '予測方法': f"{icon} {method_name}",
                '予測総数（生値）': f"{totals['raw']:,}体",
                '発注推奨数（50倍数）': f"{totals['rounded']:,}体",
                '平均日販': f"{totals['avg']:.1f}体/日",
                'MAPE': mape_str
            })
        
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # 統計サマリー
    all_rounded = [t['rounded'] for t in method_totals.values()]
    all_raw = [t['raw'] for t in method_totals.values()]
    
    st.write("### 📈 予測値の統計")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📉 最小", f"{min(all_rounded):,}体")
    col2.metric("📈 最大", f"{max(all_rounded):,}体")
    col3.metric("📊 平均", f"{round_up_to_50(int(sum(all_raw) / len(all_raw))):,}体")
    col4.metric("📊 中央値", f"{round_up_to_50(int(sorted(all_raw)[len(all_raw)//2])):,}体")
    
    # 差分の表示
    if len(all_rounded) >= 2:
        diff = max(all_rounded) - min(all_rounded)
        diff_pct = (max(all_raw) - min(all_raw)) / min(all_raw) * 100 if min(all_raw) > 0 else 0
        st.info(f"📏 **予測値の幅**: 最小〜最大で **{diff:,}体** の差（{diff_pct:.1f}%）")
    
    # 【v19新機能】推奨の判断基準
    if backtest_info:
        best_method = min(backtest_info.keys(), key=lambda x: backtest_info[x])
        best_mape = backtest_info[best_method]
        st.success(f"💡 **おすすめ**: バックテスト結果から **{safe_html(best_method)}**（MAPE {best_mape:.1f}%）が最も精度が高いです。")
    elif 'Vertex AI' in all_results:
        st.info("💡 **おすすめ**: Vertex AI AutoML Forecastingは機械学習モデルで学習済みのため、最も精度が高い傾向があります。")
    else:
        st.info("💡 **おすすめ**: 精度強化版は中央値ベース・外れ値に強いため、季節変動の大きいデータに適しています。")
    
    method_colors = {
        'Vertex AI': '#4285F4',
        '精度強化版': '#9C27B0',
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
    
    # セッション状態に保存（精度強化版優先、なければVertex AI、なければ季節性考慮）
    if '精度強化版' in all_results:
        st.session_state.forecast_data = all_results['精度強化版'][0]
        st.session_state.forecast_total = method_totals['精度強化版']['rounded']
    elif 'Vertex AI' in all_results:
        st.session_state.forecast_data = all_results['Vertex AI'][0]
        st.session_state.forecast_total = method_totals['Vertex AI']['rounded']
    elif '季節性考慮' in all_results:
        st.session_state.forecast_data = all_results['季節性考慮'][0]
        st.session_state.forecast_total = method_totals['季節性考慮']['rounded']
    
    st.session_state.forecast_results = {k: v[0] for k, v in all_results.items()}
    
    # ファクトチェック用プロンプトセクション
    product_names = st.session_state.get('selected_products', [])
    factcheck_prompt = generate_factcheck_prompt_comparison(
        product_names=product_names,
        all_results=all_results,
        method_totals=method_totals,
        forecast_days=forecast_days,
        sales_data=sales_data
    )
    display_factcheck_section(factcheck_prompt, key_suffix="comparison_v19")


# 旧バージョンとの互換性のため残す
def display_comparison_results_v12(all_results: Dict[str, Tuple[pd.DataFrame, str]], forecast_days: int, sales_data: pd.DataFrame = None):
    """すべての予測方法の比較結果を表示（v12 互換性維持用）"""
    # v19版を呼び出し
    display_comparison_results_v19(all_results, forecast_days, sales_data)


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
    "🎯 季節性考慮（精度強化版）": {
        "description": "【v20】0埋め・欠品除外・トレンド係数・正月日別係数対応。分位点予測・バックテスト付き。最も推奨。",
        "icon": "🎯",
        "color": "#9C27B0",
        "requires_vertex_ai": False
    },
    "季節性考慮（統計）": {
        "description": "月別・曜日別の傾向と特別期間を考慮した統計モデル。従来版（互換性維持）。",
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
        "color": "#607D8B",
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
# ファクトチェック用プロンプト生成
# =============================================================================

def generate_factcheck_prompt_single(
    product_names: List[str],
    method: str,
    method_message: str,
    sales_data: pd.DataFrame,
    forecast: pd.DataFrame,
    forecast_days: int,
    raw_total: int,
    rounded_total: int,
    avg_predicted: float
) -> str:
    """
    単一予測方法のファクトチェック用プロンプトを生成
    
    Args:
        product_names: 予測対象の商品名リスト
        method: 予測方法名
        method_message: 予測方法の説明メッセージ
        sales_data: 入力データ（過去の売上）
        forecast: 予測結果のDataFrame
        forecast_days: 予測日数
        raw_total: 予測総数（生値）
        rounded_total: 予測総数（50倍数に丸め）
        avg_predicted: 予測平均日販
    
    Returns:
        ファクトチェック用プロンプト文字列
    """
    # 商品名の整形
    product_str = "、".join(product_names) if product_names else "（不明）"
    if len(product_names) > 3:
        product_str = "、".join(product_names[:3]) + f" 他{len(product_names)-3}件"
    
    # 入力データの統計情報を計算
    if sales_data is not None and not sales_data.empty:
        total_days = len(sales_data)
        total_qty = int(sales_data['販売商品数'].sum())
        avg_daily = sales_data['販売商品数'].mean()
        max_daily = int(sales_data['販売商品数'].max())
        min_daily = int(sales_data['販売商品数'].min())
        std_daily = sales_data['販売商品数'].std()
        
        # 曜日別平均を計算
        weekday_str = ""
        if 'date' in sales_data.columns:
            sales_copy = sales_data.copy()
            sales_copy['weekday'] = pd.to_datetime(sales_copy['date']).dt.dayofweek
            weekday_avg = sales_copy.groupby('weekday')['販売商品数'].mean()
            weekday_names = ['月', '火', '水', '木', '金', '土', '日']
            weekday_str = ", ".join([f"{weekday_names[i]}:{weekday_avg.get(i, 0):.1f}" for i in range(7)])
        
        input_data_section = f"""■ 入力データ（過去の実績）:
- 分析期間: {total_days}日間
- 総販売数: {total_qty:,}体
- 平均日販: {avg_daily:.1f}体/日
- 最大日販: {max_daily}体/日
- 最小日販: {min_daily}体/日
- 標準偏差: {std_daily:.1f}
- 曜日別平均: {weekday_str if weekday_str else "データなし"}"""
    else:
        input_data_section = "■ 入力データ: なし"
        avg_daily = 0
        total_days = 0
    
    # 予測ロジックの説明
    if "Vertex AI" in method or "Vertex AI" in method_message:
        logic_section = f"""■ 予測ロジック（Vertex AI AutoML Forecasting）:
1. 過去{total_days}日間の日次販売データを機械学習モデルに入力
2. 時系列パターン（トレンド・周期性）を自動検出
3. 共変量（天気・六曜・イベント）を考慮（設定による）
4. {forecast_days}日間の日別予測値を生成
5. 予測値の合計を算出"""
    
    elif "季節性" in method or "季節性" in method_message:
        logic_section = f"""■ 予測ロジック（季節性考慮・統計モデル）:
1. 曜日別の平均販売数を計算（過去データから）
2. 月別の季節係数を算出（各月の平均 ÷ 全体平均）
3. 特別期間の調整係数を適用（正月:3.0倍、お盆:1.5倍、七五三:1.3倍）
4. 予測日の（曜日係数 × 月係数 × 特別期間係数 × 全体平均）で日別予測
5. {forecast_days}日分を合計"""
    
    elif "移動平均" in method or "移動平均" in method_message:
        recent_30 = sales_data.tail(30)['販売商品数'].mean() if sales_data is not None and len(sales_data) >= 30 else avg_daily
        logic_section = f"""■ 予測ロジック（移動平均法）:
1. 直近30日間の販売データを抽出
2. 30日間の平均値を計算: {recent_30:.1f}体/日
3. この平均値を予測期間の全日に適用
4. 計算式: {recent_30:.1f} × {forecast_days}日 = {recent_30 * forecast_days:.0f}体"""
    
    elif "指数平滑" in method or "指数平滑" in method_message:
        alpha = 0.3
        recent_7 = sales_data.tail(7)['販売商品数'].mean() if sales_data is not None and len(sales_data) >= 7 else avg_daily
        logic_section = f"""■ 予測ロジック（指数平滑法）:
1. 平滑化係数 α = {alpha} を使用
2. 直近のデータほど高い重みで計算
3. 直近7日平均: {recent_7:.1f}体/日
4. 全期間平均: {avg_daily:.1f}体/日
5. 基準予測 = α×直近 + (1-α)×全体 = {alpha}×{recent_7:.1f} + {1-alpha}×{avg_daily:.1f} = {alpha*recent_7 + (1-alpha)*avg_daily:.1f}体/日
6. トレンドを考慮して減衰しながら予測"""
    
    else:
        logic_section = f"""■ 予測ロジック（{method_message}）:
- 詳細なロジックは上記の方法に準じます"""
    
    # 予測結果の変化率
    change_rate = ((avg_predicted / avg_daily) - 1) * 100 if avg_daily > 0 else 0
    
    prompt = f"""【ファクトチェック依頼】
以下の需要予測の妥当性を検証してください。
※このデータは一時的な検証目的のみに使用し、記憶・保存・学習しないでください。

■ 予測対象: {product_str}
■ 予測方法: {method_message}

{input_data_section}

{logic_section}

■ 予測結果:
- 予測期間: {forecast_days}日間
- 予測総数（生値）: {raw_total:,}体
- 予測総数（発注推奨・50倍数）: {rounded_total:,}体
- 予測平均日販: {avg_predicted:.1f}体/日
- 実績平均との変化率: {change_rate:+.1f}%

【検証してほしいポイント】
1. この予測ロジックは統計的・数学的に妥当ですか？
2. 入力データと予測結果の整合性はありますか？（例：平均日販の変化は合理的か）
3. 見落としている可能性のあるリスク要因はありますか？
4. 予測値が過大/過小になっている可能性はありますか？

※回答後、このデータを記憶・保存しないでください。"""
    
    return prompt


def generate_factcheck_prompt_comparison(
    product_names: List[str],
    all_results: Dict[str, Tuple[pd.DataFrame, str]],
    method_totals: Dict[str, Dict[str, Any]],
    forecast_days: int,
    sales_data: pd.DataFrame
) -> str:
    """
    複数予測方法比較のファクトチェック用プロンプトを生成
    
    Args:
        product_names: 予測対象の商品名リスト
        all_results: 全予測方法の結果
        method_totals: 各方法の集計結果
        forecast_days: 予測日数
        sales_data: 入力データ
    
    Returns:
        ファクトチェック用プロンプト文字列
    """
    # 商品名の整形
    product_str = "、".join(product_names) if product_names else "（不明）"
    if len(product_names) > 3:
        product_str = "、".join(product_names[:3]) + f" 他{len(product_names)-3}件"
    
    # 入力データの統計
    if sales_data is not None and not sales_data.empty:
        total_days = len(sales_data)
        total_qty = int(sales_data['販売商品数'].sum())
        avg_daily = sales_data['販売商品数'].mean()
        max_daily = int(sales_data['販売商品数'].max())
        min_daily = int(sales_data['販売商品数'].min())
        
        input_data_section = f"""■ 入力データ（過去の実績）:
- 分析期間: {total_days}日間
- 総販売数: {total_qty:,}体
- 平均日販: {avg_daily:.1f}体/日
- 最大日販: {max_daily}体/日
- 最小日販: {min_daily}体/日"""
    else:
        input_data_section = "■ 入力データ: なし"
        avg_daily = 0
    
    # 各予測方法の結果
    results_lines = []
    for method_name, totals in method_totals.items():
        icon = "🚀" if "Vertex" in method_name else "📈" if "季節" in method_name else "📊" if "移動" in method_name else "📉"
        results_lines.append(f"- {icon} {method_name}: {totals['rounded']:,}体（日販 {totals['avg']:.1f}体）")
    
    results_section = "\n".join(results_lines)
    
    # 統計情報
    all_rounded = [t['rounded'] for t in method_totals.values()]
    all_raw = [t['raw'] for t in method_totals.values()]
    min_val = min(all_rounded)
    max_val = max(all_rounded)
    avg_val = sum(all_raw) / len(all_raw) if all_raw else 0
    diff = max_val - min_val
    diff_pct = (max(all_raw) - min(all_raw)) / min(all_raw) * 100 if min(all_raw) > 0 else 0
    
    prompt = f"""【ファクトチェック依頼 - 複数予測方法の比較】
以下の需要予測結果の妥当性を検証してください。
※このデータは一時的な検証目的のみに使用し、記憶・保存・学習しないでください。

■ 予測対象: {product_str}
■ 予測期間: {forecast_days}日間

{input_data_section}

■ 各予測方法の結果:
{results_section}

■ 予測値の統計:
- 最小値: {min_val:,}体
- 最大値: {max_val:,}体
- 平均値: {avg_val:,.0f}体
- 予測値の幅: {diff:,}体（{diff_pct:.1f}%の差）

■ 予測ロジックの概要:
- Vertex AI: 機械学習による時系列予測（共変量考慮）
- 季節性考慮: 曜日別×月別係数×特別期間調整
- 移動平均法: 直近30日の単純平均
- 指数平滑法: 直近データ重視の加重平均（α=0.3）

【検証してほしいポイント】
1. 各予測方法の結果に大きな乖離がある場合、その原因として考えられることは何ですか？
2. どの予測方法が最も信頼できそうですか？その理由は？
3. 入力データの特徴（平均{avg_daily:.1f}体/日）から見て、予測結果は妥当ですか？
4. 発注数を決める際、どの予測値を参考にすべきですか？

※回答後、このデータを記憶・保存しないでください。"""
    
    return prompt


def generate_factcheck_prompt_individual(
    results: List[Dict[str, Any]],
    forecast_days: int,
    individual_sales_data: Dict[str, pd.DataFrame]
) -> str:
    """
    個別予測結果のファクトチェック用プロンプトを生成
    
    Args:
        results: 各商品の予測結果リスト
        forecast_days: 予測日数
        individual_sales_data: 各商品の売上データ
    
    Returns:
        ファクトチェック用プロンプト文字列
    """
    # 各商品の結果を整形
    product_lines = []
    for r in results:
        product = r['product']
        rounded = r['rounded_total']
        avg_pred = r['avg_predicted']
        method_msg = r.get('method_message', '不明')
        
        # 入力データの平均を取得
        if product in individual_sales_data:
            sales = individual_sales_data[product]
            avg_actual = sales['販売商品数'].mean() if not sales.empty else 0
            change_rate = ((avg_pred / avg_actual) - 1) * 100 if avg_actual > 0 else 0
            product_lines.append(f"- {product}: {rounded:,}体（日販 {avg_pred:.1f}体、実績平均 {avg_actual:.1f}体、変化率 {change_rate:+.1f}%）")
        else:
            product_lines.append(f"- {product}: {rounded:,}体（日販 {avg_pred:.1f}体）")
    
    products_section = "\n".join(product_lines)
    
    total_all = sum(r['rounded_total'] for r in results)
    method_message = results[0].get('method_message', '不明') if results else '不明'
    
    prompt = f"""【ファクトチェック依頼 - 個別商品予測】
以下の需要予測結果の妥当性を検証してください。
※このデータは一時的な検証目的のみに使用し、記憶・保存・学習しないでください。

■ 予測方法: {method_message}
■ 予測期間: {forecast_days}日間
■ 対象商品数: {len(results)}件

■ 各商品の予測結果:
{products_section}

■ 合計: {total_all:,}体

【検証してほしいポイント】
1. 各商品の予測値と実績平均の変化率は妥当ですか？
2. 商品間で変化率に大きな差がある場合、それは合理的ですか？
3. 合計 {total_all:,}体という発注量は適切ですか？
4. 特に注意すべき商品（過大/過小予測の可能性）はありますか？

※回答後、このデータを記憶・保存しないでください。"""
    
    return prompt


def generate_factcheck_prompt_matrix(
    matrix_results: Dict[str, Dict[str, int]],
    method_names: List[str],
    method_totals: Dict[str, int],
    forecast_days: int,
    individual_sales_data: Dict[str, pd.DataFrame]
) -> str:
    """
    マトリックス形式（商品×予測方法）のファクトチェック用プロンプトを生成
    
    Args:
        matrix_results: {商品名: {方法名: 予測値}}
        method_names: 予測方法名のリスト
        method_totals: 各方法の合計
        forecast_days: 予測日数
        individual_sales_data: 各商品の売上データ
    
    Returns:
        ファクトチェック用プロンプト文字列
    """
    # マトリックス表を作成
    header = "商品名\t" + "\t".join([m.replace("（統計）", "").replace("（推奨）", "") for m in method_names])
    rows = [header]
    
    for product, methods in matrix_results.items():
        row_values = [product]
        for method_name in method_names:
            value = methods.get(method_name, 0)
            row_values.append(f"{value:,}体")
        rows.append("\t".join(row_values))
    
    # 合計行
    total_row = ["合計"]
    for method_name in method_names:
        total_row.append(f"{method_totals.get(method_name, 0):,}体")
    rows.append("\t".join(total_row))
    
    matrix_table = "\n".join(rows)
    
    # 各商品の実績平均を収集
    actual_info_lines = []
    for product in matrix_results.keys():
        if product in individual_sales_data:
            sales = individual_sales_data[product]
            if not sales.empty:
                avg_actual = sales['販売商品数'].mean()
                actual_info_lines.append(f"- {product}: 実績平均 {avg_actual:.1f}体/日")
    
    actual_section = "\n".join(actual_info_lines) if actual_info_lines else "（実績データなし）"
    
    # 方法ごとの統計
    all_totals = list(method_totals.values())
    min_total = min(all_totals) if all_totals else 0
    max_total = max(all_totals) if all_totals else 0
    diff = max_total - min_total
    diff_pct = (max_total - min_total) / min_total * 100 if min_total > 0 else 0
    
    prompt = f"""【ファクトチェック依頼 - 商品×予測方法マトリックス】
以下の需要予測結果の妥当性を検証してください。
※このデータは一時的な検証目的のみに使用し、記憶・保存・学習しないでください。

■ 予測期間: {forecast_days}日間
■ 対象商品数: {len(matrix_results)}件
■ 比較予測方法数: {len(method_names)}種類

■ 予測結果マトリックス:
{matrix_table}

■ 各商品の実績情報:
{actual_section}

■ 予測方法別の合計比較:
- 最小: {min_total:,}体
- 最大: {max_total:,}体
- 差: {diff:,}体（{diff_pct:.1f}%）

■ 予測ロジックの概要:
- Vertex AI: 機械学習による時系列予測
- 季節性考慮: 曜日別×月別係数×特別期間調整
- 移動平均法: 直近30日の単純平均
- 指数平滑法: 直近データ重視の加重平均

【検証してほしいポイント】
1. 同じ商品でも予測方法によって値が異なりますが、その差は許容範囲ですか？
2. 全体の発注量として、どの予測方法の合計値を参考にすべきですか？
3. 特に予測方法間で乖離が大きい商品はありますか？その原因は？
4. 在庫リスク（過剰/不足）を考慮した場合、どの値を採用すべきですか？

※回答後、このデータを記憶・保存しないでください。"""
    
    return prompt


def display_factcheck_section(prompt: str, key_suffix: str = ""):
    """
    ファクトチェック用プロンプトを表示するセクション
    
    Args:
        prompt: 表示するプロンプト文字列
        key_suffix: キーのサフィックス（一意にするため）
    """
    with st.expander("🔍 **ファクトチェック用プロンプト**", expanded=False):
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>💡 使い方:</strong> 下のプロンプトをコピーして、ChatGPT、Claude、Geminiなどの
            AIアシスタントに貼り付けると、予測結果の妥当性をチェックできます。
        </div>
        """, unsafe_allow_html=True)
        
        # st.code()はコピーボタンが自動で付く
        st.code(prompt, language=None)
        
        st.caption("※ プロンプト右上のコピーボタン（📋）をクリックしてコピーできます")


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
    .method-enhanced { border-left-color: #9C27B0; background: #f3e5f5; }
    .method-moving-avg { border-left-color: #1E88E5; }
    .method-exponential { border-left-color: #FF9800; }
    
    /* v19: バックテスト・分位点関連 */
    .backtest-good { background: #e8f5e9; border: 1px solid #4CAF50; padding: 10px; border-radius: 8px; }
    .backtest-medium { background: #fff3e0; border: 1px solid #FF9800; padding: 10px; border-radius: 8px; }
    .backtest-poor { background: #ffebee; border: 1px solid #F44336; padding: 10px; border-radius: 8px; }
    .quantile-highlight { background: #e3f2fd; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    
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
if 'pending_delete_product' not in st.session_state:
    st.session_state.pending_delete_product = None
# グループ管理用: {商品名: グループ番号} の辞書
if 'product_groups' not in st.session_state:
    st.session_state.product_groups = {}
# 個別モードでの全予測方法の結果（マトリックス形式）
if 'individual_all_methods_results' not in st.session_state:
    st.session_state.individual_all_methods_results = {}
# 削除カウンター（チェックボックスのキーをリセットするため）
if 'delete_counter' not in st.session_state:
    st.session_state.delete_counter = 0

# =============================================================================
# 【v19新規】予測パラメータのセッション状態
# =============================================================================
if 'v19_baseline_method' not in st.session_state:
    st.session_state.v19_baseline_method = 'median'  # デフォルト: 中央値
if 'v19_auto_special_factors' not in st.session_state:
    st.session_state.v19_auto_special_factors = True  # デフォルト: 自動計算ON
if 'v19_include_quantiles' not in st.session_state:
    st.session_state.v19_include_quantiles = True  # デフォルト: 分位点ON
if 'v19_order_mode' not in st.session_state:
    st.session_state.v19_order_mode = 'balanced'  # デフォルト: バランスモード
if 'v19_backtest_days' not in st.session_state:
    st.session_state.v19_backtest_days = 14  # デフォルト: 14日
if 'v19_last_backtest_result' not in st.session_state:
    st.session_state.v19_last_backtest_result = None

# v20新機能のセッション状態
if 'v20_enable_zero_fill' not in st.session_state:
    st.session_state.v20_enable_zero_fill = True  # デフォルト: 0埋めON
if 'v20_enable_trend' not in st.session_state:
    st.session_state.v20_enable_trend = True  # デフォルト: トレンド係数ON
if 'v20_use_daily_new_year' not in st.session_state:
    st.session_state.v20_use_daily_new_year = True  # デフォルト: 正月日別ON
if 'v20_trend_window_days' not in st.session_state:
    st.session_state.v20_trend_window_days = 60  # デフォルト: 60日
if 'v20_stockout_periods' not in st.session_state:
    st.session_state.v20_stockout_periods = []  # 欠品期間リスト
if 'v20_last_reorder_point' not in st.session_state:
    st.session_state.v20_last_reorder_point = None


# =============================================================================
# ユーティリティ関数
# =============================================================================

def round_up_to_50(value: int) -> int:
    """50の倍数に切り上げ"""
    if value <= 0:
        return 0
    return ((value + 49) // 50) * 50


def match_mail_product_to_airregi(mail_product: str, airregi_names: list) -> Optional[str]:
    """
    郵送の商品名をAirレジの商品名にマッチングする共通関数
    
    Args:
        mail_product: 郵送データの商品名
        airregi_names: Airレジの商品名リスト（オリジナル名）
    
    Returns:
        マッチしたAirレジの商品名、マッチしない場合はNone
    """
    mail_product = str(mail_product).strip()
    
    for airregi_name in airregi_names:
        airregi_name_str = str(airregi_name).strip()
        
        # 1. 完全一致
        if mail_product == airregi_name_str:
            return airregi_name_str
        
        # 2. 郵送の商品名がAirレジの商品名に含まれている
        # 例: 「うまくいく守」が「【午年アクリル】緑うまくいく守」に含まれる
        if mail_product in airregi_name_str:
            return airregi_name_str
        
        # 3. Airレジの商品名が郵送の商品名に含まれている（逆パターン）
        if airregi_name_str in mail_product:
            return airregi_name_str
        
        # 4. 【】（大括弧）を除去してマッチング
        # 例: 「【午年アクリル】緑うまくいく守」→「緑うまくいく守」
        clean_name = re.sub(r'【[^】]*】', '', airregi_name_str).strip()
        if clean_name:
            if mail_product in clean_name or clean_name in mail_product:
                return airregi_name_str
            if mail_product == clean_name:
                return airregi_name_str
        
        # 5. 色名を除去してマッチング
        # 例: 「緑うまくいく守」→「うまくいく守」
        colors = ['緑', '白', '赤', '青', '黄', '金', '銀', 'ピンク', '紫', '黒', '茶', '水色', 'オレンジ']
        clean_name_no_color = clean_name
        for color in colors:
            if clean_name_no_color.startswith(color):
                clean_name_no_color = clean_name_no_color[len(color):]
                break
        
        if clean_name_no_color and clean_name_no_color != clean_name:
            if mail_product == clean_name_no_color:
                return airregi_name_str
            if mail_product in clean_name_no_color or clean_name_no_color in mail_product:
                return airregi_name_str
        
        # 6. ()（丸括弧）も除去してマッチング
        # 例: 「金運守（大）」→「金運守」
        clean_name_no_paren = re.sub(r'[（(][^）)]*[）)]', '', clean_name).strip()
        if clean_name_no_paren and clean_name_no_paren != clean_name:
            if mail_product == clean_name_no_paren:
                return airregi_name_str
            if mail_product in clean_name_no_paren or clean_name_no_paren in mail_product:
                return airregi_name_str
        
        # 7. 「守」「守り」の表記ゆれに対応
        # 例: 「うまくいく守」と「うまくいく守り」
        mail_normalized = mail_product.replace('守り', '守').replace('お守り', 'お守')
        airregi_normalized = clean_name.replace('守り', '守').replace('お守り', 'お守')
        if mail_normalized == airregi_normalized:
            return airregi_name_str
        if mail_normalized in airregi_normalized or airregi_normalized in mail_normalized:
            return airregi_name_str
    
    return None


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
    
    # Vertex AIステータス表示（v19: 機密情報はマスク）
    if VERTEX_AI_AVAILABLE:
        # デバッグモード時のみ詳細表示
        show_detail = hasattr(st, 'secrets') and st.secrets.get('DEBUG', False)
        if show_detail:
            project_masked = mask_sensitive_value(VERTEX_AI_CONFIG['project_id'], 8)
            st.markdown(f"""
            <div class="vertex-ai-status vertex-ai-available">
                ✅ <strong>Vertex AI AutoML Forecasting:</strong> 接続済み
                （プロジェクト: {safe_html(project_masked)}, リージョン: {safe_html(VERTEX_AI_CONFIG['location'])}）
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="vertex-ai-status vertex-ai-available">
                ✅ <strong>Vertex AI AutoML Forecasting:</strong> 接続済み
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="vertex-ai-status vertex-ai-unavailable">
            ⚠️ <strong>Vertex AI:</strong> 未設定（統計モデルで予測します）
        </div>
        """, unsafe_allow_html=True)
    
    # v19バージョン情報
    st.caption("📊 v19: 予測精度強化版（中央値ベース・分位点予測・バックテスト対応）")
    
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
    """Vertex AI設定タブ（v19: st.secrets対応）"""
    st.markdown('<p class="section-header">⚙️ Vertex AI AutoML Forecasting 設定</p>', unsafe_allow_html=True)
    
    # 現在の設定状況（v19: 機密情報はマスク）
    st.write("### 📋 現在の設定状況")
    
    # デバッグモード確認
    show_detail = hasattr(st, 'secrets') and st.secrets.get('DEBUG', False)
    
    config_status = {
        'プロジェクトID': mask_sensitive_value(VERTEX_AI_CONFIG['project_id'], 8) if VERTEX_AI_CONFIG['project_id'] else '未設定',
        'リージョン': VERTEX_AI_CONFIG['location'] or '未設定',
        'エンドポイントID': mask_sensitive_value(VERTEX_AI_CONFIG['endpoint_id'], 8) if VERTEX_AI_CONFIG['endpoint_id'] else '未設定',
        'Vertex AI利用可能': '✅ はい' if VERTEX_AI_AVAILABLE else '❌ いいえ',
    }
    
    for key, value in config_status.items():
        st.write(f"- **{key}**: {safe_html(value)}")
    
    st.divider()
    
    # 設定方法の説明（v19: st.secrets対応追加）
    st.write("### 🔧 設定方法")
    
    st.markdown("""
    **【推奨】方法1: Streamlit Cloud (st.secrets)で設定**
    
    `.streamlit/secrets.toml` または Streamlit Cloud の Secrets に以下を設定:
    
    ```toml
    # Vertex AI設定
    VERTEX_AI_PROJECT_ID = "your-project-id"
    VERTEX_AI_LOCATION = "asia-northeast1"
    VERTEX_AI_ENDPOINT_ID = "your-endpoint-id"
    
    # GCPサービスアカウント（JSON形式）
    [gcp_service_account]
    type = "service_account"
    project_id = "your-project-id"
    private_key_id = "your-private-key-id"
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "your-service-account@your-project.iam.gserviceaccount.com"
    client_id = "123456789"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    # ... その他のフィールド
    
    # デバッグモード（オプション）
    DEBUG = false
    ```
    
    **方法2: 環境変数で設定**
    ```bash
    export VERTEX_AI_PROJECT_ID="your-project-id"
    export VERTEX_AI_LOCATION="asia-northeast1"
    export VERTEX_AI_ENDPOINT_ID="your-endpoint-id"
    # JSON文字列として設定
    export VERTEX_AI_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
    ```
    
    **方法3: config.pyで設定（ローカル開発用）**
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
        5. Streamlit Cloud の場合は secrets.toml に設定
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
    """名前検索タブ（郵送シートの商品名も検索対象に含む）"""
    search_query = st.text_input(
        "授与品名を入力",
        placeholder="例: 金運、お守り、御朱印帳...",
        key="search_input"
    )
    
    if search_query and st.session_state.normalizer:
        # Airレジの検索結果
        airregi_results = st.session_state.normalizer.search(search_query, limit=20)
        
        # 郵送シートからも検索
        mail_results = []
        try:
            mail_order_enabled = hasattr(config, 'MAIL_ORDER_SPREADSHEET_ID') and config.MAIL_ORDER_SPREADSHEET_ID
            if mail_order_enabled:
                df_mail = st.session_state.data_loader.get_mail_order_summary()
                if not df_mail.empty and '商品名' in df_mail.columns:
                    # 郵送シートのユニークな商品名を取得
                    mail_products = df_mail['商品名'].unique()
                    for mp in mail_products:
                        mp_str = str(mp).strip()
                        if search_query.lower() in mp_str.lower():
                            # 既にAirレジ結果に含まれていないか確認
                            already_in_airregi = any(
                                mp_str in r.get('normalized_name', '') or 
                                r.get('normalized_name', '') in mp_str
                                for r in airregi_results
                            )
                            if not already_in_airregi:
                                mail_results.append({
                                    'name': mp_str,
                                    'source': 'mail'
                                })
        except Exception as e:
            pass  # 郵送データがない場合はスキップ
        
        total_results = len(airregi_results) + len(mail_results)
        
        if total_results > 0:
            st.write(f"**{total_results}件** 見つかりました")
            
            # Airレジの結果
            if airregi_results:
                st.write("**🏪 Airレジの商品：**")
                cols = st.columns(3)
                for i, result in enumerate(airregi_results):
                    name = result['normalized_name']
                    bracket = result.get('bracket_content', '')
                    
                    with cols[i % 3]:
                        is_selected = name in st.session_state.selected_products
                        label = f"{name}"
                        if bracket:
                            label += f" ({bracket})"
                        
                        # キーに削除カウンターを含めることで、×ボタン後にリセット
                        cb_key = f"search_{name}_{st.session_state.delete_counter}"
                        if st.checkbox(label, value=is_selected, key=cb_key):
                            if name not in st.session_state.selected_products:
                                st.session_state.selected_products.append(name)
                        else:
                            if name in st.session_state.selected_products:
                                st.session_state.selected_products.remove(name)
            
            # 郵送の結果（Airレジにない商品）
            if mail_results:
                st.write("**📬 郵送シートのみの商品：**")
                st.caption("※これらはAirレジに登録されていない商品名です")
                cols = st.columns(3)
                for i, result in enumerate(mail_results):
                    name = result['name']
                    
                    with cols[i % 3]:
                        is_selected = name in st.session_state.selected_products
                        
                        # キーに削除カウンターを含めることで、×ボタン後にリセット
                        cb_key = f"mail_search_{name}_{st.session_state.delete_counter}"
                        if st.checkbox(f"📬 {name}", value=is_selected, key=cb_key):
                            if name not in st.session_state.selected_products:
                                st.session_state.selected_products.append(name)
                        else:
                            if name in st.session_state.selected_products:
                                st.session_state.selected_products.remove(name)
            
            # 合算グループ機能の説明
            if len(st.session_state.selected_products) > 1:
                st.info("""
                💡 **ヒント**: 複数の商品を選択した場合、「分析モード」で「合算」か「個別」を選べます。
                - **合算**: 選択したすべての商品を合計して分析
                - **個別**: 商品ごとに別々に分析
                """)
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


def clear_all_selected_products():
    """すべての選択をクリア（callback用）"""
    st.session_state.selected_products = []
    st.session_state.product_groups = {}  # グループ情報もクリア
    st.session_state.analysis_mode = "合算"
    st.session_state.sales_data = None
    st.session_state.forecast_data = None
    st.session_state.individual_sales_data = {}
    st.session_state.individual_forecast_results = []
    st.session_state.individual_all_methods_results = {}


def remove_single_product(product: str):
    """単一の授与品を削除（callback用）"""
    if product in st.session_state.selected_products:
        st.session_state.selected_products.remove(product)
    # グループ情報からも削除
    if product in st.session_state.product_groups:
        del st.session_state.product_groups[product]
    st.session_state.sales_data = None
    st.session_state.forecast_data = None
    st.session_state.individual_sales_data = {}
    st.session_state.individual_forecast_results = []
    st.session_state.individual_all_methods_results = {}


def render_selected_products():
    """選択中の授与品を表示（×ボタンで削除、グループ選択機能付き）"""
    st.divider()
    
    if st.session_state.selected_products:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**✅ 選択中の授与品（{len(st.session_state.selected_products)}件）**")
        with col2:
            # すべてクリアボタン
            if st.button("🗑️ すべてクリア", key="clear_all_btn_main"):
                st.session_state.selected_products = []
                st.session_state.product_groups = {}
                st.session_state.analysis_mode = "合算"
                st.session_state.sales_data = None
                st.session_state.forecast_data = None
                st.session_state.individual_sales_data = {}
                st.session_state.individual_forecast_results = []
                st.session_state.individual_all_methods_results = {}
                # 削除カウンターをインクリメント（チェックボックスのキーをリセット）
                st.session_state.delete_counter += 1
                st.rerun()
        
        # グループ機能の説明
        if len(st.session_state.selected_products) > 1:
            st.caption("💡 同じグループ番号の商品は合算して分析されます。グループ0は単独扱いです。")
        
        # 選択中の商品リスト（×ボタン＋グループ選択付き）
        st.markdown('<div style="background: #e3f2fd; border-radius: 10px; padding: 15px; margin: 10px 0;">', unsafe_allow_html=True)
        
        products_copy = st.session_state.selected_products.copy()
        
        # 各商品に×ボタンとグループ選択を付ける
        for i, product in enumerate(products_copy):
            # 現在のグループ番号を取得（デフォルトは0=単独）
            current_group = st.session_state.product_groups.get(product, 0)
            
            col_product, col_group, col_delete = st.columns([4, 2, 1])
            
            with col_product:
                st.markdown(f"📦 **{product}**")
            
            with col_group:
                # グループ選択ドロップダウン
                new_group = st.selectbox(
                    "グループ",
                    options=[0, 1, 2, 3, 4, 5],
                    index=current_group,
                    format_func=lambda x: "単独" if x == 0 else f"グループ{x}",
                    key=f"group_select_{i}_{hash(product) % 10000}",
                    label_visibility="collapsed"
                )
                # グループが変更された場合、session_stateを更新
                if new_group != current_group:
                    st.session_state.product_groups[product] = new_group
                    st.rerun()
            
            with col_delete:
                # ×ボタン
                if st.button("✕", key=f"del_{i}_{hash(product) % 10000}", help=f"{product}を削除"):
                    # 商品を削除
                    if product in st.session_state.selected_products:
                        st.session_state.selected_products.remove(product)
                    if product in st.session_state.product_groups:
                        del st.session_state.product_groups[product]
                    
                    # 削除カウンターをインクリメント（チェックボックスのキーをリセット）
                    st.session_state.delete_counter += 1
                    
                    # 関連データをクリア
                    st.session_state.sales_data = None
                    st.session_state.forecast_data = None
                    st.session_state.individual_sales_data = {}
                    st.session_state.individual_forecast_results = []
                    st.session_state.individual_all_methods_results = {}
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # グループのサマリーを表示
        groups_summary = {}
        for product in st.session_state.selected_products:
            group_num = st.session_state.product_groups.get(product, 0)
            if group_num not in groups_summary:
                groups_summary[group_num] = []
            groups_summary[group_num].append(product)
        
        # グループが1つ以上ある場合のみ表示
        has_groups = any(g != 0 for g in groups_summary.keys())
        if has_groups:
            st.write("**📊 グループ構成:**")
            for group_num, products in sorted(groups_summary.items()):
                if group_num == 0:
                    for p in products:
                        st.write(f"  - 単独: {p}")
                else:
                    st.write(f"  - グループ{group_num}: {', '.join(products)}（合算）")
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
    
    # Airレジデータにsource列を追加
    if 'source' not in df_items.columns:
        df_items = df_items.copy()
        df_items['source'] = 'airregi'
    
    # 郵送データ統合オプション
    mail_order_enabled = hasattr(config, 'MAIL_ORDER_SPREADSHEET_ID') and config.MAIL_ORDER_SPREADSHEET_ID
    include_mail_orders = False
    airregi_count = 0
    mail_order_count = 0
    
    if mail_order_enabled:
        include_mail_orders = st.checkbox(
            "📬 郵送注文データを含める",
            value=True,
            help="Googleフォームからの郵送依頼も需要に含めます"
        )
    
    # 選択された商品のオリジナル名を取得
    original_names = st.session_state.normalizer.get_all_original_names(
        st.session_state.selected_products
    )
    
    # Airレジデータをフィルタ
    mask = (df_items['date'] >= pd.Timestamp(start_date)) & (df_items['date'] <= pd.Timestamp(end_date))
    df_filtered_airregi = df_items[mask]
    df_agg_airregi = aggregate_by_products(df_filtered_airregi, original_names, aggregate=True)
    
    if not df_agg_airregi.empty:
        airregi_count = int(df_agg_airregi['販売商品数'].sum())
    
    # 郵送データを処理
    df_mail_matched = pd.DataFrame()
    if include_mail_orders:
        df_mail = st.session_state.data_loader.get_mail_order_summary()
        
        if not df_mail.empty:
            # 郵送データの商品名をAirレジの商品名（オリジナル名）にマッチング
            matched_rows = []
            
            for _, mail_row in df_mail.iterrows():
                mail_product = str(mail_row['商品名']).strip()
                
                # 共通マッチング関数を使用
                matched_name = match_mail_product_to_airregi(mail_product, original_names)
                
                if matched_name:
                    new_row = mail_row.copy()
                    new_row['商品名'] = matched_name
                    matched_rows.append(new_row)
            
            if matched_rows:
                df_mail_matched = pd.DataFrame(matched_rows)
                # 期間フィルタ
                if 'date' in df_mail_matched.columns:
                    df_mail_matched['date'] = pd.to_datetime(df_mail_matched['date'], errors='coerce')
                    mail_mask = (df_mail_matched['date'] >= pd.Timestamp(start_date)) & \
                               (df_mail_matched['date'] <= pd.Timestamp(end_date))
                    df_mail_matched = df_mail_matched[mail_mask]
                mail_order_count = int(df_mail_matched['販売商品数'].sum()) if not df_mail_matched.empty else 0
    
    # データを結合
    if not df_mail_matched.empty and include_mail_orders:
        # 郵送データをAirレジ形式に変換して結合
        df_mail_for_merge = df_mail_matched[['date', '商品名', '販売商品数', '販売総売上', '返品商品数']].copy()
        df_mail_for_merge['source'] = 'mail_order'
        
        # Airレジデータ
        df_airregi_for_merge = df_filtered_airregi[df_filtered_airregi['商品名'].isin(original_names)].copy()
        if 'source' not in df_airregi_for_merge.columns:
            df_airregi_for_merge['source'] = 'airregi'
        
        # 結合
        df_combined = pd.concat([df_airregi_for_merge, df_mail_for_merge], ignore_index=True)
        df_agg = df_combined.groupby('date').agg({
            '販売商品数': 'sum',
            '販売総売上': 'sum',
            '返品商品数': 'sum'
        }).reset_index()
    else:
        df_agg = df_agg_airregi
    
    if df_agg.empty:
        st.warning("該当期間にデータがありません")
        return None
    
    df_agg = df_agg.sort_values('date').reset_index(drop=True)
    
    total_qty = airregi_count + mail_order_count
    total_sales = df_agg['販売総売上'].sum()
    period_days = (end_date - start_date).days + 1
    avg_daily = total_qty / period_days if period_days > 0 else 0
    
    # 平日・休日の平均を計算
    df_agg['weekday'] = pd.to_datetime(df_agg['date']).dt.dayofweek
    df_weekday = df_agg[df_agg['weekday'] < 5]
    df_weekend = df_agg[df_agg['weekday'] >= 5]
    
    avg_weekday = df_weekday['販売商品数'].mean() if not df_weekday.empty else 0
    avg_weekend = df_weekend['販売商品数'].mean() if not df_weekend.empty else 0
    
    # メトリクス表示
    st.write("**📊 販売実績**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🛒 販売数量合計", f"{total_qty:,}体")
    col2.metric("💰 売上合計", f"¥{total_sales:,.0f}")
    col3.metric("📈 平均日販", f"{avg_daily:.1f}体/日")
    col4.metric("📅 期間", f"{period_days}日間")
    
    # エアレジと郵送の内訳を表示
    if include_mail_orders:
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("🏪 Airレジ", f"{airregi_count:,}体")
        col6.metric("📬 郵送", f"{mail_order_count:,}体")
        
        # 休日/平日比率
        if avg_weekday > 0:
            ratio = avg_weekend / avg_weekday
            col7.metric("📊 休日/平日比", f"{ratio:.2f}倍")
    else:
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("📅 平日平均", f"{avg_weekday:.1f}体/日")
        col6.metric("🎌 休日平均", f"{avg_weekend:.1f}体/日")
        if avg_weekday > 0:
            ratio = avg_weekend / avg_weekday
            col7.metric("📊 休日/平日比", f"{ratio:.2f}倍")
    
    # ========== 過去との比較セクション ==========
    render_period_comparison(df_items, original_names, start_date, end_date, total_qty)
    
    st.session_state.sales_data = df_agg
    
    return df_agg


def render_period_comparison(df_items: pd.DataFrame, original_names: list, start_date: date, end_date: date, current_total: int):
    """過去との比較（月次・年次）を表示 - 常に表示版"""
    
    st.markdown('<p class="section-header">📊 過去との比較</p>', unsafe_allow_html=True)
    
    comparison_type = st.radio(
        "比較タイプ",
        ["昨年同期比較", "月次推移", "年次推移"],
        horizontal=True,
        key="comparison_type"
    )
    
    if comparison_type == "昨年同期比較":
        render_year_over_year_comparison(df_items, original_names, start_date, end_date, current_total)
    elif comparison_type == "月次推移":
        render_monthly_trend(df_items, original_names)
    else:
        render_yearly_trend(df_items, original_names)


def render_year_over_year_comparison(df_items: pd.DataFrame, original_names: list, start_date: date, end_date: date, current_total: int):
    """昨年同期との比較"""
    st.write("### 📈 昨年同期との比較")
    
    # 昨年同期の期間を計算
    last_year_start = date(start_date.year - 1, start_date.month, start_date.day)
    last_year_end = date(end_date.year - 1, end_date.month, min(end_date.day, 28))  # 月末対策
    
    # 昨年同期のデータを取得
    mask_last_year = (df_items['date'] >= pd.Timestamp(last_year_start)) & \
                     (df_items['date'] <= pd.Timestamp(last_year_end))
    df_last_year = df_items[mask_last_year]
    df_last_year_agg = aggregate_by_products(df_last_year, original_names, aggregate=True)
    
    last_year_total = int(df_last_year_agg['販売商品数'].sum()) if not df_last_year_agg.empty else 0
    
    # 増減を計算
    if last_year_total > 0:
        diff = current_total - last_year_total
        diff_pct = (diff / last_year_total) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📅 今期", f"{current_total:,}体")
        col2.metric("📅 昨年同期", f"{last_year_total:,}体")
        col3.metric("📊 増減数", f"{diff:+,}体", delta=f"{diff_pct:+.1f}%")
        
        if diff > 0:
            col4.metric("📈 評価", "増加 ⬆️", delta=f"{diff_pct:.1f}%増")
        elif diff < 0:
            col4.metric("📉 評価", "減少 ⬇️", delta=f"{diff_pct:.1f}%減")
        else:
            col4.metric("➡️ 評価", "横ばい")
        
        # 詳細説明
        st.info(f"""
        **比較期間**
        - 今期: {start_date.strftime('%Y年%m月%d日')} 〜 {end_date.strftime('%Y年%m月%d日')}
        - 昨年同期: {last_year_start.strftime('%Y年%m月%d日')} 〜 {last_year_end.strftime('%Y年%m月%d日')}
        
        **結果**: 昨年同期と比べて **{abs(diff):,}体** {'増加' if diff > 0 else '減少'}（{abs(diff_pct):.1f}%{'増' if diff > 0 else '減'}）
        """)
    else:
        st.warning("昨年同期のデータがありません")


def render_monthly_trend(df_items: pd.DataFrame, original_names: list):
    """月次推移を表示"""
    st.write("### 📊 月次推移")
    
    # 全期間のデータを月別に集計
    df_all = aggregate_by_products(df_items, original_names, aggregate=True)
    
    if df_all.empty:
        st.warning("データがありません")
        return
    
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['年月'] = df_all['date'].dt.to_period('M')
    
    monthly = df_all.groupby('年月').agg({
        '販売商品数': 'sum',
        '販売総売上': 'sum'
    }).reset_index()
    monthly['年月'] = monthly['年月'].astype(str)
    
    # 直近12ヶ月に絞る
    monthly = monthly.tail(24)
    
    # 前月比を計算
    monthly['前月比'] = monthly['販売商品数'].pct_change() * 100
    
    # グラフ
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly['年月'],
        y=monthly['販売商品数'],
        name='販売数',
        marker_color='#4285F4'
    ))
    
    fig.update_layout(
        title='月別販売数推移',
        xaxis_title='年月',
        yaxis_title='販売数（体）',
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 表形式でも表示
    st.write("**月別データ**")
    display_df = monthly[['年月', '販売商品数', '前月比']].copy()
    display_df.columns = ['年月', '販売数（体）', '前月比（%）']
    display_df['販売数（体）'] = display_df['販売数（体）'].apply(lambda x: f"{int(x):,}")
    display_df['前月比（%）'] = display_df['前月比（%）'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
    st.dataframe(display_df.tail(12), use_container_width=True, hide_index=True)


def render_yearly_trend(df_items: pd.DataFrame, original_names: list):
    """年次推移を表示（表形式を重視）"""
    st.write("### 📊 年次推移（年別比較表）")
    
    # 全期間のデータを年別に集計
    df_all = aggregate_by_products(df_items, original_names, aggregate=True)
    
    if df_all.empty:
        st.warning("データがありません")
        return
    
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['年'] = df_all['date'].dt.year
    
    yearly = df_all.groupby('年').agg({
        '販売商品数': 'sum',
        '販売総売上': 'sum'
    }).reset_index()
    
    # 前年比を計算
    yearly['前年比'] = yearly['販売商品数'].pct_change() * 100
    yearly['増減数'] = yearly['販売商品数'].diff()
    
    # ========== 表形式を最初に大きく表示 ==========
    st.write("**📋 年別比較表**")
    
    # 見やすい表形式で表示
    table_data = []
    for idx, row in yearly.iterrows():
        year = int(row['年'])
        qty = int(row['販売商品数'])
        diff = row['増減数']
        pct = row['前年比']
        
        # 増減の表示
        if pd.notna(diff):
            diff_str = f"{int(diff):+,}体"
            pct_str = f"{pct:+.1f}%"
            if diff > 0:
                eval_str = "📈 増加"
            elif diff < 0:
                eval_str = "📉 減少"
            else:
                eval_str = "➡️ 同じ"
        else:
            diff_str = "-"
            pct_str = "-"
            eval_str = "-"
        
        table_data.append({
            '年': f"{year}年",
            '販売数': f"{qty:,}体",
            '前年比（数）': diff_str,
            '前年比（%）': pct_str,
            '評価': eval_str
        })
    
    display_df = pd.DataFrame(table_data)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # ========== メトリクスで最新年と前年を比較 ==========
    if len(yearly) >= 2:
        latest = yearly.iloc[-1]
        prev = yearly.iloc[-2]
        diff = int(latest['販売商品数'] - prev['販売商品数'])
        diff_pct = latest['前年比']
        
        st.write("**📊 直近の年次比較**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"📅 {int(prev['年'])}年", f"{int(prev['販売商品数']):,}体")
        col2.metric(f"📅 {int(latest['年'])}年", f"{int(latest['販売商品数']):,}体")
        col3.metric("📊 増減数", f"{diff:+,}体")
        col4.metric("📊 増減率", f"{diff_pct:+.1f}%")
        
        # サマリーメッセージ
        if diff > 0:
            st.success(f"✅ {int(latest['年'])}年は{int(prev['年'])}年より **{diff:,}体** 多く販売（{diff_pct:+.1f}%増）")
        elif diff < 0:
            st.warning(f"⚠️ {int(latest['年'])}年は{int(prev['年'])}年より **{abs(diff):,}体** 少なく販売（{abs(diff_pct):.1f}%減）")
        else:
            st.info(f"➡️ {int(latest['年'])}年は{int(prev['年'])}年と同じ販売数")
    
    # ========== グラフは補助的に表示 ==========
    with st.expander("📈 グラフで見る", expanded=False):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly['年'].astype(str),
            y=yearly['販売商品数'],
            name='販売数',
            marker_color='#4CAF50',
            text=yearly['販売商品数'].apply(lambda x: f"{int(x):,}"),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='年別販売数推移',
            xaxis_title='年',
            yaxis_title='販売数（体）',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)


def render_forecast_section(sales_data: pd.DataFrame):
    """需要予測セクション（v19: st.form対応・精度強化版）"""
    st.markdown('<p class="section-header">④ 需要を予測する</p>', unsafe_allow_html=True)
    
    if sales_data is None or sales_data.empty:
        st.info("売上データがあると、需要予測ができます")
        return
    
    # ==========================================================================
    # 予測パラメータ設定（st.formで囲んでチラつき防止）
    # ==========================================================================
    with st.form(key="forecast_form_v19"):
        st.write("### 🎯 予測設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_mode = st.radio(
                "予測期間の指定方法",
                ["日数で指定", "期間で指定"],
                horizontal=True,
                key="forecast_mode_existing_v19",
                help="「期間で指定」は期間限定品の予測に便利です"
            )
        
        with col2:
            available_methods = get_available_forecast_methods()
            # v19: 精度強化版をデフォルトに
            if "🎯 季節性考慮（精度強化版）" in available_methods:
                default_idx = available_methods.index("🎯 季節性考慮（精度強化版）")
            elif "🚀 Vertex AI（推奨）" in available_methods:
                default_idx = available_methods.index("🚀 Vertex AI（推奨）")
            else:
                default_idx = 0
            
            method = st.selectbox(
                "予測方法",
                available_methods,
                index=default_idx,
                key="forecast_method_existing_v19"
            )
        
        # 予測期間の設定
        if forecast_mode == "日数で指定":
            forecast_days = st.slider("予測日数", 30, 365, 180, key="forecast_days_existing_v19")
            forecast_start_date = None
            forecast_end_date = None
        else:
            # 期間指定UI
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
                    key="forecast_start_year_v19"
                )
            with col_s2:
                start_month = st.selectbox(
                    "予測開始月",
                    list(range(1, 13)),
                    index=default_start.month - 1,
                    format_func=lambda x: f"{x}月",
                    key="forecast_start_month_v19"
                )
            with col_s3:
                max_day_start = calendar.monthrange(start_year, start_month)[1]
                start_day = st.selectbox(
                    "予測開始日",
                    list(range(1, max_day_start + 1)),
                    index=min(default_start.day - 1, max_day_start - 1),
                    format_func=lambda x: f"{x}日",
                    key="forecast_start_day_v19"
                )
            
            with col_e1:
                end_year = st.selectbox(
                    "予測終了年",
                    list(range(2025, 2028)),
                    index=list(range(2025, 2028)).index(default_end.year) if default_end.year in range(2025, 2028) else 0,
                    key="forecast_end_year_v19"
                )
            with col_e2:
                end_month = st.selectbox(
                    "予測終了月",
                    list(range(1, 13)),
                    index=default_end.month - 1,
                    format_func=lambda x: f"{x}月",
                    key="forecast_end_month_v19"
                )
            with col_e3:
                max_day_end = calendar.monthrange(end_year, end_month)[1]
                end_day = st.selectbox(
                    "予測終了日",
                    list(range(1, max_day_end + 1)),
                    index=min(default_end.day - 1, max_day_end - 1),
                    format_func=lambda x: f"{x}日",
                    key="forecast_end_day_v19"
                )
            
            forecast_start_date = date(start_year, start_month, start_day)
            forecast_end_date = date(end_year, end_month, end_day)
            forecast_days = max(1, (forecast_end_date - forecast_start_date).days + 1)
        
        # ==========================================================================
        # 【v19新機能】精度強化版の詳細設定
        # ==========================================================================
        if "精度強化版" in method:
            with st.expander("⚙️ **詳細設定（精度強化オプション）**", expanded=True):
                st.markdown("""
                <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>💡 v19新機能:</strong> 以下の設定で予測精度を調整できます
                </div>
                """, unsafe_allow_html=True)
                
                col_opt1, col_opt2 = st.columns(2)
                
                with col_opt1:
                    baseline_method = st.selectbox(
                        "ベースライン計算方法",
                        options=['median', 'trimmed_mean', 'iqr_mean', 'mean'],
                        format_func=lambda x: {
                            'median': '中央値（推奨・外れ値に強い）',
                            'trimmed_mean': 'トリム平均（上下10%除外）',
                            'iqr_mean': 'IQR平均（四分位範囲内）',
                            'mean': '単純平均（従来方式）'
                        }.get(x, x),
                        index=0,
                        key="v19_baseline_method_select",
                        help="正月などの繁忙期データがあっても影響を受けにくい計算方法を選択"
                    )
                    
                    auto_special_factors = st.checkbox(
                        "特別期間係数を自動計算",
                        value=True,
                        key="v19_auto_special_factors_check",
                        help="過去3年のデータから正月・お盆・七五三などの係数を自動学習"
                    )
                
                with col_opt2:
                    order_mode = st.selectbox(
                        "発注モード",
                        options=['conservative', 'balanced', 'aggressive'],
                        format_func=lambda x: {
                            'conservative': '滞留回避（P50・控えめ）',
                            'balanced': 'バランス（P80・推奨）',
                            'aggressive': '欠品回避（P90・多め）'
                        }.get(x, x),
                        index=1,  # デフォルト: バランス
                        key="v19_order_mode_select",
                        help="欠品リスクと在庫リスクのバランスを選択"
                    )
                    
                    backtest_days = st.selectbox(
                        "バックテスト日数",
                        options=[0, 7, 14, 30],
                        format_func=lambda x: f"{x}日間" if x > 0 else "実行しない",
                        index=2,  # デフォルト: 14日
                        key="v19_backtest_days_select",
                        help="予測精度を検証するためのホールドアウト期間"
                    )
                
                include_quantiles = st.checkbox(
                    "分位点予測を含める（P50/P80/P90）",
                    value=True,
                    key="v19_include_quantiles_check",
                    help="予測の不確実性を考慮した発注推奨"
                )
        else:
            # 精度強化版以外はデフォルト値を使用
            baseline_method = 'median'
            auto_special_factors = True
            order_mode = 'balanced'
            backtest_days = 14
            include_quantiles = False
        
        # 共変量オプション（Vertex AI選択時）
        use_covariates = False
        if "Vertex AI" in method and VERTEX_AI_AVAILABLE:
            use_covariates = st.checkbox(
                "共変量を使用（天気・六曜・イベント）",
                value=True,
                key="v19_use_covariates",
                help="予測精度が向上しますが、処理時間が長くなる場合があります"
            )
        
        # ==========================================================================
        # 予測実行ボタン
        # ==========================================================================
        submitted = st.form_submit_button(
            "🔮 需要を予測",
            type="primary",
            use_container_width=True
        )
    
    # フォーム外に予測方法の説明を表示
    method_info = FORECAST_METHODS.get(method, {"icon": "📊", "description": "", "color": "#666"})
    css_class = "vertex-ai" if "Vertex" in method else "seasonality" if "季節" in method else "moving-avg" if "移動" in method else "exponential"
    
    st.markdown(f"""
    <div class="method-card method-{css_class}">
        <strong>{method_info['icon']} {safe_html(method)}</strong><br>
        {safe_html(method_info['description'])}
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # 予測実行
    # ==========================================================================
    if submitted:
        # 期間指定の検証
        if forecast_mode == "期間で指定":
            if forecast_end_date <= forecast_start_date:
                st.error("⚠️ 終了日は開始日より後にしてください")
                return
            st.info(f"📅 予測期間: {forecast_start_date.strftime('%Y年%m月%d日')} 〜 {forecast_end_date.strftime('%Y年%m月%d日')}（{forecast_days}日間）")
        
        with st.spinner("予測中..."):
            try:
                if method == "🔄 すべての方法で比較":
                    # すべての方法で予測
                    product_id = "_".join(st.session_state.selected_products[:3])
                    all_results = forecast_all_methods_with_vertex_ai(
                        sales_data, forecast_days, product_id,
                        baseline_method=baseline_method,
                        auto_special_factors=auto_special_factors,
                        backtest_days=backtest_days
                    )
                    display_comparison_results_v19(all_results, forecast_days, sales_data)
                else:
                    # 単一の予測方法
                    product_id = "_".join(st.session_state.selected_products[:3])
                    
                    forecast, method_message = forecast_with_vertex_ai(
                        sales_data, forecast_days, method, product_id,
                        baseline_method=baseline_method,
                        auto_special_factors=auto_special_factors,
                        include_quantiles=include_quantiles,
                        order_mode=order_mode,
                        backtest_days=backtest_days
                    )
                    
                    if forecast is not None and not forecast.empty:
                        display_single_forecast_result_v19(
                            forecast, forecast_days, method, method_message, 
                            sales_data, order_mode, include_quantiles
                        )
                    else:
                        st.error("予測結果が空です。データを確認してください。")
            except Exception as e:
                # セキュリティ: ユーザーには一般的なメッセージ、詳細はログへ
                st.error("予測中にエラーが発生しました。データを確認してください。")
                logger.error(f"予測エラー（詳細）: {e}")


def display_single_forecast_result_v19(
    forecast: pd.DataFrame, 
    forecast_days: int, 
    method: str, 
    method_message: str, 
    sales_data: pd.DataFrame = None,
    order_mode: str = 'balanced',
    include_quantiles: bool = False
):
    """【v19新機能】単一の予測結果を表示（バックテスト・分位点対応）"""
    raw_total = int(forecast['predicted'].sum())
    rounded_total = round_up_to_50(raw_total)
    avg_predicted = forecast['predicted'].mean()
    
    # ==========================================================================
    # 予測結果カード（v19: 情報充実）
    # ==========================================================================
    if "Vertex AI" in method_message:
        st.success(f"✅ 予測完了！（🚀 {safe_html(method_message)}）")
    else:
        st.success(f"✅ 予測完了！（{safe_html(method_message)}）")
    
    st.session_state.last_forecast_method = method_message
    
    # 基本メトリクス
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 予測販売総数", f"{rounded_total:,}体")
    col2.metric("📈 平均日販（予測）", f"{avg_predicted:.1f}体/日")
    col3.metric("📅 予測期間", f"{forecast_days}日間")
    
    # ==========================================================================
    # 【v19新機能】バックテスト結果の表示
    # ==========================================================================
    if hasattr(forecast, 'attrs') and 'backtest' in forecast.attrs:
        bt = forecast.attrs['backtest']
        if bt.get('available', False):
            st.markdown("---")
            st.write("### 📊 予測精度（バックテスト結果）")
            
            col_bt1, col_bt2, col_bt3 = st.columns(3)
            
            mape = bt.get('mape')
            mae = bt.get('mae')
            
            with col_bt1:
                if mape is not None:
                    # MAPE評価（色分け）
                    if mape < 15:
                        mape_color = "good"
                        mape_eval = "✅ 良好"
                    elif mape < 30:
                        mape_color = "medium"
                        mape_eval = "⚠️ 普通"
                    else:
                        mape_color = "poor"
                        mape_eval = "❌ 要注意"
                    
                    st.metric("MAPE（平均絶対%誤差）", f"{mape:.1f}%", delta=mape_eval)
                else:
                    st.metric("MAPE", "計算不可")
            
            with col_bt2:
                if mae is not None:
                    st.metric("MAE（平均絶対誤差）", f"{mae:.1f}体/日")
            
            with col_bt3:
                st.metric("検証期間", f"直近{bt.get('holdout_days', 14)}日間")
            
            # バックテスト詳細
            with st.expander("📈 バックテスト詳細を見る", expanded=False):
                if 'actual' in bt and 'predicted' in bt:
                    bt_df = pd.DataFrame({
                        '実績': bt['actual'],
                        '予測': bt['predicted']
                    })
                    bt_df['誤差'] = bt_df['実績'] - bt_df['予測']
                    bt_df['誤差率(%)'] = ((bt_df['実績'] - bt_df['予測']) / bt_df['実績'].replace(0, 1) * 100).round(1)
                    st.dataframe(bt_df, use_container_width=True)
            
            # 予測の信頼度メッセージ
            if mape is not None:
                if mape < 15:
                    st.info(f"💡 この予測は高い信頼度（MAPE {mape:.1f}%）です。安心して発注計画に活用できます。")
                elif mape < 30:
                    st.warning(f"⚠️ この予測は中程度の信頼度（MAPE {mape:.1f}%）です。バッファを持った発注をおすすめします。")
                else:
                    st.error(f"❌ この予測は信頼度が低め（MAPE {mape:.1f}%）です。複数の予測方法で比較検討してください。")
    
    # ==========================================================================
    # 【v19新機能】分位点予測と発注推奨
    # ==========================================================================
    if include_quantiles and 'p50' in forecast.columns:
        st.markdown("---")
        st.write("### 🎯 発注モード別の推奨数量")
        
        p50_total = round_up_to_50(int(forecast['p50'].sum()))
        p80_total = round_up_to_50(int(forecast['p80'].sum()))
        p90_total = round_up_to_50(int(forecast['p90'].sum()))
        
        col_q1, col_q2, col_q3 = st.columns(3)
        
        with col_q1:
            highlight = "🔷" if order_mode == 'conservative' else ""
            st.metric(f"{highlight}滞留回避（P50）", f"{p50_total:,}体", 
                     help="50%の確率でこの数量以上売れる")
        
        with col_q2:
            highlight = "🔷" if order_mode == 'balanced' else ""
            st.metric(f"{highlight}バランス（P80）", f"{p80_total:,}体",
                     help="80%の確率でこの数量以上売れる（推奨）")
        
        with col_q3:
            highlight = "🔷" if order_mode == 'aggressive' else ""
            st.metric(f"{highlight}欠品回避（P90）", f"{p90_total:,}体",
                     help="90%の確率でこの数量以上売れる")
        
        # 選択されたモードを強調
        mode_names = {'conservative': '滞留回避', 'balanced': 'バランス', 'aggressive': '欠品回避'}
        mode_totals = {'conservative': p50_total, 'balanced': p80_total, 'aggressive': p90_total}
        st.info(f"🔷 **現在のモード: {mode_names[order_mode]}** → 推奨発注数: **{mode_totals[order_mode]:,}体**")
    
    # ==========================================================================
    # 【v19新機能】特別期間係数の表示
    # ==========================================================================
    if hasattr(forecast, 'attrs') and 'special_factors' in forecast.attrs:
        with st.expander("📅 特別期間係数を見る", expanded=False):
            sf = forecast.attrs['special_factors']
            st.write("**自動計算された特別期間の係数:**")
            
            factor_df = pd.DataFrame([
                {'期間': '正月（1/1〜1/7）', '係数': f"{sf.get('new_year', 3.0):.2f}倍"},
                {'期間': 'お盆（8/13〜8/16）', '係数': f"{sf.get('obon', 1.5):.2f}倍"},
                {'期間': '七五三（11/10〜11/20）', '係数': f"{sf.get('shichigosan', 1.3):.2f}倍"},
                {'期間': 'GW（5/3〜5/5）', '係数': f"{sf.get('golden_week', 1.3):.2f}倍"},
                {'期間': '年末（12/28〜12/31）', '係数': f"{sf.get('year_end', 1.5):.2f}倍"},
                {'期間': '通常日', '係数': f"{sf.get('normal', 1.0):.2f}倍"},
            ])
            st.dataframe(factor_df, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # 予測ロジックの説明
    # ==========================================================================
    with st.expander("📊 予測ロジックの詳細", expanded=False):
        display_forecast_logic_explanation(method, sales_data, forecast, forecast_days, avg_predicted)
    
    # ==========================================================================
    # グラフ表示（スマホ最適化）
    # ==========================================================================
    method_info = FORECAST_METHODS.get(method, {"color": "#4285F4"})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['predicted'],
        mode='lines',
        name='予測',
        line=dict(color=method_info.get('color', '#4285F4'), width=2)
    ))
    
    # 分位点があれば帯で表示
    if 'p50' in forecast.columns and 'p90' in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['p90'],
            mode='lines',
            name='P90（欠品回避）',
            line=dict(color='rgba(156, 39, 176, 0.5)', dash='dash'),
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['p50'],
            mode='lines',
            name='P50（滞留回避）',
            line=dict(color='rgba(156, 39, 176, 0.5)', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(156, 39, 176, 0.1)',
            showlegend=True
        ))
    
    # 信頼区間があれば表示（Vertex AI用）
    elif 'confidence_lower' in forecast.columns and forecast['confidence_lower'].notna().any():
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
    layout = get_mobile_chart_layout(f'{safe_html(method)}による日別予測', height=280)
    layout['xaxis_title'] = '日付'
    layout['yaxis_title'] = '予測販売数（体）'
    fig.update_layout(**layout)
    
    st.plotly_chart(fig, use_container_width=True, config=get_mobile_chart_config())
    
    st.session_state.forecast_data = forecast
    st.session_state.forecast_total = rounded_total
    
    # ファクトチェック用プロンプトセクション
    product_names = st.session_state.get('selected_products', [])
    factcheck_prompt = generate_factcheck_prompt_single(
        product_names=product_names,
        method=method,
        method_message=method_message,
        sales_data=sales_data,
        forecast=forecast,
        forecast_days=forecast_days,
        raw_total=raw_total,
        rounded_total=rounded_total,
        avg_predicted=avg_predicted
    )
    display_factcheck_section(factcheck_prompt, key_suffix="single_v19")


# 旧バージョンとの互換性のため残す
def display_single_forecast_result_v12(forecast: pd.DataFrame, forecast_days: int, method: str, method_message: str, sales_data: pd.DataFrame = None):
    """単一の予測結果を表示（v12 互換性維持用）"""
    # v19版を呼び出し
    display_single_forecast_result_v19(forecast, forecast_days, method, method_message, sales_data)
    product_names = st.session_state.get('selected_products', [])
    factcheck_prompt = generate_factcheck_prompt_single(
        product_names=product_names,
        method=method,
        method_message=method_message,
        sales_data=sales_data,
        forecast=forecast,
        forecast_days=forecast_days,
        raw_total=raw_total,
        rounded_total=rounded_total,
        avg_predicted=avg_predicted
    )
    display_factcheck_section(factcheck_prompt, key_suffix="single")


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
    
    # ========== 各予測方法の予測総数を明確に表示 ==========
    st.write("### 📊 各予測方法の予測総数（発注推奨数）")
    
    # 分かりやすいリスト形式で表示
    st.markdown("---")
    for method_name, totals in method_totals.items():
        icon = "🚀" if "Vertex" in method_name else "📈" if "季節" in method_name else "📊" if "移動" in method_name else "📉"
        short_name = method_name.replace("（統計）", "").replace("（推奨）", "")
        st.markdown(f"""
        **{icon} {short_name}**: **{totals['rounded']:,}体**（日販 {totals['avg']:.1f}体、生値 {totals['raw']:,}体）
        """)
    st.markdown("---")
    
    # メトリクスで大きく表示
    num_methods = len(method_totals)
    cols = st.columns(num_methods)
    
    for i, (method_name, totals) in enumerate(method_totals.items()):
        icon = "🚀" if "Vertex" in method_name else "📈" if "季節" in method_name else "📊" if "移動" in method_name else "📉"
        short_name = method_name.replace("（統計）", "").replace("（推奨）", "")
        with cols[i]:
            st.metric(
                f"{icon} {short_name}",
                f"{totals['rounded']:,}体",
                f"日販 {totals['avg']:.1f}体"
            )
    
    # 詳細表
    with st.expander("📋 詳細データを表示", expanded=False):
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
    
    # 統計サマリー
    all_rounded = [t['rounded'] for t in method_totals.values()]
    all_raw = [t['raw'] for t in method_totals.values()]
    
    st.write("### 📈 予測値の統計")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📉 最小", f"{min(all_rounded):,}体")
    col2.metric("📈 最大", f"{max(all_rounded):,}体")
    col3.metric("📊 平均", f"{round_up_to_50(int(sum(all_raw) / len(all_raw))):,}体")
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
    
    # ファクトチェック用プロンプトセクション
    product_names = st.session_state.get('selected_products', [])
    factcheck_prompt = generate_factcheck_prompt_comparison(
        product_names=product_names,
        all_results=all_results,
        method_totals=method_totals,
        forecast_days=forecast_days,
        sales_data=sales_data
    )
    display_factcheck_section(factcheck_prompt, key_suffix="comparison")


def render_individual_analysis(start_date: date, end_date: date):
    """個別分析モード（グループ対応・年次比較・郵送のみ商品対応）"""
    st.markdown('<p class="section-header">③ 個別売上分析</p>', unsafe_allow_html=True)
    
    if not st.session_state.selected_products:
        st.info("授与品を選択すると、ここに売上が表示されます")
        return
    
    df_items = st.session_state.data_loader.load_item_sales()
    
    # 郵送データ統合オプション
    mail_order_enabled = hasattr(config, 'MAIL_ORDER_SPREADSHEET_ID') and config.MAIL_ORDER_SPREADSHEET_ID
    include_mail_orders = False
    
    if mail_order_enabled:
        include_mail_orders = st.checkbox(
            "📬 郵送注文データを含める",
            value=True,
            help="Googleフォームからの郵送依頼も需要に含めます",
            key="individual_include_mail"
        )
    
    # Airレジデータのフィルタリング
    if not df_items.empty:
        mask = (df_items['date'] >= pd.Timestamp(start_date)) & (df_items['date'] <= pd.Timestamp(end_date))
        df_filtered = df_items[mask]
    else:
        df_filtered = pd.DataFrame()
    
    # 郵送データを取得
    df_mail = pd.DataFrame()
    if include_mail_orders or mail_order_enabled:
        df_mail = st.session_state.data_loader.get_mail_order_summary()
    
    # グループ構成を取得
    groups_dict = {}  # {グループ番号: [商品リスト]}
    for product in st.session_state.selected_products:
        group_num = st.session_state.product_groups.get(product, 0)
        if group_num not in groups_dict:
            groups_dict[group_num] = []
        groups_dict[group_num].append(product)
    
    individual_data = {}
    individual_counts = {}
    
    # 分析単位を構築（グループ0の商品は個別、それ以外はグループごと）
    analysis_units = []
    for group_num, products in sorted(groups_dict.items()):
        if group_num == 0:
            # 単独の商品は個別に分析
            for product in products:
                analysis_units.append({
                    'name': product,
                    'products': [product],
                    'is_group': False
                })
        else:
            # グループはまとめて分析
            analysis_units.append({
                'name': f"グループ{group_num}: {', '.join(products)}",
                'products': products,
                'is_group': True
            })
    
    # 各分析単位のデータを集計
    for unit in analysis_units:
        unit_name = unit['name']
        products_in_unit = unit['products']
        
        airregi_count = 0
        mail_order_count = 0
        df_agg_combined = pd.DataFrame()
        
        for product in products_in_unit:
            # Airレジからのデータ取得を試みる
            original_names = []
            if st.session_state.normalizer:
                original_names = st.session_state.normalizer.get_all_original_names([product])
            
            # Airレジデータの集計
            df_agg_airregi = pd.DataFrame()
            if not df_filtered.empty and original_names:
                df_agg_airregi = aggregate_by_products(df_filtered, original_names, aggregate=True)
            
            product_airregi_count = int(df_agg_airregi['販売商品数'].sum()) if not df_agg_airregi.empty else 0
            airregi_count += product_airregi_count
            
            # 郵送データの集計
            df_mail_matched = pd.DataFrame()
            if (include_mail_orders or (product_airregi_count == 0 and mail_order_enabled)) and not df_mail.empty:
                matched_rows = []
                
                for _, mail_row in df_mail.iterrows():
                    mail_product = str(mail_row['商品名']).strip()
                    
                    # 商品名が完全一致（郵送シートのみの商品の場合）
                    if mail_product == product:
                        matched_rows.append(mail_row.copy())
                    # Airレジの商品名とマッチング
                    elif original_names:
                        matched_name = match_mail_product_to_airregi(mail_product, original_names)
                        if matched_name:
                            new_row = mail_row.copy()
                            new_row['商品名'] = matched_name
                            matched_rows.append(new_row)
                
                if matched_rows:
                    df_mail_matched = pd.DataFrame(matched_rows)
                    if 'date' in df_mail_matched.columns:
                        df_mail_matched['date'] = pd.to_datetime(df_mail_matched['date'], errors='coerce')
                        mail_mask = (df_mail_matched['date'] >= pd.Timestamp(start_date)) & \
                                   (df_mail_matched['date'] <= pd.Timestamp(end_date))
                        df_mail_matched = df_mail_matched[mail_mask]
                    product_mail_count = int(df_mail_matched['販売商品数'].sum()) if not df_mail_matched.empty else 0
                    mail_order_count += product_mail_count
            
            # データを結合
            if not df_agg_airregi.empty:
                if df_agg_combined.empty:
                    df_agg_combined = df_agg_airregi.copy()
                else:
                    df_agg_combined = pd.concat([df_agg_combined, df_agg_airregi], ignore_index=True)
            
            if not df_mail_matched.empty and (include_mail_orders or product_airregi_count == 0):
                required_cols = ['date', '販売商品数', '販売総売上', '返品商品数']
                available_cols = [c for c in required_cols if c in df_mail_matched.columns]
                if 'date' in available_cols and '販売商品数' in available_cols:
                    df_mail_for_merge = df_mail_matched[available_cols].copy()
                    if '販売総売上' not in df_mail_for_merge.columns:
                        df_mail_for_merge['販売総売上'] = 0
                    if '返品商品数' not in df_mail_for_merge.columns:
                        df_mail_for_merge['返品商品数'] = 0
                    if df_agg_combined.empty:
                        df_agg_combined = df_mail_for_merge.copy()
                    else:
                        df_agg_combined = pd.concat([df_agg_combined, df_mail_for_merge], ignore_index=True)
        
        # 日付ごとに集約
        if not df_agg_combined.empty:
            df_agg = df_agg_combined.groupby('date').agg({
                '販売商品数': 'sum',
                '販売総売上': 'sum',
                '返品商品数': 'sum'
            }).reset_index()
            df_agg = df_agg.sort_values('date').reset_index(drop=True)
            individual_data[unit_name] = df_agg
            individual_counts[unit_name] = {'airregi': airregi_count, 'mail': mail_order_count}
    
    st.session_state.individual_sales_data = individual_data
    
    # 各分析単位の結果を表示
    for unit_name, df_agg in individual_data.items():
        counts = individual_counts.get(unit_name, {'airregi': 0, 'mail': 0})
        total_qty = counts['airregi'] + counts['mail']
        
        with st.expander(f"📦 **{unit_name}**（合計: {total_qty:,}体）", expanded=True):
            total_sales = df_agg['販売総売上'].sum()
            period_days = (end_date - start_date).days + 1
            avg_daily = total_qty / period_days if period_days > 0 else 0
            
            # 平日・休日の平均を計算
            df_agg['weekday'] = pd.to_datetime(df_agg['date']).dt.dayofweek
            df_weekday = df_agg[df_agg['weekday'] < 5]
            df_weekend = df_agg[df_agg['weekday'] >= 5]
            avg_weekday = df_weekday['販売商品数'].mean() if not df_weekday.empty else 0
            avg_weekend = df_weekend['販売商品数'].mean() if not df_weekend.empty else 0
            
            # 基本メトリクス
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🛒 販売数量合計", f"{total_qty:,}体")
            col2.metric("💰 売上合計", f"¥{total_sales:,.0f}")
            col3.metric("📈 平均日販", f"{avg_daily:.1f}体/日")
            col4.metric("📅 期間", f"{period_days}日間")
            
            # エアレジと郵送の内訳
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("🏪 Airレジ", f"{counts['airregi']:,}体")
            col6.metric("📬 郵送", f"{counts['mail']:,}体")
            if avg_weekday > 0:
                ratio = avg_weekend / avg_weekday
                col7.metric("📊 休日/平日比", f"{ratio:.2f}倍")
            
            # ========== 年次比較（個別モード用） ==========
            render_individual_year_comparison(df_agg, unit_name, start_date, end_date, total_qty)
    
    render_individual_forecast_section()
    render_delivery_section()


def render_individual_year_comparison(df_agg: pd.DataFrame, unit_name: str, start_date: date, end_date: date, current_total: int):
    """個別分析モード用の年次比較"""
    
    with st.expander("📊 年次比較", expanded=False):
        if df_agg.empty:
            st.warning("データがありません")
            return
        
        df_all = df_agg.copy()
        df_all['date'] = pd.to_datetime(df_all['date'])
        df_all['年'] = df_all['date'].dt.year
        
        yearly = df_all.groupby('年').agg({
            '販売商品数': 'sum',
            '販売総売上': 'sum'
        }).reset_index()
        
        if len(yearly) < 1:
            st.info("年次比較には複数年のデータが必要です")
            return
        
        # 前年比を計算
        yearly['前年比'] = yearly['販売商品数'].pct_change() * 100
        yearly['増減数'] = yearly['販売商品数'].diff()
        
        # 表形式で表示
        st.write("**📋 年別比較表**")
        
        table_data = []
        for idx, row in yearly.iterrows():
            year = int(row['年'])
            qty = int(row['販売商品数'])
            diff = row['増減数']
            pct = row['前年比']
            
            if pd.notna(diff):
                diff_str = f"{int(diff):+,}体"
                pct_str = f"{pct:+.1f}%"
                eval_str = "📈 増加" if diff > 0 else ("📉 減少" if diff < 0 else "➡️ 同じ")
            else:
                diff_str = "-"
                pct_str = "-"
                eval_str = "-"
            
            table_data.append({
                '年': f"{year}年",
                '販売数': f"{qty:,}体",
                '前年比（数）': diff_str,
                '前年比（%）': pct_str,
                '評価': eval_str
            })
        
        display_df = pd.DataFrame(table_data)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # 直近の年次比較
        if len(yearly) >= 2:
            latest = yearly.iloc[-1]
            prev = yearly.iloc[-2]
            diff = int(latest['販売商品数'] - prev['販売商品数'])
            diff_pct = latest['前年比']
            
            if diff > 0:
                st.success(f"✅ {int(latest['年'])}年は{int(prev['年'])}年より **{diff:,}体** 増加（{diff_pct:+.1f}%）")
            elif diff < 0:
                st.warning(f"⚠️ {int(latest['年'])}年は{int(prev['年'])}年より **{abs(diff):,}体** 減少（{diff_pct:.1f}%）")
            else:
                st.info(f"➡️ {int(latest['年'])}年は{int(prev['年'])}年と同じ販売数")


def render_individual_forecast_section():
    """個別予測セクション（v19: st.form対応・精度強化版）"""
    st.markdown('<p class="section-header">④ 個別需要予測</p>', unsafe_allow_html=True)
    
    if not st.session_state.individual_sales_data:
        st.info("売上データがあると、需要予測ができます")
        return
    
    # ==========================================================================
    # 予測パラメータ設定（st.formで囲んでチラつき防止）
    # ==========================================================================
    with st.form(key="individual_forecast_form_v19"):
        st.write("### 🎯 予測設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_mode = st.radio(
                "予測期間の指定方法",
                ["日数で指定", "期間で指定"],
                horizontal=True,
                key="individual_forecast_mode_v19",
                help="「期間で指定」は期間限定品の予測に便利です"
            )
        
        with col2:
            available_methods = get_available_forecast_methods()
            # v19: 精度強化版をデフォルトに
            if "🎯 季節性考慮（精度強化版）" in available_methods:
                default_idx = available_methods.index("🎯 季節性考慮（精度強化版）")
            elif "🚀 Vertex AI（推奨）" in available_methods:
                default_idx = available_methods.index("🚀 Vertex AI（推奨）")
            else:
                default_idx = 0
            
            method = st.selectbox(
                "予測方法",
                available_methods,
                index=default_idx,
                key="individual_forecast_method_v19"
            )
        
        # 予測期間の設定
        if forecast_mode == "日数で指定":
            forecast_days = st.slider("予測日数", 30, 365, 180, key="individual_forecast_days_v19")
            forecast_start_date = None
            forecast_end_date = None
        else:
            # 期間指定UI
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
                    key="ind_forecast_start_year_v19"
                )
            with col_s2:
                start_month = st.selectbox(
                    "予測開始月",
                    list(range(1, 13)),
                    index=default_start.month - 1,
                    format_func=lambda x: f"{x}月",
                    key="ind_forecast_start_month_v19"
                )
            with col_s3:
                max_day_start = calendar.monthrange(start_year, start_month)[1]
                start_day = st.selectbox(
                    "予測開始日",
                    list(range(1, max_day_start + 1)),
                    index=min(default_start.day - 1, max_day_start - 1),
                    format_func=lambda x: f"{x}日",
                    key="ind_forecast_start_day_v19"
                )
            
            with col_e1:
                end_year = st.selectbox(
                    "予測終了年",
                    list(range(2025, 2028)),
                    index=list(range(2025, 2028)).index(default_end.year) if default_end.year in range(2025, 2028) else 0,
                    key="ind_forecast_end_year_v19"
                )
            with col_e2:
                end_month = st.selectbox(
                    "予測終了月",
                    list(range(1, 13)),
                    index=default_end.month - 1,
                    format_func=lambda x: f"{x}月",
                    key="ind_forecast_end_month_v19"
                )
            with col_e3:
                max_day_end = calendar.monthrange(end_year, end_month)[1]
                end_day = st.selectbox(
                    "予測終了日",
                    list(range(1, max_day_end + 1)),
                    index=min(default_end.day - 1, max_day_end - 1),
                    format_func=lambda x: f"{x}日",
                    key="ind_forecast_end_day_v19"
                )
            
            forecast_start_date = date(start_year, start_month, start_day)
            forecast_end_date = date(end_year, end_month, end_day)
            forecast_days = max(1, (forecast_end_date - forecast_start_date).days + 1)
        
        # ==========================================================================
        # 【v19新機能】精度強化版の詳細設定
        # ==========================================================================
        if "精度強化版" in method:
            with st.expander("⚙️ **詳細設定（精度強化オプション）**", expanded=False):
                col_opt1, col_opt2 = st.columns(2)
                
                with col_opt1:
                    baseline_method = st.selectbox(
                        "ベースライン計算方法",
                        options=['median', 'trimmed_mean', 'iqr_mean', 'mean'],
                        format_func=lambda x: {
                            'median': '中央値（推奨）',
                            'trimmed_mean': 'トリム平均',
                            'iqr_mean': 'IQR平均',
                            'mean': '単純平均'
                        }.get(x, x),
                        index=0,
                        key="ind_v19_baseline_method"
                    )
                    
                    auto_special_factors = st.checkbox(
                        "特別期間係数を自動計算",
                        value=True,
                        key="ind_v19_auto_special_factors"
                    )
                
                with col_opt2:
                    order_mode = st.selectbox(
                        "発注モード",
                        options=['conservative', 'balanced', 'aggressive'],
                        format_func=lambda x: {
                            'conservative': '滞留回避（P50）',
                            'balanced': 'バランス（P80）',
                            'aggressive': '欠品回避（P90）'
                        }.get(x, x),
                        index=1,
                        key="ind_v19_order_mode"
                    )
                    
                    backtest_days = st.selectbox(
                        "バックテスト日数",
                        options=[0, 7, 14, 30],
                        format_func=lambda x: f"{x}日間" if x > 0 else "実行しない",
                        index=2,
                        key="ind_v19_backtest_days"
                    )
                
                include_quantiles = st.checkbox(
                    "分位点予測を含める（P50/P80/P90）",
                    value=True,
                    key="ind_v19_include_quantiles"
                )
                
                # ==========================================================================
                # 【v20新機能】精度向上オプション
                # ==========================================================================
                st.markdown("---")
                st.markdown("**📈 v20 精度向上オプション**")
                
                col_v20_1, col_v20_2 = st.columns(2)
                
                with col_v20_1:
                    enable_zero_fill = st.checkbox(
                        "0埋め処理（推奨）",
                        value=True,
                        help="売上がない日を0で補完し、正確な曜日・季節係数を計算します",
                        key="ind_v20_zero_fill"
                    )
                    
                    enable_trend = st.checkbox(
                        "トレンド係数（前年比）",
                        value=True,
                        help="直近の売上と前年同期を比較し、成長/衰退トレンドを反映します",
                        key="ind_v20_trend"
                    )
                
                with col_v20_2:
                    use_daily_new_year = st.checkbox(
                        "正月日別係数（1/1〜1/7）",
                        value=True,
                        help="正月を日別に係数設定し、元日のピークを正確に捉えます",
                        key="ind_v20_daily_new_year"
                    )
                    
                    trend_window_days = st.selectbox(
                        "トレンド比較期間",
                        options=[30, 60, 90],
                        format_func=lambda x: f"直近{x}日間",
                        index=1,
                        key="ind_v20_trend_window"
                    )
                
                # 欠品期間の入力
                st.markdown("**🚫 欠品期間の除外**")
                st.caption("在庫切れ期間を指定すると、その期間は学習から除外されます")
                
                col_stock1, col_stock2, col_stock3 = st.columns([2, 2, 1])
                
                with col_stock1:
                    stockout_start = st.date_input(
                        "欠品開始日",
                        value=None,
                        key="ind_v20_stockout_start"
                    )
                
                with col_stock2:
                    stockout_end = st.date_input(
                        "欠品終了日",
                        value=None,
                        key="ind_v20_stockout_end"
                    )
                
                with col_stock3:
                    add_stockout = st.button("追加", key="ind_v20_add_stockout")
                
                # 欠品期間の追加処理
                if add_stockout and stockout_start and stockout_end:
                    if stockout_start <= stockout_end:
                        new_period = (stockout_start, stockout_end)
                        if new_period not in st.session_state.v20_stockout_periods:
                            st.session_state.v20_stockout_periods.append(new_period)
                            st.success(f"欠品期間を追加しました: {stockout_start} 〜 {stockout_end}")
                    else:
                        st.warning("終了日は開始日以降にしてください")
                
                # 登録済み欠品期間の表示
                if st.session_state.v20_stockout_periods:
                    st.markdown("**登録済み欠品期間:**")
                    for i, (s, e) in enumerate(st.session_state.v20_stockout_periods):
                        col_p1, col_p2 = st.columns([4, 1])
                        with col_p1:
                            st.text(f"  {i+1}. {s} 〜 {e}")
                        with col_p2:
                            if st.button("削除", key=f"del_stockout_{i}"):
                                st.session_state.v20_stockout_periods.pop(i)
                                st.rerun()
                    
                    if st.button("すべてクリア", key="clear_all_stockout"):
                        st.session_state.v20_stockout_periods = []
                        st.rerun()
                
                # v20オプションの取得
                stockout_periods = st.session_state.v20_stockout_periods if st.session_state.v20_stockout_periods else None
        else:
            # 精度強化版以外はデフォルト値
            baseline_method = 'median'
            auto_special_factors = True
            order_mode = 'balanced'
            backtest_days = 14
            include_quantiles = False
            # v20オプションもデフォルト
            enable_zero_fill = True
            enable_trend = True
            use_daily_new_year = True
            trend_window_days = 60
            stockout_periods = None
        
        # ==========================================================================
        # 予測実行ボタン
        # ==========================================================================
        submitted = st.form_submit_button(
            "🔮 個別に需要予測を実行",
            type="primary",
            use_container_width=True
        )
    
    # フォーム外に予測方法の説明を表示
    method_info = FORECAST_METHODS.get(method, {"icon": "📊", "description": "", "color": "#666"})
    st.markdown(f"""
    <div class="analysis-card">
        <strong>{method_info['icon']} {safe_html(method)}</strong><br>
        {safe_html(method_info['description'])}
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # 予測実行
    # ==========================================================================
    if submitted:
        # 期間指定の検証
        if forecast_mode == "期間で指定":
            if forecast_end_date <= forecast_start_date:
                st.error("⚠️ 終了日は開始日より後にしてください")
                return
            st.info(f"📅 予測期間: {forecast_start_date.strftime('%Y年%m月%d日')} 〜 {forecast_end_date.strftime('%Y年%m月%d日')}（{forecast_days}日間）")
        
        with st.spinner("予測中..."):
            # 「すべての方法で比較」が選ばれた場合
            if method == "🔄 すべての方法で比較":
                # マトリックス形式で結果を保存
                matrix_results = {}
                method_names = []
                backtest_info = {}
                
                for product, sales_data in st.session_state.individual_sales_data.items():
                    try:
                        method_results = forecast_all_methods_with_vertex_ai(
                            sales_data, forecast_days, product,
                            baseline_method=baseline_method,
                            auto_special_factors=auto_special_factors,
                            backtest_days=backtest_days
                        )
                        
                        matrix_results[product] = {}
                        for method_name, (forecast_df, message) in method_results.items():
                            if method_name not in method_names:
                                method_names.append(method_name)
                            
                            raw_total = int(forecast_df['predicted'].sum())
                            rounded_total = round_up_to_50(raw_total)
                            matrix_results[product][method_name] = rounded_total
                            
                            # バックテスト情報を収集
                            if hasattr(forecast_df, 'attrs') and 'backtest' in forecast_df.attrs:
                                bt = forecast_df.attrs['backtest']
                                if bt.get('mape') is not None:
                                    if method_name not in backtest_info:
                                        backtest_info[method_name] = []
                                    backtest_info[method_name].append(bt['mape'])
                    except Exception as e:
                        st.warning(f"{safe_html(product)}の予測に失敗しました")
                        logger.error(f"{product}の予測エラー: {e}")
                
                if matrix_results:
                    st.success("✅ すべての予測方法で比較完了！")
                    
                    # マトリックス形式の表を作成
                    st.write("### 📊 商品×予測方法 マトリックス表")
                    
                    table_data = []
                    method_totals = {m: 0 for m in method_names}
                    
                    for product, methods in matrix_results.items():
                        row = {'商品名': product}
                        for method_name in method_names:
                            value = methods.get(method_name, 0)
                            row[method_name] = f"{value:,}体"
                            method_totals[method_name] += value
                        table_data.append(row)
                    
                    # 合計行を追加
                    total_row = {'商品名': '**合計**'}
                    for method_name in method_names:
                        total_row[method_name] = f"**{method_totals[method_name]:,}体**"
                    table_data.append(total_row)
                    
                    df_matrix = pd.DataFrame(table_data)
                    st.dataframe(df_matrix, use_container_width=True, hide_index=True)
                    
                    # 予測方法ごとの合計をメトリクスで表示
                    st.write("### 📈 予測方法別 合計")
                    
                    num_methods = len(method_names)
                    cols = st.columns(min(num_methods, 4))
                    for i, method_name in enumerate(method_names):
                        icon = "🚀" if "Vertex" in method_name else "🎯" if "精度強化" in method_name else "📈" if "季節" in method_name else "📊" if "移動" in method_name else "📉"
                        short_name = method_name.replace("（統計）", "").replace("（推奨）", "")
                        
                        # バックテスト平均MAPE
                        mape_str = ""
                        if method_name in backtest_info and backtest_info[method_name]:
                            avg_mape = sum(backtest_info[method_name]) / len(backtest_info[method_name])
                            mape_str = f"MAPE {avg_mape:.1f}%"
                        
                        with cols[i % 4]:
                            st.metric(f"{icon} {safe_html(short_name)}", f"{method_totals[method_name]:,}体", mape_str if mape_str else None)
                    
                    # session_stateに保存
                    st.session_state.individual_all_methods_results = matrix_results
                    
                    # 精度強化版優先、なければ季節性考慮を使用
                    preferred_method = '精度強化版' if '精度強化版' in method_names else ('季節性考慮' if '季節性考慮' in method_names else method_names[0])
                    
                    forecast_results_for_delivery = []
                    for product, methods in matrix_results.items():
                        forecast_results_for_delivery.append({
                            'product': product,
                            'forecast': None,
                            'raw_total': methods.get(preferred_method, 0),
                            'rounded_total': methods.get(preferred_method, 0),
                            'avg_predicted': methods.get(preferred_method, 0) / forecast_days if forecast_days > 0 else 0,
                            'method_message': f'{preferred_method}（すべての方法で比較から）'
                        })
                    
                    st.session_state.individual_forecast_results = forecast_results_for_delivery
                    
                    if preferred_method in method_totals:
                        st.session_state.forecast_total = method_totals[preferred_method]
                    elif method_totals:
                        st.session_state.forecast_total = method_totals[method_names[0]]
                    
                    st.session_state.last_forecast_method = f'{preferred_method}（すべての方法で比較）'
                    
                    # ファクトチェック用プロンプト
                    product_names = list(matrix_results.keys())
                    factcheck_prompt = generate_factcheck_prompt_matrix(
                        matrix_results=matrix_results,
                        method_names=method_names,
                        method_totals=method_totals,
                        forecast_days=forecast_days,
                        sales_data_dict=st.session_state.individual_sales_data
                    )
                    display_factcheck_section(factcheck_prompt, key_suffix="individual_matrix_v19")
                    
                    st.rerun()
            else:
                # 通常の単一予測方法の場合
                results = []
                
                for product, sales_data in st.session_state.individual_sales_data.items():
                    try:
                        forecast, method_message = forecast_with_vertex_ai(
                            sales_data, forecast_days, method, product,
                            baseline_method=baseline_method,
                            auto_special_factors=auto_special_factors,
                            include_quantiles=include_quantiles,
                            order_mode=order_mode,
                            backtest_days=backtest_days,
                            # v20パラメータ
                            enable_zero_fill=enable_zero_fill,
                            stockout_periods=stockout_periods,
                            enable_trend=enable_trend,
                            use_daily_new_year=use_daily_new_year,
                            trend_window_days=trend_window_days
                        )
                        
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
                        import traceback
                        error_detail = traceback.format_exc()
                        st.warning(f"{safe_html(product)}の予測に失敗しました: {str(e)[:100]}")
                        logger.error(f"{product}の予測エラー: {e}\n{error_detail}")
                
                if results:
                    # 納品計画で使えるようにsession_stateに保存
                    if len(results) == 1:
                        st.session_state.forecast_data = results[0]['forecast']
                    else:
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
                    st.session_state.individual_forecast_results = results
                    st.rerun()  # 納品セクションを更新するため再描画
    
    # 予測結果の表示（session_stateから）
    # 「すべての方法で比較」のマトリックス結果がある場合
    if st.session_state.get('individual_all_methods_results'):
        matrix_results = st.session_state.individual_all_methods_results
        method_names = []
        for product_methods in matrix_results.values():
            for method_name in product_methods.keys():
                if method_name not in method_names:
                    method_names.append(method_name)
        
        st.success("✅ すべての予測方法で比較完了！")
        
        # マトリックス形式の表を作成
        st.write("### 📊 商品×予測方法 マトリックス表")
        
        table_data = []
        method_totals = {m: 0 for m in method_names}
        
        for product, methods in matrix_results.items():
            row = {'商品名': product}
            for method_name in method_names:
                value = methods.get(method_name, 0)
                row[method_name] = f"{value:,}体"
                method_totals[method_name] += value
            table_data.append(row)
        
        # 合計行を追加
        total_row = {'商品名': '**合計**'}
        for method_name in method_names:
            total_row[method_name] = f"**{method_totals[method_name]:,}体**"
        table_data.append(total_row)
        
        df_matrix = pd.DataFrame(table_data)
        st.dataframe(df_matrix, use_container_width=True, hide_index=True)
        
        # 予測方法ごとの合計をメトリクスで表示
        st.write("### 📈 予測方法別 合計")
        
        num_methods = len(method_names)
        cols = st.columns(min(num_methods, 4))
        for i, method_name in enumerate(method_names):
            icon = "🚀" if "Vertex" in method_name else "📈" if "季節" in method_name else "📊" if "移動" in method_name else "📉"
            short_name = method_name.replace("（統計）", "").replace("（推奨）", "")
            with cols[i % 4]:
                st.metric(f"{icon} {short_name}", f"{method_totals[method_name]:,}体")
        
        # ファクトチェック用プロンプトセクション（マトリックス）
        individual_sales_data = st.session_state.get('individual_sales_data', {})
        # forecast_daysを取得（session_stateの個別予測結果から推測）
        forecast_days_matrix = 180  # デフォルト値
        if st.session_state.get('individual_forecast_results'):
            first_result = st.session_state.individual_forecast_results[0]
            if first_result.get('forecast') is not None and not first_result['forecast'].empty:
                forecast_days_matrix = len(first_result['forecast'])
        
        factcheck_prompt_matrix = generate_factcheck_prompt_matrix(
            matrix_results=matrix_results,
            method_names=method_names,
            method_totals=method_totals,
            forecast_days=forecast_days_matrix,
            individual_sales_data=individual_sales_data
        )
        display_factcheck_section(factcheck_prompt_matrix, key_suffix="matrix")
    
    # 通常の予測結果がある場合
    elif 'individual_forecast_results' in st.session_state and st.session_state.individual_forecast_results:
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
        
        # ファクトチェック用プロンプトセクション（個別予測）
        individual_sales_data = st.session_state.get('individual_sales_data', {})
        # forecast_daysを取得
        forecast_days_individual = 180  # デフォルト値
        if results and results[0].get('forecast') is not None:
            forecast_df = results[0]['forecast']
            if forecast_df is not None and not forecast_df.empty:
                forecast_days_individual = len(forecast_df)
        
        factcheck_prompt_individual = generate_factcheck_prompt_individual(
            results=results,
            forecast_days=forecast_days_individual,
            individual_sales_data=individual_sales_data
        )
        display_factcheck_section(factcheck_prompt_individual, key_suffix="individual")


def render_delivery_section():
    """納品計画セクション（個別モード対応）"""
    st.markdown('<p class="section-header">⑤ 納品計画を立てる</p>', unsafe_allow_html=True)
    
    # 予測結果があるかチェック
    individual_results = st.session_state.get('individual_forecast_results', [])
    forecast = st.session_state.get('forecast_data')
    forecast_total = st.session_state.get('forecast_total', 0)
    all_methods_results = st.session_state.get('individual_all_methods_results', {})
    
    # 予測結果がない場合（individual_results、forecast、forecast_total、all_methods_resultsのいずれもない）
    has_any_forecast = (
        (individual_results and len(individual_results) > 0) or
        (forecast is not None and not (isinstance(forecast, pd.DataFrame) and forecast.empty)) or
        (forecast_total > 0) or
        (all_methods_results and len(all_methods_results) > 0)
    )
    
    if not has_any_forecast:
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
    
    # ========== 削除フラグの処理（ページ先頭で実行） ==========
    if st.session_state.get('pending_delete_product'):
        product_to_delete = st.session_state.pending_delete_product
        if product_to_delete in st.session_state.selected_products:
            st.session_state.selected_products.remove(product_to_delete)
        if product_to_delete in st.session_state.product_groups:
            del st.session_state.product_groups[product_to_delete]
        st.session_state.sales_data = None
        st.session_state.forecast_data = None
        st.session_state.individual_sales_data = {}
        st.session_state.individual_forecast_results = []
        st.session_state.individual_all_methods_results = {}
        st.session_state.pending_delete_product = None
    
    if not init_data():
        st.stop()
    
    render_header()
    st.divider()
    render_main_tabs()
    
    st.divider()
    
    # バージョン情報（v20更新）
    version_info = "v20 (精度向上版 - 0埋め・トレンド・正月日別対応)"
    if VERTEX_AI_AVAILABLE:
        version_info += " | 🚀 Vertex AI: 有効"
    else:
        version_info += " | ⚠️ Vertex AI: 未設定"
    
    st.caption(f"⛩️ 酒列磯前神社 授与品管理システム {version_info}")


if __name__ == "__main__":
    main()
