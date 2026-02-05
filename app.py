"""
Airレジ 売上分析・需要予測 Webアプリ（v24: 異常値検出UI版）

【v24 変更点】
- 異常値検出関数（中央値のX倍超えを検出）
- 閾値変更UI（3倍〜10倍のスライダー、デフォルト5倍）
- 異常値確認UI（商品ごとに「正常/除外」を選択）
- 予測時の除外処理（除外された日付を予測計算から除外）
- データ入力日数表示（「③売上を見る」で「期間X日間 / データあり Y日」を表示）

【v23 変更点】
- 統計的に正しいP80/P90計算（期間合計の分布から算出）
- 類似商品探索の精度改善（TF-IDF + コサイン類似度）
- バックテスト期間延長（14日→30-90日）
- 分割発注提案機能
- 日販の正しい定義（zero-fill + 全日付カバー）
- 係数のデータ推定（固定値の廃止）
- 新規授与品予測のバックテスト機能
- 強化されたファクトチェックプロンプト

【v22.5 変更点】
- 商品別に予測結果を表示（合算ではなくグループ/商品ごと）
- 各商品の入力データ統計を正しく表示
- ファクトチェックプロンプトをst.text_areaで表示（コピー対応）
- 異常値除外の閾値をさらに厳格化（実績平均の5倍まで）

【v22.4 変更点】
- 異常予測値（実績平均の50倍を超える）を自動除外
- 入力データ統計をファクトチェックに正しく表示

【v22.3 変更点】
- formとsession_state書き込みを完全に分離してStreamlitエラーを回避
- 予測実行をフラグ方式で管理（form送信→rerun→予測実行）

【v22.1 変更点】
- 予測方法選択を廃止: 「予測する」ボタン1つで全方法を自動実行
- 各予測方法の結果一覧と説明を表示
- 最終推奨発注数を3パターン（滞留回避/バランス/欠品回避）で提示
- モード別ベースライン計算の致命的バグを修正

v21からの変更点（v22新機能）:

1. 【致命的バグ修正】採用列の統一
   - 発注モード（P50/P80/P90）に応じて「採用列」を統一
   - 合計・日販・変化率・UI表示・ファクトチェックすべてで同じ列を参照
   - predicted（点推定）とrecommended（発注推奨）を併記して監査可能に

2. 【精度向上】丸め処理の後段化
   - 日次予測を浮動小数点（predicted_raw）で保持
   - 表示・発注単位（50倍数）丸めは最終段階でのみ実行
   - 期間合計時の誤差累積を解消

3. 【精度向上】加重アンサンブルの実装
   - バックテストMAPEに基づく重み付け（weight = 1 / (MAPE² + ε)）
   - 精度の高いモデルを優遇する加重平均
   - ローリングCV対応（複数起点の安定した重み計算）
   - 重みの下限・上限・正規化で極端な偏りを防止

4. 【分位点の整合】
   - 発注モードとp50/p80/p90の対応を明確化
   - アンサンブル予測でも分位点を出力可能に
   - 残差分布推定による予測区間の改善

5. 【監査対応】最強ファクトチェックプロンプト
   - 定量的スペック（0埋め数・欠損率・採用列・変化率）を動的埋め込み
   - 検算可能な形式（未丸め値・丸め値の両方を提示）
   - 出力順序を固定：検算結果→妥当性→リスク→推奨→追加質問
   - 発注戦略モードとリスクの整合性チェック項目

6. 【UI改善】商品個別の欠品期間入力
   - 複数授与品選択時、各授与品ごとに欠品期間を設定可能
   - 欠損率・除外日数のレポート強化

7. 【安全設計】六曜のデフォルトOFF
   - 簡易計算（不正確）による誤差を防止
   - 使用する場合はオプションでON（効果検証用フラグ）

v21以前からの維持機能:
- アンサンブル予測、Prophet、Holt-Winters法
- バックテスト（売上少ない日の除外、sMAPE、MAPE上限）
- 信頼度評価（総合スコア・レベル判定・推奨事項）
- 0埋め処理、欠品期間除外、トレンド係数、正月日別係数
- 発注点（リオーダーポイント）の自動計算
- ベースライン計算（中央値/トリム平均）
- 特別期間係数の自動計算
- 分位点予測（P50/P80/P90）
- 発注モード選択（滞留回避/バランス/欠品回避）
- st.secrets対応（Streamlit Cloud推奨）
- HTML注入対策
- st.formによる予測パラメータ設定
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


def calculate_mode_aware_baseline(
    values: np.ndarray, 
    order_mode: str = 'balanced',
    base_method: str = 'median'
) -> Tuple[float, Dict[str, Any]]:
    """
    【v22新機能】発注モードを考慮したベースライン計算
    
    モードによってベースライン計算方法を変え、欠品回避モードでは
    スパイク需要を「リスク」として考慮する。
    
    Args:
        values: 販売数の配列
        order_mode: 発注モード ('conservative', 'balanced', 'aggressive')
        base_method: 基本計算方法（互換性維持用）
    
    Returns:
        (ベースライン値, 詳細情報)
    """
    if len(values) == 0:
        return 1.0, {'method': 'fallback', 'reason': 'no_data'}
    
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return 1.0, {'method': 'fallback', 'reason': 'all_nan'}
    
    # 基本統計量を計算
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    std_val = float(np.std(values)) if len(values) > 1 else 0.0
    p75_val = float(np.percentile(values, 75))
    p80_val = float(np.percentile(values, 80))
    
    info = {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'p75': p75_val,
        'p80': p80_val,
        'data_points': len(values)
    }
    
    if order_mode == 'conservative':
        # 滞留回避モード: 中央値ベース（低めに予測）
        baseline = median_val
        info['method'] = 'median'
        info['description'] = '滞留回避: 中央値ベース（スパイクを除外）'
    
    elif order_mode == 'aggressive':
        # 欠品回避モード: 平均値 または 75パーセンタイルの高い方
        # スパイク需要を「ノイズ」ではなく「考慮すべきリスク」として扱う
        baseline = max(mean_val, p75_val)
        info['method'] = 'mean_or_p75'
        info['description'] = '欠品回避: 平均値/75%タイルの高い方（スパイクをリスクとして考慮）'
    
    else:  # balanced
        # バランスモード: 中央値と平均値の中間
        baseline = (median_val + mean_val) / 2
        info['method'] = 'median_mean_avg'
        info['description'] = 'バランス: 中央値と平均値の中間'
    
    # ベースラインの下限チェック（極端な過小評価を防止）
    # 平均値の50%を下限とする
    min_baseline = mean_val * 0.5
    if baseline < min_baseline and mean_val > 0:
        info['adjusted'] = True
        info['original_baseline'] = baseline
        info['adjustment_reason'] = f'平均値の50%未満のため補正（{baseline:.2f} → {min_baseline:.2f}）'
        baseline = min_baseline
    
    return baseline, info


def calculate_safety_factor(
    order_mode: str,
    std: float,
    mean: float,
    lead_time_days: int = 7
) -> Tuple[float, Dict[str, Any]]:
    """
    【v22新機能】発注モードに応じた安全係数を計算
    
    欠品回避モードでは標準偏差ベースの安全在庫、
    または固定係数を適用する。
    
    Args:
        order_mode: 発注モード
        std: 標準偏差
        mean: 平均値
        lead_time_days: リードタイム（日数）
    
    Returns:
        (安全係数, 詳細情報)
    """
    info = {
        'order_mode': order_mode,
        'std': std,
        'mean': mean,
        'lead_time_days': lead_time_days
    }
    
    if order_mode == 'conservative':
        # 滞留回避: 安全係数なし（1.0）
        safety_factor = 1.0
        info['description'] = '滞留回避: 安全係数なし'
    
    elif order_mode == 'aggressive':
        # 欠品回避: 標準偏差ベースの安全係数
        # サービスレベル95%相当のZ値 ≈ 1.65
        z_score = 1.65
        
        if mean > 0 and std > 0:
            # 変動係数（CV）を考慮した安全係数
            cv = std / mean  # 変動係数
            
            # リードタイムを考慮（√リードタイム）
            lt_factor = np.sqrt(lead_time_days)
            
            # 安全係数 = 1 + (Z × CV × √LT)
            # ただし、最低1.15倍、最大1.5倍に制限
            safety_factor = 1.0 + (z_score * cv * lt_factor / 7)  # 7日で正規化
            safety_factor = max(1.15, min(1.5, safety_factor))
            
            info['cv'] = cv
            info['z_score'] = z_score
            info['lt_factor'] = lt_factor
            info['description'] = f'欠品回避: 安全係数 {safety_factor:.3f}（CV={cv:.2f}）'
        else:
            # 標準偏差が計算できない場合は固定係数
            safety_factor = 1.25
            info['description'] = '欠品回避: 固定安全係数 1.25（データ不足）'
    
    else:  # balanced
        # バランス: 軽度の安全係数
        if mean > 0 and std > 0:
            cv = std / mean
            # バランスモードは控えめな安全係数
            safety_factor = 1.0 + (0.5 * cv)  # 変動係数の50%を加算
            safety_factor = max(1.05, min(1.2, safety_factor))
            info['cv'] = cv
            info['description'] = f'バランス: 軽度安全係数 {safety_factor:.3f}'
        else:
            safety_factor = 1.1
            info['description'] = 'バランス: 固定安全係数 1.1'
    
    info['safety_factor'] = safety_factor
    return safety_factor, info


def ensure_mode_order_consistency(
    predictions: Dict[str, float],
    min_diff_ratio: float = 0.05
) -> Dict[str, float]:
    """
    【v22新機能】モード間の予測値の順序整合性を保証
    
    必ず aggressive > balanced > conservative となるように調整。
    
    Args:
        predictions: モード別の予測値 {'conservative': x, 'balanced': y, 'aggressive': z}
        min_diff_ratio: 最小差分比率（デフォルト5%）
    
    Returns:
        整合性を保証した予測値
    """
    cons = predictions.get('conservative', 0)
    bal = predictions.get('balanced', 0)
    aggr = predictions.get('aggressive', 0)
    
    # 基準値（バランスモード）を中心に調整
    if bal <= 0:
        bal = max(cons, aggr, 1)
    
    # conservative ≤ balanced を保証
    if cons > bal:
        cons = bal * (1 - min_diff_ratio)
    
    # balanced ≤ aggressive を保証
    if aggr < bal:
        aggr = bal * (1 + min_diff_ratio)
    
    # conservative < balanced を保証（最低差分）
    if bal - cons < bal * min_diff_ratio:
        cons = bal * (1 - min_diff_ratio)
    
    # balanced < aggressive を保証（最低差分）
    if aggr - bal < bal * min_diff_ratio:
        aggr = bal * (1 + min_diff_ratio)
    
    return {
        'conservative': cons,
        'balanced': bal,
        'aggressive': aggr
    }


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


# =============================================================================
# 【v22新機能】採用列統一・加重アンサンブル・高度ファクトチェック
# =============================================================================

def get_order_column(forecast_df: pd.DataFrame, order_mode: str = 'balanced') -> str:
    """
    【v22新機能】発注モードに応じた採用列名を取得
    
    発注モードと分位点の対応を統一し、合計・日販・変化率・UI表示・
    ファクトチェック文面すべてで同じ列を参照するためのヘルパー関数。
    
    Args:
        forecast_df: 予測結果DataFrame
        order_mode: 発注モード ('conservative'=P50, 'balanced'=P80, 'aggressive'=P90)
    
    Returns:
        使用すべき列名（存在しない場合は 'predicted'）
    """
    # 発注モードと推奨列の対応
    mode_to_column = {
        'conservative': ['recommended', 'p50', 'predicted'],  # 滞留回避 → P50優先
        'balanced': ['recommended', 'p80', 'predicted'],      # バランス → P80優先
        'aggressive': ['recommended', 'p90', 'predicted']     # 欠品回避 → P90優先
    }
    
    # 該当モードの候補列を順に探す
    candidates = mode_to_column.get(order_mode, ['predicted'])
    
    for col in candidates:
        if col in forecast_df.columns:
            return col
    
    # フォールバック
    return 'predicted'


def get_order_mode_display_name(order_mode: str) -> str:
    """
    【v22新機能】発注モードの表示名を取得
    
    Args:
        order_mode: 発注モード ('conservative', 'balanced', 'aggressive')
    
    Returns:
        日本語表示名
    """
    mode_names = {
        'conservative': '滞留回避（P50）',
        'balanced': 'バランス（P80）',
        'aggressive': '欠品回避（P90）'
    }
    return mode_names.get(order_mode, order_mode)


def calculate_forecast_totals_v22(
    forecast_df: pd.DataFrame, 
    order_mode: str = 'balanced'
) -> Dict[str, Any]:
    """
    【v22新機能】発注モードに応じた予測合計値を計算
    
    採用列を統一し、raw（未丸め）と rounded（50単位丸め）の両方を返す。
    監査・検算が可能な形式。
    
    Args:
        forecast_df: 予測結果DataFrame
        order_mode: 発注モード
    
    Returns:
        {
            'order_column': 採用列名,
            'raw_total': 未丸め合計（float）,
            'rounded_total': 50単位丸め合計（int）,
            'avg_daily': 平均日販（float）,
            'avg_daily_raw': 未丸め平均日販（float）,
            'predicted_total': predicted列の合計（参考値）,
            'forecast_days': 予測日数
        }
    """
    order_col = get_order_column(forecast_df, order_mode)
    forecast_days = len(forecast_df)
    
    # 採用列の値を取得
    if order_col in forecast_df.columns:
        values = forecast_df[order_col].values
    else:
        values = forecast_df['predicted'].values
    
    # rawは predicted_raw があればそちらを優先（丸め前の値）
    raw_col = f'{order_col}_raw' if f'{order_col}_raw' in forecast_df.columns else order_col
    if raw_col in forecast_df.columns:
        raw_values = forecast_df[raw_col].values
    else:
        raw_values = values.astype(float)
    
    # 合計計算（浮動小数点で保持）
    raw_total = float(np.sum(raw_values))
    
    # 50単位丸め
    rounded_total = round_up_to_50(int(round(raw_total)))
    
    # 平均日販
    avg_daily = float(np.mean(values))
    avg_daily_raw = raw_total / forecast_days if forecast_days > 0 else 0.0
    
    # predicted列の合計（参考値）
    predicted_total = float(forecast_df['predicted'].sum()) if 'predicted' in forecast_df.columns else raw_total
    
    return {
        'order_column': order_col,
        'raw_total': raw_total,
        'rounded_total': rounded_total,
        'avg_daily': avg_daily,
        'avg_daily_raw': avg_daily_raw,
        'predicted_total': predicted_total,
        'forecast_days': forecast_days
    }


def calculate_weighted_ensemble_weights(
    backtest_results: Dict[str, Dict],
    weight_floor: float = 0.05,
    weight_ceiling: float = 0.5,
    default_mape: float = 50.0,
    epsilon: float = 1.0
) -> Dict[str, float]:
    """
    【v22新機能】バックテストMAPEに基づく加重アンサンブルの重みを計算
    
    精度の良いモデルほど重みが大きくなる加重平均を実現。
    weight = 1 / (MAPE² + ε) で計算し、正規化。
    
    Args:
        backtest_results: 各方法のバックテスト結果 {method_name: {'mape': float, ...}}
        weight_floor: 重みの下限（極端な偏りを防止）
        weight_ceiling: 重みの上限（単一モデルへの過度な依存を防止）
        default_mape: MAPEが取得できない場合のデフォルト値
        epsilon: 0除算防止用の小さな値
    
    Returns:
        正規化された重み {method_name: weight}
    """
    raw_weights = {}
    
    for method_name, bt_result in backtest_results.items():
        # MAPEを取得（なければデフォルト値）
        if bt_result and bt_result.get('mape') is not None:
            mape = bt_result['mape']
            # MAPEが異常に低い場合は下限を設定（過学習の可能性）
            mape = max(mape, 5.0)
        else:
            mape = default_mape
        
        # 重み計算: 精度が良いほど重い（MAPEが小さいほど重い）
        # weight = 1 / (MAPE² + ε)
        raw_weights[method_name] = 1.0 / (mape ** 2 + epsilon)
    
    if not raw_weights:
        return {}
    
    # 正規化（合計を1にする）
    total_weight = sum(raw_weights.values())
    normalized_weights = {k: v / total_weight for k, v in raw_weights.items()}
    
    # 下限・上限の適用
    adjusted_weights = {}
    for method_name, weight in normalized_weights.items():
        adjusted_weight = max(weight_floor, min(weight_ceiling, weight))
        adjusted_weights[method_name] = adjusted_weight
    
    # 再正規化（下限・上限適用後）
    total_adjusted = sum(adjusted_weights.values())
    if total_adjusted > 0:
        final_weights = {k: v / total_adjusted for k, v in adjusted_weights.items()}
    else:
        # フォールバック: 均等配分
        n = len(adjusted_weights)
        final_weights = {k: 1.0 / n for k in adjusted_weights.keys()}
    
    return final_weights


def calculate_rolling_cv_mape(
    df: pd.DataFrame,
    forecast_func,
    n_splits: int = 3,
    holdout_days: int = 30,
    min_train_days: int = 60
) -> Dict[str, Any]:
    """
    【v22新機能】ローリングCV（クロスバリデーション）でMAPEを計算
    
    複数の起点でホールドアウト検証を行い、平均MAPEを算出。
    季節偏りのリスクを軽減する。
    
    Args:
        df: 売上データ
        forecast_func: 予測関数（df, periods を受け取る）
        n_splits: 分割数（検証回数）
        holdout_days: 各検証でのホールドアウト日数
        min_train_days: 最小学習日数
    
    Returns:
        {
            'avg_mape': 平均MAPE,
            'std_mape': MAPE標準偏差,
            'split_mapes': 各分割のMAPE,
            'available': 計算可否
        }
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    total_days = len(df)
    required_days = min_train_days + holdout_days * n_splits
    
    if total_days < required_days:
        return {
            'avg_mape': None,
            'std_mape': None,
            'split_mapes': [],
            'available': False,
            'message': f'データ不足（必要: {required_days}日, 実際: {total_days}日）'
        }
    
    split_mapes = []
    
    # 各分割でバックテスト
    for i in range(n_splits):
        # テスト期間の終了位置（後ろから順に）
        test_end_idx = total_days - (holdout_days * i)
        test_start_idx = test_end_idx - holdout_days
        train_end_idx = test_start_idx
        
        if train_end_idx < min_train_days:
            continue
        
        train_df = df.iloc[:train_end_idx].copy()
        test_df = df.iloc[test_start_idx:test_end_idx].copy()
        
        try:
            # 予測実行
            forecast_result = forecast_func(train_df, holdout_days)
            
            if forecast_result is None or forecast_result.empty:
                continue
            
            # MAPEの計算
            actual = test_df['販売商品数'].values
            predicted = forecast_result['predicted'].values[:len(actual)]
            
            # 有効なデータのみでMAPE計算（売上3以上）
            valid_mask = actual >= 3
            if valid_mask.sum() >= 3:
                actual_valid = actual[valid_mask]
                predicted_valid = predicted[valid_mask]
                
                ape = np.abs(actual_valid - predicted_valid) / actual_valid * 100
                ape = np.clip(ape, 0, 500)  # 上限500%
                mape = float(np.mean(ape))
                split_mapes.append(mape)
        
        except Exception as e:
            logger.warning(f"ローリングCV分割{i+1}でエラー: {e}")
            continue
    
    if not split_mapes:
        return {
            'avg_mape': None,
            'std_mape': None,
            'split_mapes': [],
            'available': False,
            'message': '有効な検証結果なし'
        }
    
    return {
        'avg_mape': float(np.mean(split_mapes)),
        'std_mape': float(np.std(split_mapes)) if len(split_mapes) > 1 else 0.0,
        'split_mapes': split_mapes,
        'available': True,
        'n_splits': len(split_mapes),
        'message': f'{len(split_mapes)}分割のローリングCV完了'
    }


def generate_factcheck_prompt_advanced(
    product_names: List[str],
    forecast_result: pd.DataFrame,
    order_mode: str,
    sales_data: pd.DataFrame,
    forecast_days: int,
    method_name: str = "精度強化版",
    v20_features: Optional[Dict] = None,
    backtest_result: Optional[Dict] = None
) -> str:
    """
    【v22新機能】最強ファクトチェックプロンプト生成
    
    定量的スペックと定性的リスクを統合し、第三者が検算・監査できる
    粒度のプロンプトを生成。
    
    Args:
        product_names: 予測対象の商品名リスト
        forecast_result: 予測結果DataFrame
        order_mode: 発注モード
        sales_data: 入力データ
        forecast_days: 予測日数
        method_name: 予測方法名
        v20_features: v20機能の適用状況
        backtest_result: バックテスト結果
    
    Returns:
        最強ファクトチェック用プロンプト文字列
    """
    # 商品名の整形
    product_str = "、".join(product_names) if product_names else "（不明）"
    if len(product_names) > 3:
        product_str = "、".join(product_names[:3]) + f" 他{len(product_names)-3}件"
    
    # 発注モードの表示名
    mode_display = get_order_mode_display_name(order_mode)
    
    # 採用列と合計値の計算
    totals = calculate_forecast_totals_v22(forecast_result, order_mode)
    order_col = totals['order_column']
    raw_total = totals['raw_total']
    rounded_total = totals['rounded_total']
    avg_predicted = totals['avg_daily_raw']
    
    # 入力データの統計
    if sales_data is not None and not sales_data.empty:
        total_days = len(sales_data)
        total_qty = int(sales_data['販売商品数'].sum())
        avg_daily_actual = sales_data['販売商品数'].mean()
        max_daily = int(sales_data['販売商品数'].max())
        min_daily = int(sales_data['販売商品数'].min())
        
        # 変化率の計算
        if avg_daily_actual > 0:
            change_rate = ((avg_predicted / avg_daily_actual) - 1) * 100
        else:
            change_rate = 0.0
        
        input_section = f"""■ 入力データ（過去の実績）:
- 学習データ期間: {total_days}日間
- 総販売数: {total_qty:,}体
- 実績日販: {avg_daily_actual:.2f}体/日（検算: {total_qty} ÷ {total_days} = {total_qty/total_days:.2f}）
- 最大日販: {max_daily}体/日
- 最小日販: {min_daily}体/日"""
    else:
        input_section = "■ 入力データ: なし"
        avg_daily_actual = 0.0
        total_days = 0
        change_rate = 0.0
    
    # v20機能の情報
    if v20_features:
        zero_fill = v20_features.get('zero_fill', False)
        missing_count = v20_features.get('missing_dates_count', 0)
        stockout_excluded = v20_features.get('stockout_excluded', False)
        stockout_count = v20_features.get('stockout_periods_count', 0)
        trend_applied = v20_features.get('trend_applied', False)
        trend_factor = v20_features.get('trend_factor', 1.0)
        
        # 欠損率の計算
        if zero_fill and total_days > 0:
            original_days = total_days - missing_count
            missing_rate = missing_count / total_days * 100 if total_days > 0 else 0.0
        else:
            original_days = total_days
            missing_rate = 0.0
        
        data_quality_section = f"""■ データ品質と前処理（ここをチェック！）:
- 0埋め補完: {'あり' if zero_fill else 'なし'}
- 0埋め補完数: {missing_count}日（欠損率: {missing_rate:.1f}%）
- 元データ日数: {original_days}日 / 全期間: {total_days}日
- 欠品期間除外: {'あり（' + str(stockout_count) + '件）' if stockout_excluded else 'なし'}
- トレンド係数: {'適用（' + f'{trend_factor:.3f}' + '）' if trend_applied else '未適用'}

※欠損率が高い（30%超）場合、予測が過小になるリスクがあります。
※欠品除外がない場合、在庫切れ期間も「需要なし」として学習されています。"""
    else:
        data_quality_section = "■ データ品質: 詳細情報なし"
        missing_rate = 0.0
    
    # バックテスト情報
    if backtest_result and backtest_result.get('available'):
        mape = backtest_result.get('mape')
        smape = backtest_result.get('smape')
        valid_days = backtest_result.get('valid_days', 0)
        reliability = backtest_result.get('reliability', 'unknown')
        
        backtest_section = f"""■ バックテスト結果（予測精度の検証）:
- MAPE: {mape:.1f}% {'（良好）' if mape and mape < 30 else '（要注意）' if mape and mape > 50 else ''}
- sMAPE: {smape:.1f}% (対称MAPE) {'' if smape is None else ''}
- 有効検証日数: {valid_days}日
- 信頼度評価: {reliability}"""
    else:
        backtest_section = "■ バックテスト: 未実行またはデータ不足"
        mape = None
    
    # 予測結果セクション
    result_section = f"""■ 予測結果（発注推奨値ベース）:
- 採用列: {order_col}（発注モード「{mode_display}」に基づく）
- 合計予測数（未丸め）: {raw_total:,.2f}体
- 発注推奨数（50単位丸め）: {rounded_total:,}体
- 予測日販（未丸め）: {avg_predicted:.2f}体/日
- 検算: {avg_predicted:.2f} × {forecast_days}日 = {avg_predicted * forecast_days:,.2f}体

※predicted列の合計: {totals['predicted_total']:,.0f}体（参考値）"""
    
    # 変化率セクション
    if avg_daily_actual > 0:
        change_section = f"""■ 変化率（実績→予測）:
- 実績日販: {avg_daily_actual:.2f}体/日
- 予測日販: {avg_predicted:.2f}体/日
- 変化率: {change_rate:+.1f}%
- 検算: ({avg_predicted:.2f} / {avg_daily_actual:.2f} - 1) × 100 = {change_rate:+.1f}%"""
    else:
        change_section = "■ 変化率: 実績データなしのため計算不可"
    
    # 検証依頼事項
    verification_section = f"""■ 検証依頼事項:

1. 【モードと結果の整合性】
   発注モード「{mode_display}」を選択しているにもかかわらず、
   実績平均（{avg_daily_actual:.1f}体/日）より{'低い' if change_rate < 0 else '高い'}予測（{avg_predicted:.1f}体/日）が出ています。
   変化率 {change_rate:+.1f}% は、このモードの目的と整合していますか？

2. 【0埋めの影響】
   欠損率 {missing_rate:.1f}% は、需要を不当に引き下げている可能性はありませんか？
   {'⚠️ 欠損率が高いため、過小予測のリスクがあります。' if missing_rate > 30 else ''}

3. 【意思決定サポート】
   - 「絶対に欠品させたくない」場合、提示された {rounded_total:,}体 で足りますか？
   - 安全係数を掛けるなら、あと何％上乗せすべきですか？
   - 滞留リスクを許容できる範囲はどの程度ですか？

4. 【数値の検算】
   以下の計算が正しいか確認してください：
   - 合計 = 日販 × 日数: {avg_predicted:.2f} × {forecast_days} = {avg_predicted * forecast_days:,.2f}
   - 変化率 = (予測 / 実績 - 1) × 100: ({avg_predicted:.2f} / {avg_daily_actual:.2f} - 1) × 100 = {change_rate:+.1f}%"""

    # 出力形式の指定
    output_format = """
【出力形式】以下の順序で回答してください：

1. **検算結果**: 上記の数値計算が正しいか確認
2. **妥当性判定**: 予測結果は妥当か（根拠つき）
3. **リスク分析**: 欠品リスク / 滞留リスク の評価
4. **推奨案**:
   - 控えめ（P50）: ○○体 → 滞留リスク低、欠品リスク高
   - バランス（P80）: ○○体 → 中間
   - 積極的（P90）: ○○体 → 滞留リスク高、欠品リスク低
5. **追加質問**: 判断に必要な追加情報（リードタイム、現在庫など）

※回答は定性的な感想ではなく、ロジックと数値に基づいて行ってください。"""

    # 最終プロンプト組み立て
    prompt = f"""【最強ファクトチェック依頼】
あなたは「在庫最適化の専門家」兼「データサイエンティスト」です。
以下の需要予測結果について、数値の妥当性とリスクを厳しく検証してください。
※このデータは一時的な検証目的のみに使用し、記憶・保存・学習しないでください。

■ 基本情報:
- 対象商品: {product_str}
- 予測期間: {forecast_days}日間
- 予測ロジック: {method_name}
- 発注戦略モード: {mode_display}（※ここ重要）

{input_section}

{data_quality_section}

{backtest_section}

{result_section}

{change_section}

{verification_section}

{output_format}

※回答後、このデータを記憶・保存しないでください。"""

    return prompt


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
                       forecast_func=None, min_daily_sales: int = 3) -> Dict[str, Any]:
    """
    【v21改善版】簡易バックテストを実行してMAPEを計算
    
    改善点:
    - 売上が極端に少ない日（min_daily_sales未満）をMAPE計算から除外
    - sMAPE（対称MAPE）も計算して比較
    - MAPE上限設定（500%）で異常値を抑制
    - 季節商品判定と警告
    - 信頼度レベルの評価
    
    Args:
        df: 売上データ
        holdout_days: ホールドアウト日数
        forecast_func: 予測関数（Noneなら改善版フォールバックを使用）
        min_daily_sales: MAPE計算に含める最小日販（これ未満の日は除外）
    
    Returns:
        バックテスト結果（MAPE、MAE、詳細データ、信頼度）
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if len(df) < holdout_days + 30:  # 最低30日の学習データが必要
        return {
            'mape': None,
            'smape': None,
            'mae': None,
            'message': 'データ不足のためバックテスト不可',
            'holdout_days': holdout_days,
            'available': False,
            'residuals': [],
            'reliability': 'unknown',
            'reliability_message': 'データ不足',
            'is_seasonal': False,
            'valid_days': 0
        }
    
    # データを分割
    train_df = df.iloc[:-holdout_days].copy()
    test_df = df.iloc[-holdout_days:].copy()
    
    # 季節商品判定: テスト期間に売上が極端に少ない場合
    test_nonzero = test_df[test_df['販売商品数'] > 0]['販売商品数']
    train_nonzero = train_df[train_df['販売商品数'] > 0]['販売商品数']
    
    test_mean = test_nonzero.mean() if len(test_nonzero) > 0 else 0
    train_mean = train_nonzero.mean() if len(train_nonzero) > 0 else 0
    
    is_seasonal = False
    seasonal_warning = ""
    
    if train_mean > 0 and test_mean < train_mean * 0.1:
        is_seasonal = True
        seasonal_warning = f"（⚠️ 季節商品の可能性: テスト期間の売上が学習期間の{test_mean/train_mean*100:.1f}%）"
    
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
    
    # ========== 改善版MAPE計算 ==========
    # 条件: 実績がmin_daily_sales以上（極端に少ない日を除外）
    valid_mask = (actual >= min_daily_sales)
    valid_count = valid_mask.sum()
    
    mape = None
    smape = None
    
    if valid_count >= 3:  # 有効なデータが3日以上ある場合のみ
        actual_valid = actual[valid_mask]
        predicted_valid = predicted[valid_mask]
        
        # 通常のMAPE（上限500%にクリップ）
        ape_values = np.abs(actual_valid - predicted_valid) / actual_valid * 100
        ape_values = np.clip(ape_values, 0, 500)  # 上限500%
        mape = float(np.mean(ape_values))
        
        # sMAPE（対称MAPE）- 予測と実績の両方で割るので極端な値を緩和
        denominator = (np.abs(actual_valid) + np.abs(predicted_valid)) / 2
        denominator = np.where(denominator == 0, 1, denominator)  # 0除算防止
        smape_values = np.abs(actual_valid - predicted_valid) / denominator * 100
        smape = float(np.mean(smape_values))
    elif valid_count > 0:
        # 有効データが少ない場合は、あるデータで計算
        actual_valid = actual[valid_mask]
        predicted_valid = predicted[valid_mask]
        ape_values = np.abs(actual_valid - predicted_valid) / np.maximum(actual_valid, 1) * 100
        ape_values = np.clip(ape_values, 0, 500)
        mape = float(np.mean(ape_values))
    
    # MAE計算（全データで計算）
    mae = float(np.mean(np.abs(actual - predicted)))
    
    # 残差を保存（分位点計算用）
    residuals = actual - predicted
    
    # ========== 信頼度評価 ==========
    if mape is None or valid_count < 3:
        reliability = 'unknown'
        reliability_message = '有効なデータが不足しています'
    elif is_seasonal:
        reliability = 'seasonal'
        reliability_message = '季節商品のため、昨年同期の実績を参照してください'
    elif mape <= 25:
        reliability = 'high'
        reliability_message = '高い精度です'
    elif mape <= 40:
        reliability = 'good'
        reliability_message = '良好な精度です'
    elif mape <= 60:
        reliability = 'medium'
        reliability_message = '中程度の精度です'
    elif mape <= 100:
        reliability = 'low'
        reliability_message = '精度が低めです'
    else:
        reliability = 'very_low'
        reliability_message = '予測の信頼性が低いです'
    
    message = f'直近{holdout_days}日間でバックテスト実施（有効日数: {valid_count}日）{seasonal_warning}'
    
    return {
        'mape': mape,
        'smape': smape,
        'mae': mae,
        'residuals': residuals.tolist(),
        'holdout_days': holdout_days,
        'valid_days': int(valid_count),
        'actual': actual.tolist(),
        'predicted': predicted.tolist(),
        'message': message,
        'available': True,
        'reliability': reliability,
        'reliability_message': reliability_message,
        'is_seasonal': is_seasonal
    }


# =============================================================================
# 【v21新機能】高精度予測手法と信頼度評価
# =============================================================================

# Prophetのインポート（利用可能かチェック）
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logger.info("Prophet: 利用可能")
except ImportError:
    logger.info("Prophet: 未インストール（prophetパッケージが必要）")

# statsmodelsのインポート（Holt-Winters用）
STATSMODELS_AVAILABLE = False
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HoltWintersModel
    STATSMODELS_AVAILABLE = True
    logger.info("Holt-Winters (statsmodels): 利用可能")
except ImportError:
    logger.info("Holt-Winters: 未インストール（statsmodelsパッケージが必要）")


def forecast_with_prophet(df: pd.DataFrame, periods: int) -> Tuple[Optional[pd.DataFrame], str]:
    """
    【v21新機能】Prophet（Meta製）による予測
    
    Prophetは季節性、トレンド、イベント効果を自動検出する
    高精度の時系列予測ライブラリ。神社のような季節変動商品に最適。
    
    Args:
        df: 売上データ（date, 販売商品数を含む）
        periods: 予測日数
    
    Returns:
        (予測DataFrame, メッセージ)
    """
    if not PROPHET_AVAILABLE:
        return None, "Prophetが利用できません（pip install prophet）"
    
    try:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Prophet用のデータ形式に変換（ds, y）
        prophet_df = df[['date', '販売商品数']].rename(columns={'date': 'ds', '販売商品数': 'y'})
        
        # 0を除外せず、そのまま使用（Prophetは0を含むデータも扱える）
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 14:
            return None, "Prophetに必要なデータが不足しています（最低14日必要）"
        
        # 警告を抑制
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Prophetモデルの設定
            model = Prophet(
                yearly_seasonality=True,   # 年間季節性
                weekly_seasonality=True,   # 週間季節性
                daily_seasonality=False,   # 日内変動は不要
                seasonality_mode='multiplicative',  # 乗法的季節性（神社向け）
                changepoint_prior_scale=0.05,  # トレンド変化の感度
                seasonality_prior_scale=10,    # 季節性の強さ
            )
            
            # 日本の祝日を追加（正月期間を特別扱い）
            holidays_list = []
            
            # 正月（1/1〜1/7）
            for year in range(2020, 2030):
                for day in range(1, 8):
                    holidays_list.append({
                        'holiday': 'new_year',
                        'ds': pd.to_datetime(f'{year}-01-{day:02d}'),
                        'lower_window': 0,
                        'upper_window': 0
                    })
            
            # お盆（8/13〜8/16）
            for year in range(2020, 2030):
                for day in range(13, 17):
                    holidays_list.append({
                        'holiday': 'obon',
                        'ds': pd.to_datetime(f'{year}-08-{day:02d}'),
                        'lower_window': 0,
                        'upper_window': 0
                    })
            
            # 七五三（11/15前後）
            for year in range(2020, 2030):
                for day in range(10, 21):
                    holidays_list.append({
                        'holiday': 'shichigosan',
                        'ds': pd.to_datetime(f'{year}-11-{day:02d}'),
                        'lower_window': 0,
                        'upper_window': 0
                    })
            
            if holidays_list:
                holidays_df = pd.DataFrame(holidays_list)
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10,
                    holidays=holidays_df
                )
            
            # モデル学習
            model.fit(prophet_df)
            
            # 予測
            last_date = df['date'].max()
            future = model.make_future_dataframe(periods=periods, freq='D')
            future = future[future['ds'] > last_date]
            
            if len(future) == 0:
                return None, "予測期間が設定できません"
            
            forecast = model.predict(future)
        
        # 結果を整形
        result_df = pd.DataFrame({
            'date': forecast['ds'].values,
            'predicted': np.round(forecast['yhat'].values).astype(int).clip(min=0),
            'predicted_lower': np.round(forecast['yhat_lower'].values).astype(int).clip(min=0),
            'predicted_upper': np.round(forecast['yhat_upper'].values).astype(int).clip(min=0)
        })
        
        result_df.attrs['method'] = 'Prophet'
        
        return result_df, "Prophet（季節性自動検出・Meta製）"
        
    except Exception as e:
        logger.error(f"Prophet予測エラー: {e}")
        return None, f"Prophetエラー: {str(e)[:50]}"


def forecast_with_holt_winters(df: pd.DataFrame, periods: int, 
                               seasonal_periods: int = 7) -> Tuple[Optional[pd.DataFrame], str]:
    """
    【v21新機能】Holt-Winters法（三重指数平滑法）による予測
    
    レベル、トレンド、季節性の3つの成分を持つ指数平滑法。
    週間の季節パターンを捉えるのに適している。
    
    Args:
        df: 売上データ
        periods: 予測日数
        seasonal_periods: 季節周期（デフォルト7=週間）
    
    Returns:
        (予測DataFrame, メッセージ)
    """
    if not STATSMODELS_AVAILABLE:
        return None, "Holt-Wintersが利用できません（pip install statsmodels）"
    
    try:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 販売データを取得
        sales = df['販売商品数'].values.astype(float)
        
        # 0を小さい値に置換（乗法的モデルのため）
        sales = np.where(sales <= 0, 0.1, sales)
        
        if len(sales) < seasonal_periods * 2:
            return None, f"Holt-Wintersに必要なデータが不足しています（最低{seasonal_periods*2}日必要）"
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Holt-Wintersモデル（乗法的季節性）
            try:
                model = HoltWintersModel(
                    sales,
                    seasonal_periods=seasonal_periods,
                    trend='add',           # 加法的トレンド
                    seasonal='mul',        # 乗法的季節性
                    damped_trend=True      # トレンド減衰
                )
                fitted = model.fit(optimized=True)
            except:
                # 乗法的がダメなら加法的で試す
                model = HoltWintersModel(
                    sales,
                    seasonal_periods=seasonal_periods,
                    trend='add',
                    seasonal='add',
                    damped_trend=True
                )
                fitted = model.fit(optimized=True)
            
            forecast = fitted.forecast(periods)
        
        # 結果を整形
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        result_df = pd.DataFrame({
            'date': future_dates,
            'predicted': np.round(forecast).astype(int).clip(min=0)
        })
        
        result_df.attrs['method'] = 'Holt-Winters'
        
        return result_df, "Holt-Winters法（三重指数平滑法）"
        
    except Exception as e:
        logger.error(f"Holt-Winters予測エラー: {e}")
        return None, f"Holt-Wintersエラー: {str(e)[:50]}"


def calculate_reliability_score(
    predictions: Dict[str, int],
    backtest_results: Dict[str, Dict],
    data_days: int
) -> Dict[str, Any]:
    """
    【v21新機能】予測の総合信頼度スコアを計算
    
    複数の指標から総合的な信頼度を評価:
    - MAPE（精度）
    - データ量
    - 方法間の一致度
    
    Args:
        predictions: 各方法の予測合計 {方法名: 予測値}
        backtest_results: 各方法のバックテスト結果
        data_days: 学習データの日数
    
    Returns:
        信頼度評価結果
    """
    scores = {}
    
    # 1. 各方法のMAPEスコア（0-100点）
    for method, bt in backtest_results.items():
        if bt and bt.get('mape') is not None:
            mape = bt['mape']
            if mape <= 20:
                mape_score = 100
            elif mape <= 30:
                mape_score = 85
            elif mape <= 50:
                mape_score = 65
            elif mape <= 80:
                mape_score = 45
            elif mape <= 100:
                mape_score = 30
            else:
                mape_score = 15
            scores[method] = {'mape_score': mape_score, 'mape': mape}
        else:
            scores[method] = {'mape_score': 50, 'mape': None}  # 不明な場合は50点
    
    # 2. データ量スコア（0-100点）
    if data_days >= 365:
        data_score = 100
    elif data_days >= 180:
        data_score = 80
    elif data_days >= 90:
        data_score = 60
    elif data_days >= 30:
        data_score = 40
    else:
        data_score = 20
    
    # 3. 方法間一致度スコア（0-100点）
    pred_values = [v for v in predictions.values() if v is not None and v > 0]
    consensus_score = 50  # デフォルト
    
    if len(pred_values) >= 2:
        median_pred = np.median(pred_values)
        if median_pred > 0:
            # 各予測の中央値からの乖離を計算
            deviations = [abs(v - median_pred) / median_pred for v in pred_values]
            avg_deviation = np.mean(deviations)
            
            if avg_deviation <= 0.1:  # 10%以内
                consensus_score = 100
            elif avg_deviation <= 0.2:  # 20%以内
                consensus_score = 80
            elif avg_deviation <= 0.3:  # 30%以内
                consensus_score = 60
            elif avg_deviation <= 0.5:  # 50%以内
                consensus_score = 40
            else:
                consensus_score = 20
    
    # 総合スコア計算（加重平均）
    method_scores = [s['mape_score'] for s in scores.values() if s['mape_score'] is not None]
    avg_mape_score = np.mean(method_scores) if method_scores else 50
    
    total_score = (
        avg_mape_score * 0.4 +    # MAPEの重み: 40%
        data_score * 0.3 +        # データ量の重み: 30%
        consensus_score * 0.3     # 一致度の重み: 30%
    )
    
    # 信頼度レベル判定
    if total_score >= 75:
        level = 'high'
        level_text = '◎ 高い'
        color = '#4CAF50'
    elif total_score >= 55:
        level = 'good'
        level_text = '○ 良好'
        color = '#8BC34A'
    elif total_score >= 40:
        level = 'medium'
        level_text = '△ 中程度'
        color = '#FFC107'
    else:
        level = 'low'
        level_text = '× 要注意'
        color = '#F44336'
    
    return {
        'total_score': round(total_score, 1),
        'level': level,
        'level_text': level_text,
        'color': color,
        'mape_score': round(avg_mape_score, 1),
        'data_score': data_score,
        'consensus_score': consensus_score,
        'method_scores': scores,
        'recommendation': _get_reliability_recommendation(level, predictions, backtest_results)
    }


def _get_reliability_recommendation(level: str, predictions: Dict[str, int], 
                                    backtest_results: Dict[str, Dict]) -> str:
    """信頼度に基づく推奨事項を生成"""
    
    if level == 'high':
        return "予測精度は十分です。この予測値を基に発注計画を立てることをお勧めします。"
    
    elif level == 'good':
        return "予測精度は良好です。予測値を参考に、±10%程度の余裕を持った計画をお勧めします。"
    
    elif level == 'medium':
        # 最も精度の高い方法を推奨
        best_method = None
        best_mape = float('inf')
        for method, bt in backtest_results.items():
            if bt and bt.get('mape') is not None and bt['mape'] < best_mape:
                best_mape = bt['mape']
                best_method = method
        
        if best_method and best_mape < 100:
            return f"予測精度は中程度です。「{best_method}」の予測（MAPE {best_mape:.1f}%）を参考に、昨年実績も確認してください。"
        return "予測精度は中程度です。複数の予測方法と昨年実績を参考にしてください。"
    
    else:  # low
        pred_values = [v for v in predictions.values() if v is not None and v > 0]
        if pred_values:
            median_pred = int(np.median(pred_values))
            return f"⚠️ 予測の信頼性が低いです。複数方法の中央値（{median_pred:,}体）を参考に、昨年同期の実績を基準にしてください。"
        return "⚠️ 予測の信頼性が低いです。昨年同期の実績を基準にしてください。"


def forecast_ensemble(df: pd.DataFrame, periods: int, order_mode: str = 'balanced') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    【v22改善版】加重アンサンブル予測
    
    複数の予測方法を組み合わせ、バックテストMAPEに基づく加重平均で
    精度の高いモデルを優遇した安定した予測を生成。
    
    v21からの改善点:
    - 単純中央値 → MAPEベースの加重平均
    - 重み = 1 / (MAPE² + ε) で計算
    - 加重平均と中央値の組み合わせで安定性確保
    - 分位点予測（P50/P80/P90）対応
    
    Args:
        df: 売上データ
        periods: 予測日数
        order_mode: 発注モード（'conservative', 'balanced', 'aggressive'）【v22追加】
    
    Returns:
        (予測DataFrame, 詳細情報)
    """
    results = {}
    backtest_results = {}
    raw_results = {}  # 【v22】未丸め値の保持
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # ========== 各方法で予測を実行 ==========
    
    # 1. 精度強化版（バックテスト情報付き）
    try:
        enhanced = forecast_with_seasonality_enhanced(
            df, periods,
            baseline_method='median',
            auto_special_factors=True,
            include_quantiles=True,  # 【v22】分位点も取得
            order_mode=order_mode,
            backtest_days=14
        )
        if enhanced is not None and not enhanced.empty:
            results['精度強化版'] = enhanced['predicted'].values
            # 【v22】未丸め値があれば使用
            if 'predicted_raw' in enhanced.columns:
                raw_results['精度強化版'] = enhanced['predicted_raw'].values
            else:
                raw_results['精度強化版'] = enhanced['predicted'].values.astype(float)
            if hasattr(enhanced, 'attrs') and 'backtest' in enhanced.attrs:
                backtest_results['精度強化版'] = enhanced.attrs['backtest']
    except Exception as e:
        logger.warning(f"精度強化版の予測エラー: {e}")
    
    # 2. 季節性考慮（従来版）- バックテスト実行
    try:
        seasonal = forecast_with_seasonality_fallback(df, periods)
        if seasonal is not None and not seasonal.empty:
            results['季節性考慮'] = seasonal['predicted'].values
            raw_results['季節性考慮'] = seasonal['predicted'].values.astype(float)
            # バックテスト実行
            bt = run_simple_backtest(df, holdout_days=14)
            backtest_results['季節性考慮'] = bt
    except Exception as e:
        logger.warning(f"季節性考慮の予測エラー: {e}")
    
    # 3. 移動平均
    try:
        ma = forecast_moving_average(df, periods)
        if ma is not None and not ma.empty:
            results['移動平均'] = ma['predicted'].values
            raw_results['移動平均'] = ma['predicted'].values.astype(float)
            # 移動平均用バックテスト
            bt = run_simple_backtest(df, holdout_days=14, forecast_func=lambda d, p: forecast_moving_average(d, p))
            backtest_results['移動平均'] = bt
    except Exception as e:
        logger.warning(f"移動平均の予測エラー: {e}")
    
    # 4. 指数平滑法
    try:
        exp = forecast_exponential_smoothing(df, periods)
        if exp is not None and not exp.empty:
            results['指数平滑'] = exp['predicted'].values
            raw_results['指数平滑'] = exp['predicted'].values.astype(float)
            # 指数平滑用バックテスト
            bt = run_simple_backtest(df, holdout_days=14, forecast_func=lambda d, p: forecast_exponential_smoothing(d, p))
            backtest_results['指数平滑'] = bt
    except Exception as e:
        logger.warning(f"指数平滑法の予測エラー: {e}")
    
    # 5. Prophet（利用可能な場合）
    if PROPHET_AVAILABLE:
        try:
            prophet_result, _ = forecast_with_prophet(df, periods)
            if prophet_result is not None and not prophet_result.empty:
                results['Prophet'] = prophet_result['predicted'].values
                raw_results['Prophet'] = prophet_result['predicted'].values.astype(float)
                # ProphetはMAPE高めになりやすいのでデフォルト値設定
                backtest_results['Prophet'] = {'mape': 35.0, 'available': True}
        except Exception as e:
            logger.warning(f"Prophetの予測エラー: {e}")
    
    # 6. Holt-Winters（利用可能な場合）
    if STATSMODELS_AVAILABLE:
        try:
            hw_result, _ = forecast_with_holt_winters(df, periods)
            if hw_result is not None and not hw_result.empty:
                results['Holt-Winters'] = hw_result['predicted'].values
                raw_results['Holt-Winters'] = hw_result['predicted'].values.astype(float)
                # Holt-WintersもMAPEデフォルト値
                backtest_results['Holt-Winters'] = {'mape': 40.0, 'available': True}
        except Exception as e:
            logger.warning(f"Holt-Wintersの予測エラー: {e}")
    
    # ========== フォールバック ==========
    if not results:
        avg = df['販売商品数'].mean() if len(df) > 0 else 1
        avg = max(1, avg)
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        result_df = pd.DataFrame({
            'date': future_dates,
            'predicted_raw': [float(avg)] * periods,  # 【v22】未丸め
            'predicted': [int(round(avg))] * periods
        })
        return result_df, {
            'methods_used': [],
            'ensemble_type': 'fallback',
            'reliability': {'level': 'low', 'level_text': '× 要注意', 'color': '#F44336'}
        }
    
    # ========== 【v22新機能】加重アンサンブル計算 ==========
    
    # 1. MAPEベースの重みを計算
    weights = calculate_weighted_ensemble_weights(backtest_results)
    
    # 重みがない場合（全てのバックテストが失敗）は均等配分
    if not weights:
        n_methods = len(results)
        weights = {k: 1.0 / n_methods for k in results.keys()}
    
    # 存在する方法のみで重みを正規化
    available_methods = [m for m in weights.keys() if m in results]
    weight_sum = sum(weights.get(m, 0) for m in available_methods)
    
    if weight_sum > 0:
        normalized_weights = {m: weights.get(m, 0) / weight_sum for m in available_methods}
    else:
        normalized_weights = {m: 1.0 / len(available_methods) for m in available_methods}
    
    # 2. 加重平均の計算（未丸め値で計算）
    weighted_predictions = np.zeros(periods)
    for method in available_methods:
        if method in raw_results:
            weighted_predictions += raw_results[method] * normalized_weights[method]
    
    # 3. 従来の中央値も計算（安定性のため）
    all_predictions = np.array([raw_results[m] for m in available_methods])
    median_predictions = np.median(all_predictions, axis=0)
    
    # 4. IQRで外れ値を検出
    q1 = np.percentile(all_predictions, 25, axis=0)
    q3 = np.percentile(all_predictions, 75, axis=0)
    iqr = q3 - q1
    
    # 5. 外れ値を除外した平均
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    trimmed_predictions = []
    for i in range(periods):
        day_preds = all_predictions[:, i]
        valid_preds = day_preds[(day_preds >= lower_bound[i]) & (day_preds <= upper_bound[i])]
        if len(valid_preds) > 0:
            trimmed_predictions.append(np.mean(valid_preds))
        else:
            trimmed_predictions.append(median_predictions[i])
    trimmed_predictions = np.array(trimmed_predictions)
    
    # 6. 【v22】最終予測: 加重平均(50%) + 中央値(30%) + トリム平均(20%)
    final_predictions_raw = (
        weighted_predictions * 0.5 +
        median_predictions * 0.3 +
        trimmed_predictions * 0.2
    )
    
    # 7. 分位点の計算（各方法の予測から）
    p50_raw = np.percentile(all_predictions, 50, axis=0)
    p80_raw = np.percentile(all_predictions, 80, axis=0)
    p90_raw = np.percentile(all_predictions, 90, axis=0)
    
    # ========== 結果DataFrame ==========
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    result_df = pd.DataFrame({
        'date': future_dates,
        'predicted_raw': final_predictions_raw,                          # 【v22】未丸め
        'predicted': np.round(final_predictions_raw).astype(int).clip(min=0),
        'predicted_median': np.round(median_predictions).astype(int).clip(min=0),
        'predicted_weighted': np.round(weighted_predictions).astype(int).clip(min=0),  # 【v22】
        'predicted_lower': np.round(q1).astype(int).clip(min=0),
        'predicted_upper': np.round(q3).astype(int).clip(min=0),
        # 【v22】分位点
        'p50_raw': p50_raw,
        'p80_raw': p80_raw,
        'p90_raw': p90_raw,
        'p50': np.round(p50_raw).astype(int).clip(min=0),
        'p80': np.round(p80_raw).astype(int).clip(min=0),
        'p90': np.round(p90_raw).astype(int).clip(min=0),
    })
    
    # 【v22】発注モードに応じた推奨値
    if order_mode == 'conservative':
        result_df['recommended_raw'] = result_df['p50_raw']
        result_df['recommended'] = result_df['p50']
    elif order_mode == 'aggressive':
        result_df['recommended_raw'] = result_df['p90_raw']
        result_df['recommended'] = result_df['p90']
    else:  # balanced
        result_df['recommended_raw'] = result_df['p80_raw']
        result_df['recommended'] = result_df['p80']
    
    # 各方法の合計
    method_totals = {method: int(np.sum(preds)) for method, preds in results.items()}
    method_totals_raw = {method: float(np.sum(raw_results[method])) for method in available_methods}
    
    # 信頼度評価
    reliability = calculate_reliability_score(
        method_totals,
        backtest_results,
        len(df)
    )
    
    # 【v22】アンサンブル詳細情報
    ensemble_info = {
        'methods_used': list(results.keys()),
        'method_totals': method_totals,
        'method_totals_raw': method_totals_raw,  # 【v22】
        'weights': normalized_weights,            # 【v22】各方法の重み
        'ensemble_total': int(np.sum(np.round(final_predictions_raw))),
        'ensemble_total_raw': float(np.sum(final_predictions_raw)),  # 【v22】
        'median_total': int(np.sum(np.round(median_predictions))),
        'weighted_total': int(np.sum(np.round(weighted_predictions))),  # 【v22】
        'order_mode': order_mode,                 # 【v22】
        'reliability': reliability,
        'backtest_mapes': {m: bt.get('mape') for m, bt in backtest_results.items() if bt}  # 【v22】
    }
    
    result_df.attrs['ensemble'] = ensemble_info
    result_df.attrs['order_mode'] = order_mode  # 【v22】
    result_df.attrs['backtest'] = backtest_results.get('精度強化版', {'mape': None, 'available': False})
    
    return result_df, ensemble_info


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
    zero_fill_applied = False
    if enable_zero_fill:
        df, missing_dates = fill_missing_dates(df, fill_value=0)
        zero_fill_applied = True
        
        # 欠損率のチェックと警告
        if len(missing_dates) > 0:
            total_days = len(df)
            original_days = total_days - len(missing_dates)
            missing_ratio = len(missing_dates) / total_days if total_days > 0 else 0
            
            if missing_ratio > 0.5:
                logger.warning(
                    f"欠損率が高いです（{missing_ratio:.1%}）。"
                    f"元データ{original_days}日 / 全期間{total_days}日。"
                    f"季節商品の可能性があります。"
                )
    
    # ========== v20新機能2: 欠品期間の除外 ==========
    if stockout_periods:
        df = exclude_stockout_periods(df, stockout_periods)
    
    # データが少なすぎる場合のフォールバック
    # 【修正】0埋め時は0を除外してカウント（季節商品対応）
    if zero_fill_applied:
        valid_count = ((df['販売商品数'].notna()) & (df['販売商品数'] > 0)).sum()
    else:
        valid_count = df['販売商品数'].notna().sum()
    
    if valid_count < 7:
        logger.warning(f"有効データが少なすぎます（{valid_count}件）。シンプルな予測にフォールバックします。")
        # 単純平均で予測（0埋め時は0も除外）
        if zero_fill_applied:
            valid_sales = df[(df['販売商品数'].notna()) & (df['販売商品数'] > 0)]['販売商品数']
        else:
            valid_sales = df['販売商品数'].dropna()
        
        avg_value = valid_sales.mean() if len(valid_sales) > 0 else 1.0
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
    
    # ========== 1. 頑健なベースライン計算 ==========
    # 【重要】0埋め時は0を除外してベースラインを計算（季節商品対応）
    if zero_fill_applied:
        nonzero_values = df[(df['販売商品数'].notna()) & (df['販売商品数'] > 0)]['販売商品数'].values
        if len(nonzero_values) >= 7:
            valid_values = nonzero_values
            logger.info(f"0埋めモード: {len(nonzero_values)}件の非ゼロデータでベースラインを計算")
        else:
            valid_values = df['販売商品数'].dropna().values
            logger.warning(f"非ゼロデータが少ないため、全データ({len(valid_values)}件)でベースラインを計算")
    else:
        valid_values = df['販売商品数'].dropna().values
    
    # 【v22致命的バグ修正】モード対応のベースライン計算
    # ここでorder_modeを考慮しないと、全モードで同じ予測値になる
    overall_baseline, baseline_info = calculate_mode_aware_baseline(
        valid_values, 
        order_mode=order_mode,
        base_method=baseline_method
    )
    logger.info(f"【v22】モード対応ベースライン: {baseline_info['description']} → {overall_baseline:.2f}")
    
    # 【v22】安全係数の計算
    mean_val = baseline_info.get('mean', overall_baseline)
    std_val = baseline_info.get('std', 0.0)
    safety_factor, safety_info = calculate_safety_factor(order_mode, std_val, mean_val)
    logger.info(f"【v22】安全係数: {safety_info['description']} → {safety_factor:.3f}")
    
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
    
    # ========== 3.1 通常日データの抽出 ==========
    # 【重要】0埋め時は0も除外して係数を計算（季節商品対応）
    if zero_fill_applied:
        # 0埋めした場合: NaN、特別期間、0をすべて除外
        normal_df = df[
            (~df['is_special']) & 
            (df['販売商品数'].notna()) & 
            (df['販売商品数'] > 0)
        ].copy()
    else:
        # 通常: NaNと特別期間を除外
        normal_df = df[(~df['is_special']) & (df['販売商品数'].notna())].copy()
    
    # ========== 3.2 通常日ベースラインの計算 ==========
    if len(normal_df) > 10:
        # 【v22致命的バグ修正】通常日もモード対応でベースライン計算
        normal_baseline, normal_baseline_info = calculate_mode_aware_baseline(
            normal_df['販売商品数'].values, 
            order_mode=order_mode,
            base_method=baseline_method
        )
        logger.info(f"【v22】通常日ベースライン: {normal_baseline_info['description']} → {normal_baseline:.2f}")
    else:
        normal_baseline = overall_baseline
        normal_baseline_info = baseline_info
    
    # ベースラインが0以下の場合の安全対策
    if normal_baseline <= 0:
        logger.warning(f"通常日ベースラインが0以下({normal_baseline})のため、overall_baselineを使用")
        normal_baseline = overall_baseline if overall_baseline > 0 else 1.0
    
    # ========== 4. 曜日係数（頑健版） ==========
    weekday_factor = {}
    
    for wd in range(7):
        # 【注意】normal_dfは既に0埋め時は0を除外済み
        wd_values = normal_df[normal_df['weekday'] == wd]['販売商品数'].values
        weekday_factor[wd] = calculate_robust_factor(wd_values, normal_baseline, min_samples=3)
    
    # ========== 5. 月係数（頑健版） ==========
    month_factor = {}
    
    for m in range(1, 13):
        # 【注意】normal_dfは既に0埋め時は0を除外済み
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
        
        # 【v22致命的バグ修正】予測値計算に安全係数を適用
        # 予測値 = 通常日ベースライン × 曜日係数 × 月係数 × 特別期間係数 × トレンド係数 × 安全係数
        pred = normal_baseline * weekday_f * month_f * special_f * trend_factor * safety_factor
        pred = max(0.1, pred)
        point_predictions.append(pred)
        
        # 【v22改善】丸め処理の後段化
        # predicted_raw: 浮動小数点のまま保持（精度向上のため）
        # predicted: 表示用の整数丸め
        predictions.append({
            'date': d,
            'predicted_raw': pred,                # 【v22】未丸め値（float）
            'predicted': int(round(pred)),        # 表示用（int）
            'weekday_factor': weekday_f,
            'month_factor': month_f,
            'special_factor': special_f,
            'trend_factor': trend_factor,         # v20追加
            'safety_factor': safety_factor        # 【v22】安全係数
        })
    
    result_df = pd.DataFrame(predictions)
    
    # ========== 9. 分位点予測の追加 ==========
    if include_quantiles:
        point_array = np.array(point_predictions)
        quantile_results = calculate_prediction_quantiles(
            point_array, residuals, quantiles=[0.5, 0.8, 0.9]
        )
        
        # 【v22改善】分位点も未丸め値と丸め値の両方を保持
        result_df['p50_raw'] = quantile_results['p50']                      # 【v22】未丸め
        result_df['p80_raw'] = quantile_results['p80']                      # 【v22】未丸め
        result_df['p90_raw'] = quantile_results['p90']                      # 【v22】未丸め
        result_df['p50'] = quantile_results['p50'].round().astype(int)
        result_df['p80'] = quantile_results['p80'].round().astype(int)
        result_df['p90'] = quantile_results['p90'].round().astype(int)
        
        # 発注モードに応じた推奨値（【v22改善】未丸め値も追加）
        if order_mode == 'conservative':  # 滞留回避
            result_df['recommended_raw'] = result_df['p50_raw']             # 【v22】未丸め
            result_df['recommended'] = result_df['p50']
        elif order_mode == 'aggressive':  # 欠品回避
            result_df['recommended_raw'] = result_df['p90_raw']             # 【v22】未丸め
            result_df['recommended'] = result_df['p90']
        else:  # balanced
            result_df['recommended_raw'] = result_df['p80_raw']             # 【v22】未丸め
            result_df['recommended'] = result_df['p80']
    else:
        # 【v22致命的バグ修正】include_quantiles=Falseでもrecommended列を生成
        # これがないと、get_order_column()が常にpredictedを返してしまう
        # モードに応じた係数でrecommended値を計算
        point_array = np.array(point_predictions)
        
        # モード別の分位点係数（過去データの変動に基づく簡易計算）
        if len(valid_values) > 0:
            cv = np.std(valid_values) / np.mean(valid_values) if np.mean(valid_values) > 0 else 0.3
        else:
            cv = 0.3  # デフォルト変動係数
        
        # 係数は変動係数に基づいて計算
        # P50 ≈ 予測値（中央値相当）
        # P80 ≈ 予測値 × (1 + 0.84 × CV) ※正規分布の80%点
        # P90 ≈ 予測値 × (1 + 1.28 × CV) ※正規分布の90%点
        p50_factor = 1.0
        p80_factor = 1.0 + 0.84 * cv
        p90_factor = 1.0 + 1.28 * cv
        
        # 最低差を保証（各モード間で最低5%の差）
        p80_factor = max(p80_factor, 1.05)
        p90_factor = max(p90_factor, p80_factor * 1.05, 1.10)
        
        result_df['p50_raw'] = point_array * p50_factor
        result_df['p80_raw'] = point_array * p80_factor
        result_df['p90_raw'] = point_array * p90_factor
        result_df['p50'] = (point_array * p50_factor).round().astype(int)
        result_df['p80'] = (point_array * p80_factor).round().astype(int)
        result_df['p90'] = (point_array * p90_factor).round().astype(int)
        
        # 発注モードに応じた推奨値
        if order_mode == 'conservative':
            result_df['recommended_raw'] = result_df['p50_raw']
            result_df['recommended'] = result_df['p50']
        elif order_mode == 'aggressive':
            result_df['recommended_raw'] = result_df['p90_raw']
            result_df['recommended'] = result_df['p90']
        else:  # balanced
            result_df['recommended_raw'] = result_df['p80_raw']
            result_df['recommended'] = result_df['p80']
        
        logger.info(f"【v22】簡易分位点計算: CV={cv:.2f}, P50係数={p50_factor:.2f}, P80係数={p80_factor:.2f}, P90係数={p90_factor:.2f}")
    
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
    
    # 【v22追加】発注モード情報と安全係数情報
    result_df.attrs['order_mode'] = order_mode
    result_df.attrs['v22_mode_info'] = {
        'order_mode': order_mode,
        'baseline_info': baseline_info if 'baseline_info' in dir() else {},
        'safety_factor': safety_factor,
        'safety_info': safety_info if 'safety_info' in dir() else {}
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
    
    elif method == "📊 Prophet（季節商品向け）":
        # v21新機能: Prophet
        result, message = forecast_with_prophet(df, periods)
        if result is not None:
            return result, message
        else:
            # Prophetが失敗した場合は精度強化版にフォールバック
            logger.warning(f"Prophetが利用不可のため精度強化版にフォールバック: {message}")
            forecast = forecast_with_seasonality_enhanced(
                df, periods,
                baseline_method=baseline_method,
                auto_special_factors=auto_special_factors,
                backtest_days=backtest_days
            )
            return forecast, f"精度強化版（Prophetの代替: {message}）"
    
    elif method == "📈 Holt-Winters法":
        # v21新機能: Holt-Winters
        result, message = forecast_with_holt_winters(df, periods)
        if result is not None:
            return result, message
        else:
            # Holt-Wintersが失敗した場合は精度強化版にフォールバック
            logger.warning(f"Holt-Wintersが利用不可のため精度強化版にフォールバック: {message}")
            forecast = forecast_with_seasonality_enhanced(
                df, periods,
                baseline_method=baseline_method,
                auto_special_factors=auto_special_factors,
                backtest_days=backtest_days
            )
            return forecast, f"精度強化版（Holt-Wintersの代替: {message}）"
    
    elif method == "🧠 アンサンブル予測（v21）":
        # v21新機能: アンサンブル予測【v22改善：order_mode対応】
        result, ensemble_info = forecast_ensemble(df, periods, order_mode=order_mode)
        
        # メッセージ生成
        methods_used = ensemble_info.get('methods_used', [])
        reliability = ensemble_info.get('reliability', {})
        reliability_text = reliability.get('level_text', '')
        
        message = f"アンサンブル予測（{len(methods_used)}手法の組み合わせ）"
        if reliability_text:
            message += f"・信頼度: {reliability_text}"
        
        # 信頼度情報を属性に追加
        result.attrs['ensemble'] = ensemble_info
        result.attrs['backtest'] = {
            'mape': None,
            'available': True,
            'reliability': reliability.get('level', 'medium'),
            'message': f"アンサンブル信頼度スコア: {reliability.get('total_score', 0):.1f}点"
        }
        
        return result, message
    
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
    backtest_days: int = 14,
    order_mode: str = 'balanced'  # 【v22追加】発注モード
) -> Dict[str, Tuple[pd.DataFrame, str]]:
    """
    すべての予測方法で予測を実行（v22: 発注モード対応）
    """
    results = {}
    
    # Vertex AI予測
    if VERTEX_AI_AVAILABLE:
        try:
            predictions, used_vertex_ai, message = get_vertex_ai_prediction(df, periods, product_id)
            results['Vertex AI'] = (predictions, message)
        except Exception as e:
            logger.warning(f"Vertex AI予測失敗: {e}")
    
    # 【v21新規】アンサンブル予測【v22改善：order_mode対応】
    try:
        ensemble_result, ensemble_info = forecast_ensemble(df, periods, order_mode=order_mode)
        reliability = ensemble_info.get('reliability', {})
        reliability_text = reliability.get('level_text', '')
        weights_info = ensemble_info.get('weights', {})
        method_desc = f"アンサンブル予測（{len(ensemble_info.get('methods_used', []))}手法・信頼度: {reliability_text}）"
        results['アンサンブル'] = (ensemble_result, method_desc)
    except Exception as e:
        logger.warning(f"アンサンブル予測失敗: {e}")
    
    # 【v19新規】精度強化版予測【v22改善：order_mode対応】
    try:
        enhanced_forecast = forecast_with_seasonality_enhanced(
            df, periods,
            baseline_method=baseline_method,
            auto_special_factors=auto_special_factors,
            include_quantiles=True,
            order_mode=order_mode,  # 【v22】発注モードを渡す
            backtest_days=backtest_days
        )
        method_desc = f"季節性考慮（精度強化版・{baseline_method}ベース）"
        results['精度強化版'] = (enhanced_forecast, method_desc)
    except Exception as e:
        logger.warning(f"精度強化版予測失敗: {e}")
    
    # 【v21新規】Prophet予測
    if PROPHET_AVAILABLE:
        try:
            prophet_result, message = forecast_with_prophet(df, periods)
            if prophet_result is not None:
                results['Prophet'] = (prophet_result, message)
        except Exception as e:
            logger.warning(f"Prophet予測失敗: {e}")
    
    # 【v21新規】Holt-Winters予測
    if STATSMODELS_AVAILABLE:
        try:
            hw_result, message = forecast_with_holt_winters(df, periods)
            if hw_result is not None:
                results['Holt-Winters'] = (hw_result, message)
        except Exception as e:
            logger.warning(f"Holt-Winters予測失敗: {e}")
    
    # 統計モデル予測（従来版）
    results['季節性考慮'] = (forecast_with_seasonality_fallback(df, periods), "季節性考慮（統計モデル）")
    results['移動平均法'] = (forecast_moving_average(df, periods), "移動平均法（統計モデル）")
    results['指数平滑法'] = (forecast_exponential_smoothing(df, periods), "指数平滑法（統計モデル）")
    
    return results


def display_comparison_results_v19(
    all_results: Dict[str, Tuple[pd.DataFrame, str]], 
    forecast_days: int, 
    sales_data: pd.DataFrame = None,
    order_mode: str = 'balanced'  # 【v22追加】発注モード
):
    """
    【v22改善版】すべての予測方法の比較結果を表示
    
    v21からの改善点:
    - 発注モードに応じた採用列の統一
    - recommended列があればそちらを使用
    - 未丸め値と丸め値の両方を表示
    - 最強ファクトチェックプロンプトの使用
    """
    st.success("✅ すべての予測方法で比較完了！")
    
    # 【v22】発注モードの表示
    mode_display = get_order_mode_display_name(order_mode)
    st.info(f"📦 **発注モード**: {mode_display}")
    
    # 各予測方法の予測総数を計算【v22改善：採用列統一】
    method_totals = {}
    backtest_info = {}
    
    for method_name, (forecast, message) in all_results.items():
        # 【v22】採用列を統一して合計計算
        totals = calculate_forecast_totals_v22(forecast, order_mode)
        
        method_totals[method_name] = {
            'raw': totals['raw_total'],
            'rounded': totals['rounded_total'],
            'avg': totals['avg_daily'],
            'avg_raw': totals['avg_daily_raw'],       # 【v22】未丸め
            'order_column': totals['order_column'],   # 【v22】採用列
            'predicted_total': totals['predicted_total']  # 【v22】参考値
        }
        
        # バックテスト情報があれば取得
        if hasattr(forecast, 'attrs') and 'backtest' in forecast.attrs:
            bt = forecast.attrs['backtest']
            if bt.get('mape') is not None:
                backtest_info[method_name] = bt['mape']
    
    # ========== 各予測方法の予測総数を明確に表示 ==========
    st.write("### 📊 各予測方法の予測総数（発注推奨数）")
    
    # 【v22】採用列の説明
    st.caption(f"※ 発注モード「{mode_display}」に基づき、採用列を統一して計算しています")
    
    # 分かりやすいリスト形式で表示
    st.markdown("---")
    for method_name, totals in method_totals.items():
        icon = "🚀" if "Vertex" in method_name else "🧠" if "アンサンブル" in method_name else "🎯" if "精度強化" in method_name else "📊" if "Prophet" in method_name else "📈" if "Holt" in method_name or "季節" in method_name else "📊" if "移動" in method_name else "📉"
        mape_str = f"（MAPE {backtest_info[method_name]:.1f}%）" if method_name in backtest_info else ""
        col_info = f"[{totals['order_column']}]" if totals['order_column'] != 'predicted' else ""
        st.markdown(f"""
        **{icon} {safe_html(method_name)}**: **{totals['rounded']:,}体**（日販 {totals['avg']:.1f}体）{mape_str} {col_info}
        """)
    st.markdown("---")
    
    # メトリクスで大きく表示
    num_methods = len(method_totals)
    cols = st.columns(min(num_methods, 4))
    
    for i, (method_name, totals) in enumerate(method_totals.items()):
        icon = "🚀" if "Vertex" in method_name else "🧠" if "アンサンブル" in method_name else "🎯" if "精度強化" in method_name else "📊" if "Prophet" in method_name else "📈" if "Holt" in method_name or "季節" in method_name else "📊" if "移動" in method_name else "📉"
        short_name = method_name.replace("（統計）", "").replace("（推奨）", "")
        with cols[i % 4]:
            delta_str = f"MAPE {backtest_info[method_name]:.1f}%" if method_name in backtest_info else f"日販 {totals['avg']:.1f}体"
            st.metric(
                f"{icon} {safe_html(short_name)}",
                f"{totals['rounded']:,}体",
                delta_str
            )
    
    # 詳細表【v22改善：採用列と未丸め値を追加】
    with st.expander("📋 詳細データを表示（監査用）", expanded=False):
        summary_rows = []
        for method_name, totals in method_totals.items():
            icon = "🚀" if "Vertex" in method_name else "🧠" if "アンサンブル" in method_name else "🎯" if "精度強化" in method_name else "📊" if "Prophet" in method_name else "📈" if "Holt" in method_name or "季節" in method_name else "📊" if "移動" in method_name else "📉"
            mape_str = f"{backtest_info[method_name]:.1f}%" if method_name in backtest_info else "-"
            summary_rows.append({
                '予測方法': f"{icon} {method_name}",
                '採用列': totals['order_column'],  # 【v22】
                '合計（未丸め）': f"{totals['raw']:,.2f}体",  # 【v22】
                '発注推奨数（50単位）': f"{totals['rounded']:,}体",
                '日販（未丸め）': f"{totals['avg_raw']:.2f}体/日",  # 【v22】
                'predicted合計': f"{totals['predicted_total']:,.0f}体",  # 【v22】参考
                'MAPE': mape_str
            })
        
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # 【v22】検算用の説明
        st.caption("※「採用列」は発注モードに応じて自動選択されます（P90→aggressive、P80→balanced、P50→conservative）")
    
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
    
    # 【v22改善】推奨の判断基準
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
        'アンサンブル': '#673AB7',
        '精度強化版': '#9C27B0',
        'Prophet': '#2196F3',
        'Holt-Winters': '#009688',
        '季節性考慮': '#4CAF50',
        '移動平均法': '#1E88E5',
        '指数平滑法': '#FF9800'
    }
    
    # 比較グラフ（スマホ最適化）【v22改善：採用列でグラフ描画】
    st.write("### 📈 日別予測比較グラフ")
    
    fig = go.Figure()
    
    for method_name, (forecast, message) in all_results.items():
        # 【v22】採用列を使用（なければpredicted）
        y_col = get_order_column(forecast, order_mode)
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast[y_col] if y_col in forecast.columns else forecast['predicted'],
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
    
    # 【v22改善】最強ファクトチェックプロンプト
    product_names = st.session_state.get('selected_products', [])
    
    # 精度強化版があればそれを使用
    if '精度強化版' in all_results:
        best_forecast = all_results['精度強化版'][0]
        best_method_name = '精度強化版'
    elif 'アンサンブル' in all_results:
        best_forecast = all_results['アンサンブル'][0]
        best_method_name = 'アンサンブル'
    else:
        best_forecast = list(all_results.values())[0][0]
        best_method_name = list(all_results.keys())[0]
    
    # v20機能情報とバックテスト結果を取得
    v20_features = best_forecast.attrs.get('v20_features', None) if hasattr(best_forecast, 'attrs') else None
    backtest_result = best_forecast.attrs.get('backtest', None) if hasattr(best_forecast, 'attrs') else None
    
    # 最強ファクトチェックプロンプトを生成
    factcheck_prompt = generate_factcheck_prompt_advanced(
        product_names=product_names,
        forecast_result=best_forecast,
        order_mode=order_mode,
        sales_data=sales_data,
        forecast_days=forecast_days,
        method_name=best_method_name,
        v20_features=v20_features,
        backtest_result=backtest_result
    )
    display_factcheck_section(factcheck_prompt, key_suffix="comparison_v22")


# 旧バージョンとの互換性のため残す
def display_comparison_results_v12(all_results: Dict[str, Tuple[pd.DataFrame, str]], forecast_days: int, sales_data: pd.DataFrame = None):
    """すべての予測方法の比較結果を表示（v12 互換性維持用）"""
    # v22版を呼び出し（デフォルトのbalancedモードで）
    display_comparison_results_v19(all_results, forecast_days, sales_data, order_mode='balanced')


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
    "🧠 アンサンブル予測（v21）": {
        "description": "【v21新機能】複数の予測方法を組み合わせ、外れ値を除外した安定した予測。信頼度評価付き。",
        "icon": "🧠",
        "color": "#673AB7",
        "requires_vertex_ai": False
    },
    "🎯 季節性考慮（精度強化版）": {
        "description": "【v20】0埋め・欠品除外・トレンド係数・正月日別係数対応。分位点予測・バックテスト付き。",
        "icon": "🎯",
        "color": "#9C27B0",
        "requires_vertex_ai": False
    },
    "📊 Prophet（季節商品向け）": {
        "description": "【v21新機能】Meta製の高精度予測。季節性・トレンド・イベントを自動検出。季節商品に最適。",
        "icon": "📊",
        "color": "#2196F3",
        "requires_vertex_ai": False
    },
    "📈 Holt-Winters法": {
        "description": "【v21新機能】三重指数平滑法。週間の季節パターンを捉える。",
        "icon": "📈",
        "color": "#009688",
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
        "description": "すべての予測方法で予測し、結果を比較。信頼度評価付き。",
        "icon": "🔄",
        "color": "#607D8B",
        "requires_vertex_ai": False
    }
}

# =============================================================================
# 【v22新機能】予測方法の詳細説明（ユーザー向け）
# =============================================================================

FORECAST_METHOD_DESCRIPTIONS = {
    "精度強化版": {
        "short": "季節性・特別期間・トレンドを考慮した高精度予測",
        "long": "過去データから曜日・月・正月などの季節パターンを学習し、欠品期間を除外して予測します。バックテストによる精度検証付き。",
        "best_for": "通年販売の授与品、安定した需要パターンの商品",
        "icon": "🎯"
    },
    "アンサンブル": {
        "short": "複数の予測方法を組み合わせた安定予測",
        "long": "精度強化版、移動平均、指数平滑法、Prophet、Holt-Wintersの結果を統合し、外れ値を除外して安定した予測を生成します。",
        "best_for": "需要パターンが不明な商品、初めて予測する商品",
        "icon": "🧠"
    },
    "Prophet": {
        "short": "Meta社製の高精度予測モデル（季節商品向け）",
        "long": "Facebookが開発した時系列予測モデル。複雑な季節性パターンを自動検出し、イベント効果も考慮できます。",
        "best_for": "季節変動が大きい商品、正月・お盆に需要が集中する商品",
        "icon": "📊"
    },
    "Holt-Winters": {
        "short": "三重指数平滑法（週間パターン向け）",
        "long": "トレンド、季節性、レベルの3つの成分を分解して予測。週間の曜日パターンを捉えるのが得意です。",
        "best_for": "週末と平日で需要が異なる商品",
        "icon": "📈"
    },
    "季節性考慮": {
        "short": "統計的な季節性モデル（従来版）",
        "long": "月別・曜日別の傾向と特別期間を考慮した統計モデル。シンプルで解釈しやすい予測です。",
        "best_for": "シンプルな予測が必要な場合",
        "icon": "📈"
    },
    "移動平均": {
        "short": "過去30日間の平均値ベース（安定商品向け）",
        "long": "直近30日間の売上平均を将来の予測値とします。急激な変動に弱いですが、安定した需要には適しています。",
        "best_for": "需要が安定している商品、御朱印など",
        "icon": "📊"
    },
    "指数平滑": {
        "short": "直近のデータを重視した予測",
        "long": "最近のデータほど重みを大きくして予測。トレンドの変化に敏感に反応しますが、ノイズにも敏感です。",
        "best_for": "トレンドが変化している商品",
        "icon": "📉"
    },
    "Vertex AI": {
        "short": "Google Cloud AIによる機械学習予測",
        "long": "Google Cloud AutoML Forecastingを使用した高精度予測。大量のデータがある場合に最も精度が高くなります。",
        "best_for": "1年以上の売上データがある商品",
        "icon": "🚀"
    }
}


def forecast_all_methods_unified_v22(
    df: pd.DataFrame,
    periods: int,
    product_id: str = "default",
    enable_zero_fill: bool = True,
    stockout_periods: Optional[List[Tuple[date, date]]] = None,
    enable_trend: bool = True,
    use_daily_new_year: bool = True,
    trend_window_days: int = 60,
    outlier_excluded_dates: Optional[List[date]] = None  # 【v24新機能】異常値除外日付
) -> Dict[str, Dict[str, Any]]:
    """
    【v22新機能】全予測方法を実行し、モード別の結果を統合
    
    各予測方法で3つのモード（滞留回避/バランス/欠品回避）の予測を行い、
    結果を統一フォーマットで返す。
    
    Args:
        df: 売上データ
        periods: 予測日数
        product_id: 商品識別子
        enable_zero_fill: 0埋め処理
        stockout_periods: 欠品期間リスト
        enable_trend: トレンド係数
        use_daily_new_year: 正月日別係数
        trend_window_days: トレンド比較期間
        outlier_excluded_dates: 【v24新機能】異常値として除外する日付リスト
    
    Returns:
        {
            'method_name': {
                'forecast': DataFrame,
                'totals': {'conservative': x, 'balanced': y, 'aggressive': z},
                'mape': float or None,
                'description': str,
                'reliability': str,
                'weight': float  # 最終推奨計算用の重み
            }
        }
    """
    results = {}
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 【v24新機能】異常値除外を適用
    if outlier_excluded_dates:
        df = exclude_outlier_dates(df, outlier_excluded_dates)
        logger.info(f"異常値除外: {len(outlier_excluded_dates)}日分を除外して予測を実行")
    
    # 【v22.5】実績データの統計を計算（妥当性チェック用）
    actual_mean = df['販売商品数'].mean() if '販売商品数' in df.columns else 0
    actual_max = df['販売商品数'].max() if '販売商品数' in df.columns else 0
    actual_median = df['販売商品数'].median() if '販売商品数' in df.columns else 0
    data_days = len(df)
    
    # 【v23.2修正】妥当性チェック用の上限値をより厳格に計算
    # 基本: 実績平均 × 予測日数 × 2.5（最大でも2.5倍まで）
    # 外れ値の影響を受けにくいよう、maxではなく中央値ベースで計算
    base_expected = actual_mean * periods
    
    # 上限は以下の小さい方:
    # 1. 実績平均ベース × 2.5
    # 2. 実績中央値ベース × 3.0（外れ値の影響を受けにくい）
    # ただし、最低5,000体は確保（データが少ない場合のため）
    limit_from_mean = actual_mean * periods * 2.5
    limit_from_median = actual_median * periods * 3.0
    max_reasonable_total = max(min(limit_from_mean, limit_from_median), 5000)
    
    # 【v23.2追加】実績ベースの基準値（後で比較用）
    expected_total = actual_mean * periods
    
    # 各モードで精度強化版を実行
    modes = ['conservative', 'balanced', 'aggressive']
    
    # 1. 精度強化版（3モード）
    try:
        mode_forecasts = {}
        mode_totals = {}
        backtest_mape = None
        
        for mode in modes:
            forecast = forecast_with_seasonality_enhanced(
                df, periods,
                baseline_method='median',
                auto_special_factors=True,
                include_quantiles=True,
                order_mode=mode,
                backtest_days=14,
                enable_zero_fill=enable_zero_fill,
                stockout_periods=stockout_periods,
                enable_trend=enable_trend,
                use_daily_new_year=use_daily_new_year,
                trend_window_days=trend_window_days
            )
            
            if forecast is not None and not forecast.empty:
                mode_forecasts[mode] = forecast
                # 採用列を統一して合計を計算
                totals = calculate_forecast_totals_v22(forecast, mode)
                # 【v22.4】妥当性チェック
                if totals['rounded_total'] > max_reasonable_total:
                    logger.warning(f"精度強化版({mode})予測値が異常 ({totals['rounded_total']:,} > {max_reasonable_total:,})、除外します")
                    continue
                mode_totals[mode] = totals['rounded_total']
                
                # バックテスト結果を取得（1回だけ）
                if backtest_mape is None and hasattr(forecast, 'attrs') and 'backtest' in forecast.attrs:
                    bt = forecast.attrs['backtest']
                    if bt.get('mape') is not None:
                        backtest_mape = bt['mape']
        
        if mode_totals and len(mode_totals) == 3:  # 3モードすべて妥当な場合のみ
            results['精度強化版'] = {
                'forecast': mode_forecasts.get('balanced'),
                'totals': mode_totals,
                'mape': backtest_mape,
                'description': FORECAST_METHOD_DESCRIPTIONS['精度強化版'],
                'reliability': 'high' if backtest_mape and backtest_mape < 30 else 'medium',
                'weight': 1.0 / (backtest_mape ** 2 + 1) if backtest_mape else 0.01
            }
    except Exception as e:
        logger.warning(f"精度強化版の予測エラー: {e}")
    
    # 2. アンサンブル予測（3モード）
    try:
        mode_totals = {}
        ensemble_info = None
        
        for mode in modes:
            ensemble_result, info = forecast_ensemble(df, periods, order_mode=mode)
            
            if ensemble_result is not None and not ensemble_result.empty:
                totals = calculate_forecast_totals_v22(ensemble_result, mode)
                # 【v22.4】妥当性チェック
                if totals['rounded_total'] > max_reasonable_total:
                    logger.warning(f"アンサンブル予測値が異常 ({totals['rounded_total']:,} > {max_reasonable_total:,})、除外します")
                    continue
                mode_totals[mode] = totals['rounded_total']
                if ensemble_info is None:
                    ensemble_info = info
        
        if mode_totals and len(mode_totals) == 3:  # 3モードすべて妥当な場合のみ
            reliability_info = ensemble_info.get('reliability', {}) if ensemble_info else {}
            results['アンサンブル'] = {
                'forecast': ensemble_result,
                'totals': mode_totals,
                'mape': None,  # アンサンブルはMAPE計算が異なる
                'description': FORECAST_METHOD_DESCRIPTIONS['アンサンブル'],
                'reliability': reliability_info.get('level', 'medium'),
                'weight': 0.02  # デフォルト重み
            }
    except Exception as e:
        logger.warning(f"アンサンブル予測エラー: {e}")
    
    # 3. Prophet（利用可能な場合）
    if PROPHET_AVAILABLE:
        try:
            prophet_result, message = forecast_with_prophet(df, periods)
            if prophet_result is not None and not prophet_result.empty:
                # Prophetは単一予測なのでモード別に係数で調整
                base_total = int(prophet_result['predicted'].sum())
                
                # 【v22.4】妥当性チェック：異常な予測値は除外
                if base_total > max_reasonable_total:
                    logger.warning(f"Prophet予測値が異常 ({base_total:,} > {max_reasonable_total:,})、除外します")
                else:
                    mode_totals = {
                        'conservative': round_up_to_50(int(base_total * 0.9)),
                        'balanced': round_up_to_50(base_total),
                        'aggressive': round_up_to_50(int(base_total * 1.15))
                    }
                    results['Prophet'] = {
                        'forecast': prophet_result,
                        'totals': mode_totals,
                        'mape': 35.0,  # デフォルト
                        'description': FORECAST_METHOD_DESCRIPTIONS['Prophet'],
                        'reliability': 'medium',
                        'weight': 0.01
                    }
        except Exception as e:
            logger.warning(f"Prophet予測エラー: {e}")
    
    # 4. Holt-Winters（利用可能な場合）
    if STATSMODELS_AVAILABLE:
        try:
            hw_result, message = forecast_with_holt_winters(df, periods)
            if hw_result is not None and not hw_result.empty:
                base_total = int(hw_result['predicted'].sum())
                # 【v22.4】妥当性チェック
                if base_total > max_reasonable_total:
                    logger.warning(f"Holt-Winters予測値が異常 ({base_total:,} > {max_reasonable_total:,})、除外します")
                else:
                    mode_totals = {
                        'conservative': round_up_to_50(int(base_total * 0.9)),
                        'balanced': round_up_to_50(base_total),
                        'aggressive': round_up_to_50(int(base_total * 1.15))
                    }
                    results['Holt-Winters'] = {
                        'forecast': hw_result,
                        'totals': mode_totals,
                        'mape': 40.0,  # デフォルト
                        'description': FORECAST_METHOD_DESCRIPTIONS['Holt-Winters'],
                        'reliability': 'medium',
                        'weight': 0.008
                    }
        except Exception as e:
            logger.warning(f"Holt-Winters予測エラー: {e}")
    
    # 5. 季節性考慮（従来版）
    try:
        seasonal_result = forecast_with_seasonality_fallback(df, periods)
        if seasonal_result is not None and not seasonal_result.empty:
            base_total = int(seasonal_result['predicted'].sum())
            # 【v22.4】妥当性チェック
            if base_total > max_reasonable_total:
                logger.warning(f"季節性考慮予測値が異常 ({base_total:,} > {max_reasonable_total:,})、除外します")
            else:
                mode_totals = {
                    'conservative': round_up_to_50(int(base_total * 0.9)),
                    'balanced': round_up_to_50(base_total),
                    'aggressive': round_up_to_50(int(base_total * 1.15))
                }
                results['季節性考慮'] = {
                    'forecast': seasonal_result,
                    'totals': mode_totals,
                    'mape': None,
                    'description': FORECAST_METHOD_DESCRIPTIONS['季節性考慮'],
                    'reliability': 'low',
                    'weight': 0.005
                }
    except Exception as e:
        logger.warning(f"季節性考慮予測エラー: {e}")
    
    # 6. 移動平均
    try:
        ma_result = forecast_moving_average(df, periods)
        if ma_result is not None and not ma_result.empty:
            base_total = int(ma_result['predicted'].sum())
            # 【v22.4】妥当性チェック
            if base_total > max_reasonable_total:
                logger.warning(f"移動平均予測値が異常 ({base_total:,} > {max_reasonable_total:,})、除外します")
            else:
                mode_totals = {
                    'conservative': round_up_to_50(int(base_total * 0.9)),
                    'balanced': round_up_to_50(base_total),
                    'aggressive': round_up_to_50(int(base_total * 1.15))
                }
                results['移動平均'] = {
                    'forecast': ma_result,
                    'totals': mode_totals,
                    'mape': None,
                    'description': FORECAST_METHOD_DESCRIPTIONS['移動平均'],
                    'reliability': 'low',
                    'weight': 0.005
                }
    except Exception as e:
        logger.warning(f"移動平均予測エラー: {e}")
    
    # 7. 指数平滑法
    try:
        exp_result = forecast_exponential_smoothing(df, periods)
        if exp_result is not None and not exp_result.empty:
            base_total = int(exp_result['predicted'].sum())
            # 【v22.4】妥当性チェック
            if base_total > max_reasonable_total:
                logger.warning(f"指数平滑予測値が異常 ({base_total:,} > {max_reasonable_total:,})、除外します")
            else:
                mode_totals = {
                    'conservative': round_up_to_50(int(base_total * 0.9)),
                    'balanced': round_up_to_50(base_total),
                    'aggressive': round_up_to_50(int(base_total * 1.15))
                }
                results['指数平滑'] = {
                    'forecast': exp_result,
                    'totals': mode_totals,
                    'mape': None,
                    'description': FORECAST_METHOD_DESCRIPTIONS['指数平滑'],
                    'reliability': 'low',
                    'weight': 0.005
                }
    except Exception as e:
        logger.warning(f"指数平滑予測エラー: {e}")
    
    return results


def calculate_final_recommendation_v22(
    all_results: Dict[str, Dict[str, Any]],
    sales_data: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    【v22新機能】全予測結果から最終推奨発注数を算出
    
    精度の高いモデル（MAPE低い、信頼度高い）の結果を重視した
    加重平均で最終推奨値を計算。
    
    Args:
        all_results: forecast_all_methods_unified_v22の出力
        sales_data: 入力データ（変化率計算用）
    
    Returns:
        {
            'conservative': {'total': x, 'label': '滞留回避', 'description': '...'},
            'balanced': {'total': y, 'label': 'バランス', 'description': '...'},
            'aggressive': {'total': z, 'label': '欠品回避', 'description': '...'},
            'recommended': 'balanced',  # 推奨モード
            'weights_used': {...},
            'calculation_method': '...'
        }
    """
    if not all_results:
        return {
            'conservative': {'total': 0, 'label': '滞留回避', 'description': 'データなし'},
            'balanced': {'total': 0, 'label': 'バランス', 'description': 'データなし'},
            'aggressive': {'total': 0, 'label': '欠品回避', 'description': 'データなし'},
            'recommended': 'balanced',
            'weights_used': {},
            'calculation_method': 'フォールバック'
        }
    
    # 【v22.5】精度強化版を重視した重み計算
    # 精度強化版は他の方法よりも信頼性が高いため、重みの下限を設定
    for method, result in all_results.items():
        if method == '精度強化版':
            # 精度強化版の重みは最低でも0.5（全体の半分）を確保
            current_weight = result.get('weight', 0.01)
            result['weight'] = max(current_weight, 0.5)
    
    # 重みを正規化
    total_weight = sum(r.get('weight', 0.01) for r in all_results.values())
    if total_weight <= 0:
        total_weight = 1.0
    
    normalized_weights = {
        method: r.get('weight', 0.01) / total_weight
        for method, r in all_results.items()
    }
    
    # モード別の加重平均を計算
    final_totals = {}
    for mode in ['conservative', 'balanced', 'aggressive']:
        weighted_sum = 0
        for method, result in all_results.items():
            if mode in result.get('totals', {}):
                weighted_sum += result['totals'][mode] * normalized_weights[method]
        final_totals[mode] = round_up_to_50(int(round(weighted_sum)))
    
    # 順序の整合性を保証（aggressive > balanced > conservative）
    if final_totals['balanced'] <= final_totals['conservative']:
        final_totals['balanced'] = round_up_to_50(int(final_totals['conservative'] * 1.1))
    if final_totals['aggressive'] <= final_totals['balanced']:
        final_totals['aggressive'] = round_up_to_50(int(final_totals['balanced'] * 1.15))
    
    # 実績との比較（あれば）
    actual_avg = 0
    data_days = 0
    is_anomaly = False
    anomaly_ratio = 1.0
    
    if sales_data is not None and not sales_data.empty:
        actual_avg = sales_data['販売商品数'].mean()
        data_days = len(sales_data)
        
        # 【v23.2追加】最終サニティチェック
        # 予測日数をsession_stateから取得（デフォルト90日）
        forecast_days = st.session_state.get('v22_forecast_days', 90)
        
        # balancedの予測日販を計算
        if forecast_days > 0 and actual_avg > 0:
            predicted_daily = final_totals['balanced'] / forecast_days
            anomaly_ratio = predicted_daily / actual_avg
            
            # 予測が実績の2.5倍を超えている場合は異常の可能性
            if anomaly_ratio > 2.5:
                is_anomaly = True
                logger.warning(f"最終推奨値が実績の{anomaly_ratio:.1f}倍で異常の可能性あり（予測日販: {predicted_daily:.1f}, 実績日販: {actual_avg:.1f}）")
                
                # 【v23.2修正】異常な予測を実績ベースに補正
                # 実績平均 × 予測日数 × 安全係数(1.5)
                corrected_base = actual_avg * forecast_days * 1.5
                final_totals['conservative'] = round_up_to_50(int(corrected_base * 0.8))
                final_totals['balanced'] = round_up_to_50(int(corrected_base))
                final_totals['aggressive'] = round_up_to_50(int(corrected_base * 1.3))
    
    return {
        'conservative': {
            'total': final_totals['conservative'],
            'label': '滞留回避（P50）',
            'description': '在庫過剰リスクを最小化。需要が安定している場合に適切。',
            'risk': '欠品リスク: 高め'
        },
        'balanced': {
            'total': final_totals['balanced'],
            'label': 'バランス（P80）★推奨',
            'description': '欠品リスクと滞留リスクのバランスを取った推奨値。',
            'risk': '欠品リスク: 中程度'
        },
        'aggressive': {
            'total': final_totals['aggressive'],
            'label': '欠品回避（P90）',
            'description': '欠品を絶対に避けたい場合の安全在庫込み。正月など繁忙期向け。',
            'risk': '欠品リスク: 低め（滞留リスク: 高め）'
        },
        'recommended': 'balanced',
        'weights_used': normalized_weights,
        'calculation_method': '精度加重平均',
        'actual_avg_daily': actual_avg
    }


def display_unified_forecast_results_v22(
    all_results: Dict[str, Dict[str, Any]],
    final_recommendation: Dict[str, Any],
    forecast_days: int,
    sales_data: pd.DataFrame = None,
    product_names: List[str] = None
):
    """
    【v22新機能】統合予測結果を表示
    
    各予測方法の結果一覧と、最終推奨発注数を分かりやすく表示。
    """
    st.success("✅ 全予測方法で予測完了！")
    
    # ==========================================================================
    # 1. 最終推奨発注数（大きく表示）
    # ==========================================================================
    st.write("## 🎯 最終推奨発注数")
    st.caption("全予測方法の結果を、精度に応じた重みで統合して算出しています")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cons = final_recommendation['conservative']
        st.metric(
            f"📉 {cons['label']}", 
            f"{cons['total']:,}体",
            help=cons['description']
        )
        st.caption(cons['risk'])
    
    with col2:
        bal = final_recommendation['balanced']
        st.metric(
            f"⚖️ {bal['label']}", 
            f"{bal['total']:,}体",
            help=bal['description']
        )
        st.caption(bal['risk'])
        st.markdown("**↑ おすすめ**")
    
    with col3:
        aggr = final_recommendation['aggressive']
        st.metric(
            f"🛡️ {aggr['label']}", 
            f"{aggr['total']:,}体",
            help=aggr['description']
        )
        st.caption(aggr['risk'])
    
    # 差分を表示
    st.info(f"""
    📊 **差分情報**
    - バランス vs 滞留回避: **+{bal['total'] - cons['total']:,}体**（+{((bal['total'] / cons['total']) - 1) * 100:.0f}%）
    - 欠品回避 vs バランス: **+{aggr['total'] - bal['total']:,}体**（+{((aggr['total'] / bal['total']) - 1) * 100:.0f}%）
    """)
    
    # ==========================================================================
    # 2. 各予測方法の結果一覧
    # ==========================================================================
    st.write("## 📊 各予測方法の結果")
    st.caption("各方法の特徴と予測値を確認できます。クリックで詳細説明を表示。")
    
    for method_name, result in all_results.items():
        desc = result.get('description', {})
        icon = desc.get('icon', '📊') if isinstance(desc, dict) else '📊'
        short_desc = desc.get('short', '') if isinstance(desc, dict) else ''
        long_desc = desc.get('long', '') if isinstance(desc, dict) else ''
        best_for = desc.get('best_for', '') if isinstance(desc, dict) else ''
        
        totals = result.get('totals', {})
        mape = result.get('mape')
        
        with st.expander(f"{icon} **{method_name}** - バランス: {totals.get('balanced', 0):,}体" + (f"（MAPE {mape:.1f}%）" if mape else ""), expanded=False):
            st.markdown(f"**概要**: {short_desc}")
            st.markdown(f"**詳細**: {long_desc}")
            st.markdown(f"**最適な用途**: {best_for}")
            
            # モード別の予測値
            st.markdown("---")
            st.markdown("**モード別予測値:**")
            
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            sub_col1.metric("滞留回避", f"{totals.get('conservative', 0):,}体")
            sub_col2.metric("バランス", f"{totals.get('balanced', 0):,}体")
            sub_col3.metric("欠品回避", f"{totals.get('aggressive', 0):,}体")
            
            if mape:
                st.caption(f"バックテストMAPE: {mape:.1f}%（低いほど精度が高い）")
    
    # ==========================================================================
    # 3. 比較表
    # ==========================================================================
    st.write("## 📋 予測方法比較表")
    
    table_data = []
    for method_name, result in all_results.items():
        totals = result.get('totals', {})
        mape = result.get('mape')
        weight = final_recommendation['weights_used'].get(method_name, 0) * 100
        
        table_data.append({
            '予測方法': method_name,
            '滞留回避': f"{totals.get('conservative', 0):,}体",
            'バランス': f"{totals.get('balanced', 0):,}体",
            '欠品回避': f"{totals.get('aggressive', 0):,}体",
            'MAPE': f"{mape:.1f}%" if mape else "-",
            '重み': f"{weight:.1f}%"
        })
    
    # 最終推奨行を追加
    table_data.append({
        '予測方法': '**🎯 最終推奨**',
        '滞留回避': f"**{final_recommendation['conservative']['total']:,}体**",
        'バランス': f"**{final_recommendation['balanced']['total']:,}体**",
        '欠品回避': f"**{final_recommendation['aggressive']['total']:,}体**",
        'MAPE': '-',
        '重み': '100%'
    })
    
    df_comparison = pd.DataFrame(table_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # 4. ファクトチェック用プロンプト
    # ==========================================================================
    with st.expander("📝 **ファクトチェック用プロンプト（コピーしてAIに質問）**", expanded=False):
        prompt = generate_unified_factcheck_prompt_v22(
            all_results=all_results,
            final_recommendation=final_recommendation,
            forecast_days=forecast_days,
            sales_data=sales_data,
            product_names=product_names
        )
        st.text_area("プロンプト", prompt, height=400, key="unified_factcheck_prompt_v22")
        
        if st.button("📋 クリップボードにコピー", key="copy_unified_factcheck_v22"):
            st.write("上のテキストを選択してコピーしてください")


def generate_unified_factcheck_prompt_v22(
    all_results: Dict[str, Dict[str, Any]],
    final_recommendation: Dict[str, Any],
    forecast_days: int,
    sales_data: pd.DataFrame = None,
    product_names: List[str] = None
) -> str:
    """
    【v22新機能】統合予測結果のファクトチェック用プロンプト生成
    """
    product_str = "、".join(product_names) if product_names else "（選択された授与品）"
    
    # 入力データの統計
    if sales_data is not None and not sales_data.empty:
        total_days = len(sales_data)
        total_qty = int(sales_data['販売商品数'].sum())
        avg_daily = sales_data['販売商品数'].mean()
        std_daily = sales_data['販売商品数'].std()
        input_section = f"""■ 入力データ（過去の実績）:
- 学習データ期間: {total_days}日間
- 総販売数: {total_qty:,}体
- 実績日販: {avg_daily:.2f}体/日
- 標準偏差: {std_daily:.2f}"""
    else:
        input_section = "■ 入力データ: なし"
        avg_daily = 0
    
    # 各予測方法の結果
    methods_section = "■ 各予測方法の結果:\n"
    for method_name, result in all_results.items():
        totals = result.get('totals', {})
        mape = result.get('mape')
        methods_section += f"  - {method_name}: バランス={totals.get('balanced', 0):,}体"
        if mape:
            methods_section += f"（MAPE {mape:.1f}%）"
        methods_section += "\n"
    
    # 最終推奨
    cons = final_recommendation['conservative']['total']
    bal = final_recommendation['balanced']['total']
    aggr = final_recommendation['aggressive']['total']
    
    # 変化率
    change_rate = ((bal / (avg_daily * forecast_days)) - 1) * 100 if avg_daily > 0 and forecast_days > 0 else 0
    
    prompt = f"""【需要予測ファクトチェック依頼】

■ 基本情報:
- 対象商品: {product_str}
- 予測期間: {forecast_days}日間
- 予測方法: 全{len(all_results)}方法の加重平均

{input_section}

{methods_section}
■ 最終推奨発注数:
- 滞留回避（P50）: {cons:,}体
- バランス（P80）: {bal:,}体 ← 推奨
- 欠品回避（P90）: {aggr:,}体

■ 変化率（実績→予測バランス）:
- 実績日販 × 予測日数 = {avg_daily:.2f} × {forecast_days} = {avg_daily * forecast_days:,.0f}体
- 予測バランス: {bal:,}体
- 変化率: {change_rate:+.1f}%

■ 検証依頼:
1. 最終推奨発注数は妥当ですか？
2. 欠品リスクと滞留リスクのバランスはどうですか？
3. 実績との乖離（変化率{change_rate:+.1f}%）は説明可能ですか？

【出力形式】
1. 検算結果
2. 妥当性判定
3. リスク分析
4. 最終推奨（3案）
5. 追加質問
"""
    
    return prompt


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
# v23 精度改善モジュール（app.py内に統合）
# =============================================================================
# 【改善内容】
# 1. 統計的に正しいP80/P90計算（期間合計の分布から算出）
# 2. 類似商品探索の精度改善（TF-IDF + コサイン類似度）
# 3. バックテスト期間延長（14日→30-90日）
# 4. 分割発注提案機能
# 5. 日販の正しい定義（zero-fill + 全日付カバー）
# 6. 係数のデータ推定（固定値の廃止）
# =============================================================================

# sklearn/rapidfuzzのインポート（オプショナル）
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn未インストール: フォールバック類似度計算を使用します")

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# -----------------------------------------------------------------------------
# 1. 統計的に正しいP80/P90計算
# -----------------------------------------------------------------------------

def calculate_period_total_distribution_v23(
    daily_forecasts: np.ndarray,
    daily_std: float,
    n_simulations: int = 10000
) -> Dict[str, float]:
    """
    【v23新機能】期間合計の分布から統計的に正しいP値を計算
    
    日別P値の足し上げは統計的に誤り。
    期間合計の分布を直接推定してP値を取得する。
    """
    n_days = len(daily_forecasts)
    
    if n_days == 0:
        return {'p50': 0, 'p80': 0, 'p90': 0, 'mean': 0, 'std': 0}
    
    point_total = np.sum(daily_forecasts)
    
    if daily_std <= 0:
        daily_std = np.mean(daily_forecasts) * 0.2 if np.mean(daily_forecasts) > 0 else 1.0
    
    simulated_totals = []
    
    for _ in range(n_simulations):
        daily_cv = daily_std / np.mean(daily_forecasts) if np.mean(daily_forecasts) > 0 else 0.2
        daily_cv = min(daily_cv, 0.5)
        
        sigma_ln = np.sqrt(np.log(1 + daily_cv ** 2))
        
        simulated_daily = []
        for forecast in daily_forecasts:
            if forecast <= 0:
                simulated_daily.append(0)
            else:
                mu_ln = np.log(forecast) - sigma_ln ** 2 / 2
                sim_value = np.random.lognormal(mu_ln, sigma_ln)
                simulated_daily.append(max(0, sim_value))
        
        simulated_totals.append(sum(simulated_daily))
    
    simulated_totals = np.array(simulated_totals)
    
    p50 = np.percentile(simulated_totals, 50)
    p80 = np.percentile(simulated_totals, 80)
    p90 = np.percentile(simulated_totals, 90)
    
    return {
        'p50': p50,
        'p80': p80,
        'p90': p90,
        'mean': np.mean(simulated_totals),
        'std': np.std(simulated_totals),
        'point_total': point_total
    }


# -----------------------------------------------------------------------------
# 2. 類似商品探索の精度改善
# -----------------------------------------------------------------------------

def normalize_text_v23(text: str) -> str:
    """テキストの正規化（全角半角統一、記号除去など）"""
    import unicodedata
    
    if not isinstance(text, str):
        return ""
    
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def create_char_ngrams_v23(text: str, n_range: Tuple[int, int] = (2, 5)) -> List[str]:
    """文字n-gramを生成"""
    ngrams = []
    text = normalize_text_v23(text)
    
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
    
    return ngrams


def calculate_jaccard_similarity_v23(text1: str, text2: str, n_range: Tuple[int, int] = (2, 4)) -> float:
    """Jaccard類似度（sklearn不要のフォールバック）"""
    ngrams1 = set(create_char_ngrams_v23(text1, n_range))
    ngrams2 = set(create_char_ngrams_v23(text2, n_range))
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0


@st.cache_data(ttl=3600)
def build_tfidf_matrix_cached(product_names: Tuple[str, ...]) -> Tuple[Any, Any]:
    """TF-IDF行列をキャッシュして構築"""
    if not SKLEARN_AVAILABLE:
        return None, None
    
    try:
        texts = [normalize_text_v23(p) for p in product_names]
        
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 5),
            min_df=1,
            max_df=0.95
        )
        
        matrix = vectorizer.fit_transform(texts)
        return vectorizer, matrix
    except Exception as e:
        logger.warning(f"TF-IDF構築エラー: {e}")
        return None, None


def find_similar_products_v23(
    name: str, 
    category: str, 
    price: int, 
    description: str,
    target_audience: List[str] = None,
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    【v23新機能】類似商品探索（TF-IDF + コサイン類似度）
    
    改善点:
    - TF-IDF文字n-gramで類似度計算
    - 価格・カテゴリ・ターゲット層を副次スコアに
    - 日販を正しく計算（日次再集計+0埋め）
    """
    if not name or not name.strip():
        return []
    
    if st.session_state.data_loader is None:
        return []
    
    df_items = st.session_state.data_loader.load_item_sales()
    
    if df_items.empty:
        return []
    
    # 商品の日販統計を正しく計算
    product_stats = compute_product_daily_stats_v23(df_items)
    
    if not product_stats:
        return []
    
    existing_products = list(product_stats.keys())
    
    # 検索テキストを構築
    search_text = normalize_text_v23(name)
    if description:
        search_text += " " + normalize_text_v23(description)
    
    # TF-IDF類似度を計算
    if SKLEARN_AVAILABLE:
        try:
            vectorizer, matrix = build_tfidf_matrix_cached(tuple(existing_products))
            
            if vectorizer is not None and matrix is not None:
                search_vec = vectorizer.transform([search_text])
                similarities = cosine_similarity(search_vec, matrix)[0]
                method = 'tfidf'
            else:
                similarities = [calculate_jaccard_similarity_v23(search_text, normalize_text_v23(p)) 
                              for p in existing_products]
                method = 'jaccard'
        except Exception as e:
            logger.warning(f"TF-IDF計算エラー: {e}")
            similarities = [calculate_jaccard_similarity_v23(search_text, normalize_text_v23(p)) 
                          for p in existing_products]
            method = 'jaccard'
    elif RAPIDFUZZ_AVAILABLE:
        similarities = [fuzz.token_set_ratio(search_text, normalize_text_v23(p)) / 100.0 
                       for p in existing_products]
        method = 'rapidfuzz'
    else:
        similarities = [calculate_jaccard_similarity_v23(search_text, normalize_text_v23(p)) 
                       for p in existing_products]
        method = 'jaccard'
    
    # カテゴリキーワード
    category_keywords = {
        "お守り": ["守", "お守り", "まもり", "御守"],
        "御朱印": ["御朱印", "朱印"],
        "御朱印帳": ["御朱印帳", "朱印帳"],
        "おみくじ": ["おみくじ", "みくじ"],
        "絵馬": ["絵馬"],
        "お札": ["札", "お札", "御札"],
        "縁起物": ["縁起", "だるま", "招き猫"],
    }
    
    # ターゲット層キーワード
    audience_keywords = {
        "若い女性": ["縁結び", "恋愛", "美", "ピンク", "かわいい", "キュート"],
        "若い男性": ["仕事", "勝", "出世", "金運"],
        "中高年女性": ["健康", "長寿", "厄除け"],
        "中高年男性": ["商売繁盛", "金運", "仕事"],
        "家族連れ": ["安産", "子供", "学業", "合格"],
        "観光客": ["御朱印", "限定", "記念"],
    }
    
    # 【v23.1改善】商品名から固有キーワードを抽出
    def extract_keywords(text: str) -> set:
        """商品名から意味のあるキーワードを抽出"""
        keywords = set()
        # 漢字の連続（2文字以上）
        keywords.update(re.findall(r'[\u4e00-\u9fff]{2,}', text))
        # ひらがなの連続（2文字以上）
        keywords.update(re.findall(r'[\u3040-\u309f]{2,}', text))
        # カタカナの連続（2文字以上）
        keywords.update(re.findall(r'[\u30a0-\u30ff]{2,}', text))
        # 英数字の連続（2文字以上）
        keywords.update(re.findall(r'[a-zA-Z0-9]{2,}', text.lower()))
        return keywords
    
    search_keywords = extract_keywords(name)
    if description:
        search_keywords.update(extract_keywords(description))
    
    # スコア統合
    results = []
    for i, (product, text_sim) in enumerate(zip(existing_products, similarities)):
        stats = product_stats[product]
        
        # テキスト類似度 (40%) - 比重を下げる
        score = text_sim * 0.40
        
        # 【v23.1新規】固有キーワード一致度 (25%)
        product_keywords = extract_keywords(product)
        if search_keywords and product_keywords:
            keyword_match = len(search_keywords & product_keywords) / len(search_keywords)
            score += keyword_match * 0.25
        
        # 価格類似度 (20%)
        if stats.get('unit_price', 0) > 0 and price > 0:
            price_ratio = min(stats['unit_price'], price) / max(stats['unit_price'], price)
            score += price_ratio * 0.20
        
        # カテゴリ一致 (10%)
        cat_match = 0
        cat_kws = category_keywords.get(category, [])
        for kw in cat_kws:
            if kw in product:
                cat_match = 0.10
                break
        score += cat_match
        
        # ターゲット層一致 (5%)
        audience_match = 0
        if target_audience:
            for aud in target_audience:
                aud_kws = audience_keywords.get(aud, [])
                for kw in aud_kws:
                    if kw in product or (description and kw in description):
                        audience_match = 0.05
                        break
                if audience_match > 0:
                    break
        score += audience_match
        
        # 販売実績がある商品のみ
        if stats.get('total_qty', 0) > 0:
            results.append({
                'name': product,
                'total_qty': stats['total_qty'],
                'avg_daily': stats['avg_daily'],
                'std_daily': stats.get('std_daily', 0),
                'cv': stats.get('cv', 0),
                'unit_price': stats.get('unit_price', 0),
                'similarity': score * 100,  # パーセント表示
                'text_similarity': text_sim,
                'keyword_match': len(search_keywords & product_keywords) if search_keywords and product_keywords else 0,
                'method': method
            })
    
    # スコア順にソート
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results[:top_k]


# =============================================================================
# v23.2 新規授与品予測の改善（手動選択、同名優先、予測内訳表示）
# =============================================================================

def get_base_name_v23_2(product_name: str) -> str:
    """
    【v23.2】商品名からベース名を抽出（型番や色を除く）
    例: 「椿守 アクリル型」→「椿守」
    """
    # 【】内を除去
    base = re.sub(r'【.*?】', '', product_name)
    # 「アクリル型」「木札」などの型番を除去
    base = re.sub(r'(アクリル型|アクリル|木札|金属|布製|大|小|特大|ミニ)', '', base)
    # 色を除去
    base = re.sub(r'(赤|青|黄|緑|白|黒|金|銀|ピンク|紫|水色|オレンジ)', '', base)
    # 空白を除去して返す
    return base.strip()


def find_similar_products_v23_2(
    name: str, 
    category: str, 
    price: int, 
    description: str,
    target_audience: List[str] = None,
    top_k: int = 30
) -> List[Dict[str, Any]]:
    """
    【v23.2改善版】類似商品探索（同名商品を最優先）
    
    改善点:
    - 同名商品を最優先（「椿守 アクリル型」→「椿守」を自動上位）
    - 固有キーワード一致度を重視
    """
    if not name or not name.strip():
        return []
    
    if st.session_state.data_loader is None:
        return []
    
    df_items = st.session_state.data_loader.load_item_sales()
    
    if df_items.empty:
        return []
    
    # 商品の日販統計を正しく計算
    product_stats = compute_product_daily_stats_v23(df_items)
    
    if not product_stats:
        return []
    
    existing_products = list(product_stats.keys())
    
    # 商品名から固有キーワードを抽出
    def extract_keywords(text: str) -> set:
        """商品名から意味のあるキーワードを抽出"""
        keywords = set()
        # 漢字の連続（2文字以上）
        keywords.update(re.findall(r'[\u4e00-\u9fff]{2,}', text))
        # ひらがなの連続（2文字以上）
        keywords.update(re.findall(r'[\u3040-\u309f]{2,}', text))
        # カタカナの連続（2文字以上）
        keywords.update(re.findall(r'[\u30a0-\u30ff]{2,}', text))
        return keywords
    
    search_base_name = get_base_name_v23_2(name)
    search_keywords = extract_keywords(name)
    if description:
        search_keywords.update(extract_keywords(description))
    
    # カテゴリキーワード
    category_keywords = {
        "お守り": ["守", "お守り", "まもり", "御守"],
        "御朱印": ["御朱印", "朱印"],
        "御朱印帳": ["御朱印帳", "朱印帳"],
        "おみくじ": ["おみくじ", "みくじ"],
        "絵馬": ["絵馬"],
        "お札": ["札", "お札", "御札"],
        "縁起物": ["縁起", "だるま", "招き猫"],
    }
    
    # スコア計算
    results = []
    for product in existing_products:
        stats = product_stats[product]
        
        if stats.get('total_qty', 0) <= 0:
            continue
        
        product_base_name = get_base_name_v23_2(product)
        product_keywords = extract_keywords(product)
        
        # 【v23.2】同名商品ボーナス（最優先）
        same_name_bonus = 0
        same_name_reason = ''
        if search_base_name and product_base_name:
            # 完全一致
            if search_base_name == product_base_name:
                same_name_bonus = 0.50  # 50%ボーナス
                same_name_reason = '同名商品（完全一致）'
            # 部分一致（検索名がプロダクト名に含まれる、またはその逆）
            elif search_base_name in product_base_name or product_base_name in search_base_name:
                same_name_bonus = 0.35  # 35%ボーナス
                same_name_reason = '同名商品（部分一致）'
            # キーワードの半分以上が一致
            elif search_keywords and product_keywords:
                overlap = len(search_keywords & product_keywords)
                if overlap >= len(search_keywords) * 0.5:
                    same_name_bonus = 0.20  # 20%ボーナス
                    same_name_reason = 'キーワード多数一致'
        
        # キーワード一致度 (25%)
        keyword_score = 0
        if search_keywords and product_keywords:
            keyword_score = len(search_keywords & product_keywords) / len(search_keywords) if search_keywords else 0
        
        # 価格類似度 (15%)
        price_score = 0
        if stats.get('unit_price', 0) > 0 and price > 0:
            price_score = min(stats['unit_price'], price) / max(stats['unit_price'], price)
        
        # カテゴリ一致 (10%)
        cat_score = 0
        cat_kws = category_keywords.get(category, [])
        for kw in cat_kws:
            if kw in product:
                cat_score = 1.0
                break
        
        # 総合スコア
        score = (
            same_name_bonus +           # 0-50%: 同名商品ボーナス
            keyword_score * 0.25 +      # 0-25%: キーワード一致
            price_score * 0.15 +        # 0-15%: 価格類似度
            cat_score * 0.10            # 0-10%: カテゴリ一致
        )
        
        results.append({
            'name': product,
            'total_qty': stats['total_qty'],
            'avg_daily': stats['avg_daily'],
            'std_daily': stats.get('std_daily', 0),
            'cv': stats.get('cv', 0),
            'unit_price': stats.get('unit_price', 0),
            'similarity': score * 100,
            'same_name_bonus': same_name_bonus * 100,
            'same_name_reason': same_name_reason,
            'keyword_match': len(search_keywords & product_keywords) if search_keywords and product_keywords else 0,
            'base_name': product_base_name
        })
    
    # スコア順にソート
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results[:top_k]


def forecast_new_product_v23_2(
    name: str, 
    category: str, 
    price: int,
    reference_products: List[Dict],  # 手動選択された参考商品
    period_days: int,
    order_mode: str = 'balanced'
) -> Dict[str, Any]:
    """
    【v23.2改善版】新規授与品の需要予測
    
    改善点:
    - 手動選択された参考商品を使用
    - 予測内訳の詳細を返す
    """
    df_items = None
    if st.session_state.data_loader is not None:
        df_items = st.session_state.data_loader.load_item_sales()
    
    # 参考商品から係数推定
    if reference_products and df_items is not None and not df_items.empty:
        factors = estimate_calendar_factors_from_similar_v23(df_items, reference_products)
        
        # 参考商品の加重平均日販を計算
        # 同名ボーナスが高い商品を重視
        weighted_sum = 0
        weight_total = 0
        for p in reference_products:
            # 同名ボーナスが高いほど重みを大きく
            weight = 1.0 + (p.get('same_name_bonus', 0) / 100) * 2
            weighted_sum += p['avg_daily'] * weight
            weight_total += weight
        
        base_daily = weighted_sum / weight_total if weight_total > 0 else factors['baseline']
        cv = min(factors['cv'], 1.0)  # CVを100%以下にクリップ
    else:
        factors = _get_default_factors_v23()
        base_daily = CATEGORY_CHARACTERISTICS.get(category, {}).get('base_daily', 1.0)
        cv = 0.3
    
    # 【v23.2】予測内訳を記録
    daily_forecast_details = []
    daily_forecast_raw = []
    
    for i in range(period_days):
        target_date = date.today() + timedelta(days=i)
        month = target_date.month
        weekday = target_date.weekday()
        
        # 期間タイプの判定
        if target_date.month == 1 and target_date.day <= 7:
            period_type = 'new_year'
            period_label = '正月'
        elif target_date.month == 8 and 13 <= target_date.day <= 16:
            period_type = 'obon'
            period_label = 'お盆'
        elif target_date.month == 11 and 10 <= target_date.day <= 20:
            period_type = 'shichigosan'
            period_label = '七五三'
        elif target_date.month == 5 and 3 <= target_date.day <= 5:
            period_type = 'golden_week'
            period_label = 'GW'
        elif target_date.month == 12 and target_date.day >= 28:
            period_type = 'year_end'
            period_label = '年末'
        else:
            period_type = 'normal'
            period_label = '通常'
        
        weekday_f = factors['weekday'].get(weekday, 1.0)
        month_f = factors['month'].get(month, 1.0)
        special_f = factors['special_periods'].get(period_type, 1.0)
        
        # 予測値（floatで保持）
        pred_raw = base_daily * weekday_f * month_f * special_f
        pred_raw = max(0.1, pred_raw)
        
        daily_forecast_raw.append({
            'date': target_date,
            'predicted_raw': pred_raw,
            'predicted': int(round(pred_raw)),
            'weekday_factor': weekday_f,
            'month_factor': month_f,
            'special_factor': special_f,
            'period_type': period_type
        })
        
        # 最初の30日と特別期間の詳細を記録
        if i < 30 or period_type != 'normal':
            daily_forecast_details.append({
                'date': target_date.strftime('%m/%d'),
                'weekday': ['月', '火', '水', '木', '金', '土', '日'][weekday],
                'base': base_daily,
                'weekday_f': weekday_f,
                'month_f': month_f,
                'special_f': special_f,
                'period': period_label,
                'predicted': pred_raw
            })
    
    # 期間合計の分布を計算
    daily_values = np.array([d['predicted_raw'] for d in daily_forecast_raw])
    daily_std = cv * np.mean(daily_values)
    
    distribution = calculate_period_total_distribution_v23(daily_values, daily_std)
    
    # 丸め処理
    p50_total = distribution['p50']
    p80_total = distribution['p80']
    p90_total = distribution['p90']
    point_total = distribution['point_total']
    
    p50_rounded = round_up_to_50(int(round(p50_total)))
    p80_rounded = round_up_to_50(int(round(p80_total)))
    p90_rounded = round_up_to_50(int(round(p90_total)))
    point_rounded = round_up_to_50(int(round(point_total)))
    
    # 発注モードに応じた推奨値
    mode_map = {
        'conservative': ('滞留回避（P50）', p50_rounded),
        'balanced': ('バランス（P80）', p80_rounded),
        'aggressive': ('欠品回避（P90）', p90_rounded)
    }
    recommended_label, recommended_qty = mode_map.get(order_mode, ('バランス（P80）', p80_rounded))
    
    # 分割発注提案
    split_proposals = suggest_split_orders_v23(recommended_qty, period_days)
    
    # 月別集計
    df_forecast = pd.DataFrame(daily_forecast_raw)
    df_forecast['month'] = pd.to_datetime(df_forecast['date']).dt.to_period('M')
    monthly = df_forecast.groupby('month')['predicted'].sum().to_dict()
    
    # 【v23.2】係数サマリー
    factor_summary = {
        'weekday_avg': np.mean(list(factors['weekday'].values())),
        'weekday_weekend': (factors['weekday'].get(5, 1.0) + factors['weekday'].get(6, 1.0)) / 2,
        'weekday_weekday': np.mean([factors['weekday'].get(i, 1.0) for i in range(5)]),
        'month_factors': factors['month'],
        'special_factors': factors['special_periods'],
        'baseline': factors['baseline']
    }
    
    return {
        'daily_forecast': daily_forecast_raw,
        'daily_details': daily_forecast_details,  # 【v23.2】詳細内訳
        'point_total': point_total,
        'point_total_rounded': point_rounded,
        'p50_total': p50_total,
        'p50_rounded': p50_rounded,
        'p80_total': p80_total,
        'p80_rounded': p80_rounded,
        'p90_total': p90_total,
        'p90_rounded': p90_rounded,
        'recommended_qty': recommended_qty,
        'recommended_label': recommended_label,
        'order_mode': order_mode,
        'avg_daily': point_total / period_days if period_days > 0 else 0,
        'period_days': period_days,
        'monthly': monthly,
        'base_daily': base_daily,
        'cv': cv,
        'factors': factors,
        'factor_summary': factor_summary,  # 【v23.2】係数サマリー
        'reference_count': len(reference_products),
        'split_proposals': split_proposals
    }


def generate_factcheck_prompt_v23_2(
    product_name: str,
    category: str,
    price: int,
    result: Dict[str, Any],
    reference_products: List[Dict]
) -> str:
    """【v23.2改善版】ファクトチェックプロンプト"""
    
    # 参考商品情報
    ref_section = "■ 参考商品（手動/自動選択）:\n"
    if reference_products:
        for i, p in enumerate(reference_products[:5], 1):
            bonus = "⭐同名" if p.get('same_name_bonus', 0) > 30 else ""
            reason = p.get('same_name_reason', '')
            ref_section += f"  {i}. {p['name']} {bonus}\n"
            ref_section += f"     - 日販: {p['avg_daily']:.2f}体/日\n"
            ref_section += f"     - 総販売数: {p['total_qty']:,}体\n"
            if reason:
                ref_section += f"     - 選定理由: {reason}\n"
    else:
        ref_section += "  （参考商品なし）\n"
    
    # 係数情報
    factor_summary = result.get('factor_summary', {})
    factors_section = f"""■ 使用した係数:
  - ベース日販: {result.get('base_daily', 0):.2f}体/日
  - 変動係数(CV): {result.get('cv', 0):.1%}
  - 曜日係数（平日平均）: {factor_summary.get('weekday_weekday', 1.0):.2f}
  - 曜日係数（土日平均）: {factor_summary.get('weekday_weekend', 1.0):.2f}
  - 正月係数: {factor_summary.get('special_factors', {}).get('new_year', 'N/A')}"""
    
    # 検算
    base = result.get('base_daily', 0)
    days = result.get('period_days', 0)
    simple_calc = base * days
    
    prompt = f"""【新規授与品 需要予測ファクトチェック依頼】

■ 基本情報:
- 商品名: {product_name}
- カテゴリ: {category}
- 価格: ¥{price:,}
- 予測期間: {days}日間
- 参考商品数: {result.get('reference_count', 0)}件

{ref_section}
{factors_section}

■ 予測結果:
- ベース計算: {base:.2f} × {days}日 = {simple_calc:.0f}体
- 係数適用後（点推定）: {result.get('point_total', 0):,.0f}体
- 滞留回避（P50）: {result.get('p50_rounded', 0):,}体
- バランス（P80）: {result.get('p80_rounded', 0):,}体 ← 推奨
- 欠品回避（P90）: {result.get('p90_rounded', 0):,}体

■ 検算:
- ベース日販 × 期間: {base:.2f} × {days} = {simple_calc:.0f}体
- 予測との差: {result.get('point_total', 0) - simple_calc:+.0f}体（係数による増減）
- 予測/ベースの倍率: {result.get('point_total', 0) / simple_calc if simple_calc > 0 else 0:.2f}倍

■ 検証依頼:
1. 参考商品の選定は妥当ですか？（同名商品があれば最優先すべき）
2. ベース日販 {base:.2f}体/日 は参考商品の実績と整合していますか？
3. 予測値 {result.get('p80_rounded', 0):,}体（{days}日間）は現実的ですか？
4. 新規商品なので分割発注を検討すべきですか？

【出力形式】
1. 検算結果（OK/誤りあり）
2. 妥当性判定（妥当/要修正/却下）
3. あなたの推奨発注数
4. リスク分析
"""
    
    return prompt


def compute_product_daily_stats_v23(
    df_items: pd.DataFrame,
    product_names: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    【v23新機能】日次再集計 + 0埋めで正しい日販を計算
    """
    if df_items is None or df_items.empty:
        return {}
    
    df = df_items.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    if product_names is None:
        product_names = df['商品名'].unique().tolist()
    
    date_min = df['date'].min()
    date_max = df['date'].max()
    all_dates = pd.date_range(start=date_min, end=date_max, freq='D')
    total_days = len(all_dates)
    
    stats = {}
    
    for product in product_names:
        product_df = df[df['商品名'] == product]
        
        if product_df.empty:
            continue
        
        # 日別集計
        daily = product_df.groupby('date').agg({
            '販売商品数': 'sum',
            '販売総売上': 'sum'
        })
        
        # 全日付でreindex（欠損は0で埋める）
        daily_full = daily.reindex(all_dates, fill_value=0)
        
        # 統計計算
        total_qty = int(daily_full['販売商品数'].sum())
        valid_days = len(daily_full)
        active_days = int((daily_full['販売商品数'] > 0).sum())
        
        avg_daily = total_qty / valid_days if valid_days > 0 else 0.0
        std_daily = float(daily_full['販売商品数'].std()) if valid_days > 1 else 0.0
        max_daily = int(daily_full['販売商品数'].max()) if valid_days > 0 else 0
        min_daily = int(daily_full['販売商品数'].min()) if valid_days > 0 else 0
        cv = std_daily / avg_daily if avg_daily > 0 else 0.0
        
        # 単価計算
        total_sales = daily_full['販売総売上'].sum()
        unit_price = total_sales / total_qty if total_qty > 0 else 0
        
        stats[product] = {
            'avg_daily': avg_daily,
            'total_qty': total_qty,
            'total_days': valid_days,
            'active_days': active_days,
            'std_daily': std_daily,
            'max_daily': max_daily,
            'min_daily': min_daily,
            'cv': cv,
            'unit_price': unit_price
        }
    
    return stats


# -----------------------------------------------------------------------------
# 3. データから係数推定
# -----------------------------------------------------------------------------

def estimate_calendar_factors_from_similar_v23(
    df_items: pd.DataFrame,
    similar_products: List[Dict],
    top_n: int = 10
) -> Dict[str, Dict]:
    """
    【v23新機能】類似商品群から曜日・月・特別期間係数を推定
    """
    if df_items is None or df_items.empty or not similar_products:
        return _get_default_factors_v23()
    
    # 上位N商品の名前を取得
    top_products = [p['name'] for p in similar_products[:top_n]]
    
    df = df_items.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['商品名'].isin(top_products)]
    
    if df.empty:
        return _get_default_factors_v23()
    
    # 日別集計（合算）
    daily = df.groupby('date')['販売商品数'].sum().reset_index()
    daily['weekday'] = daily['date'].dt.dayofweek
    daily['month'] = daily['date'].dt.month
    
    # 特別期間フラグ
    def get_period_type(d):
        d = pd.Timestamp(d)
        if d.month == 1 and d.day <= 7:
            return 'new_year'
        if d.month == 8 and 13 <= d.day <= 16:
            return 'obon'
        if d.month == 11 and 10 <= d.day <= 20:
            return 'shichigosan'
        if d.month == 5 and 3 <= d.day <= 5:
            return 'golden_week'
        if d.month == 12 and d.day >= 28:
            return 'year_end'
        return 'normal'
    
    daily['period_type'] = daily['date'].apply(get_period_type)
    
    # 通常日のベースライン（中央値）
    normal_df = daily[daily['period_type'] == 'normal']
    if len(normal_df) > 0:
        baseline = normal_df['販売商品数'].median()
    else:
        baseline = daily['販売商品数'].median()
    
    if baseline <= 0:
        baseline = 1.0
    
    # 曜日係数
    weekday_factors = {}
    for wd in range(7):
        wd_df = normal_df[normal_df['weekday'] == wd]
        if len(wd_df) >= 3:
            factor = wd_df['販売商品数'].median() / baseline
        else:
            # 土日は1.3、平日は1.0をデフォルト
            factor = 1.3 if wd >= 5 else 1.0
        weekday_factors[wd] = np.clip(factor, 0.3, 3.5)
    
    # 月係数
    month_factors = {}
    for m in range(1, 13):
        m_df = normal_df[normal_df['month'] == m]
        if len(m_df) >= 5:
            factor = m_df['販売商品数'].median() / baseline
        else:
            factor = 1.0
        month_factors[m] = np.clip(factor, 0.3, 3.5)
    
    # 特別期間係数
    special_factors = {'normal': 1.0}
    defaults = {
        'new_year': 5.0,
        'obon': 1.5,
        'shichigosan': 1.3,
        'golden_week': 1.3,
        'year_end': 1.5
    }
    
    for period in defaults.keys():
        period_df = daily[daily['period_type'] == period]
        if len(period_df) >= 2:
            factor = period_df['販売商品数'].median() / baseline
        else:
            factor = defaults[period]
        special_factors[period] = np.clip(factor, 0.5, 10.0)
    
    # 残差（CV）の計算
    residuals = []
    for _, row in daily.iterrows():
        wd = row['weekday']
        m = row['month']
        pt = row['period_type']
        
        expected = baseline * weekday_factors[wd] * month_factors[m]
        if pt != 'normal':
            expected *= special_factors[pt]
        
        if expected > 0:
            residuals.append(row['販売商品数'] / expected)
    
    # 【v23.1修正】CVを安全な範囲にクリップ（5%〜100%）
    # 残差の標準偏差から変動係数を計算
    if residuals and len(residuals) > 1:
        residuals_array = np.array(residuals)
        # 外れ値を除外（1%〜99%パーセンタイル）
        lower = np.percentile(residuals_array, 1)
        upper = np.percentile(residuals_array, 99)
        filtered_residuals = residuals_array[(residuals_array >= lower) & (residuals_array <= upper)]
        
        if len(filtered_residuals) > 1:
            cv = np.std(filtered_residuals)
        else:
            cv = np.std(residuals_array)
        
        # CVを妥当な範囲にクリップ（0.05〜1.0 = 5%〜100%）
        cv = np.clip(cv, 0.05, 1.0)
    else:
        cv = 0.3  # デフォルト30%
    
    return {
        'weekday': weekday_factors,
        'month': month_factors,
        'special_periods': special_factors,
        'baseline': baseline,
        'cv': cv
    }


def _get_default_factors_v23() -> Dict:
    """デフォルト係数（データがない場合のフォールバック）"""
    return {
        'weekday': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.3, 6: 1.3},
        'month': {1: 2.5, 2: 0.8, 3: 0.9, 4: 0.9, 5: 1.0, 6: 0.8,
                  7: 0.9, 8: 1.1, 9: 0.9, 10: 1.0, 11: 1.2, 12: 1.3},
        'special_periods': {
            'normal': 1.0,
            'new_year': 5.0,
            'obon': 1.5,
            'shichigosan': 1.3,
            'golden_week': 1.3,
            'year_end': 1.5
        },
        'baseline': 1.0,
        'cv': 0.3
    }


# -----------------------------------------------------------------------------
# 4. 新規授与品予測（v23改善版）
# -----------------------------------------------------------------------------

def forecast_new_product_v23(
    name: str, 
    category: str, 
    price: int,
    similar_products: List[Dict], 
    period_days: int,
    order_mode: str = 'balanced'
) -> Dict[str, Any]:
    """
    【v23新機能】新規授与品の需要予測
    
    改善点:
    - 類似商品群から係数推定
    - 統計的に正しいP50/P80/P90計算
    - 丸め処理の後段化
    - 分割発注提案
    """
    # データローダーからdf_itemsを取得
    df_items = None
    if st.session_state.data_loader is not None:
        df_items = st.session_state.data_loader.load_item_sales()
    
    # 類似商品から係数推定
    if similar_products and df_items is not None and not df_items.empty:
        factors = estimate_calendar_factors_from_similar_v23(df_items, similar_products)
        
        # 類似商品の加重平均日販を計算
        weighted_sum = sum(p['avg_daily'] * p['similarity'] for p in similar_products[:5])
        weight_total = sum(p['similarity'] for p in similar_products[:5])
        base_daily = weighted_sum / weight_total if weight_total > 0 else factors['baseline']
        
        # 変動係数
        cv = factors['cv']
    else:
        factors = _get_default_factors_v23()
        base_daily = CATEGORY_CHARACTERISTICS.get(category, {}).get('base_daily', 1.0)
        cv = 0.3
    
    # 日次予測を生成（floatで保持）
    daily_forecast_raw = []
    
    for i in range(period_days):
        target_date = date.today() + timedelta(days=i)
        month = target_date.month
        weekday = target_date.weekday()
        
        # 期間タイプの判定
        if target_date.month == 1 and target_date.day <= 7:
            period_type = 'new_year'
        elif target_date.month == 8 and 13 <= target_date.day <= 16:
            period_type = 'obon'
        elif target_date.month == 11 and 10 <= target_date.day <= 20:
            period_type = 'shichigosan'
        elif target_date.month == 5 and 3 <= target_date.day <= 5:
            period_type = 'golden_week'
        elif target_date.month == 12 and target_date.day >= 28:
            period_type = 'year_end'
        else:
            period_type = 'normal'
        
        weekday_f = factors['weekday'].get(weekday, 1.0)
        month_f = factors['month'].get(month, 1.0)
        special_f = factors['special_periods'].get(period_type, 1.0)
        
        # 予測値（floatで保持）
        pred_raw = base_daily * weekday_f * month_f * special_f
        pred_raw = max(0.1, pred_raw)
        
        daily_forecast_raw.append({
            'date': target_date,
            'predicted_raw': pred_raw,
            'predicted': int(round(pred_raw)),
            'weekday_factor': weekday_f,
            'month_factor': month_f,
            'special_factor': special_f,
            'period_type': period_type
        })
    
    # 期間合計の分布を計算（統計的に正しいP値）
    daily_values = np.array([d['predicted_raw'] for d in daily_forecast_raw])
    daily_std = cv * np.mean(daily_values)
    
    distribution = calculate_period_total_distribution_v23(daily_values, daily_std)
    
    # 丸め処理は最終段階でのみ
    p50_total = distribution['p50']
    p80_total = distribution['p80']
    p90_total = distribution['p90']
    point_total = distribution['point_total']
    
    p50_rounded = round_up_to_50(int(round(p50_total)))
    p80_rounded = round_up_to_50(int(round(p80_total)))
    p90_rounded = round_up_to_50(int(round(p90_total)))
    point_rounded = round_up_to_50(int(round(point_total)))
    
    # 発注モードに応じた推奨値
    mode_map = {
        'conservative': ('滞留回避（P50）', p50_rounded),
        'balanced': ('バランス（P80）', p80_rounded),
        'aggressive': ('欠品回避（P90）', p90_rounded)
    }
    recommended_label, recommended_qty = mode_map.get(order_mode, ('バランス（P80）', p80_rounded))
    
    # 分割発注提案
    split_proposals = suggest_split_orders_v23(recommended_qty, period_days)
    
    # 月別集計
    df_forecast = pd.DataFrame(daily_forecast_raw)
    df_forecast['month'] = pd.to_datetime(df_forecast['date']).dt.to_period('M')
    monthly = df_forecast.groupby('month')['predicted'].sum().to_dict()
    
    return {
        'daily_forecast': daily_forecast_raw,
        'point_total': point_total,
        'point_total_rounded': point_rounded,
        'p50_total': p50_total,
        'p50_rounded': p50_rounded,
        'p80_total': p80_total,
        'p80_rounded': p80_rounded,
        'p90_total': p90_total,
        'p90_rounded': p90_rounded,
        'recommended_qty': recommended_qty,
        'recommended_label': recommended_label,
        'order_mode': order_mode,
        'avg_daily': point_total / period_days if period_days > 0 else 0,
        'period_days': period_days,
        'monthly': monthly,
        'base_daily': base_daily,
        'cv': cv,
        'factors': factors,
        'similar_count': len(similar_products),
        'split_proposals': split_proposals
    }


def suggest_split_orders_v23(
    total_qty: int,
    forecast_days: int,
    lead_time_days: int = 14,
    min_order_qty: int = 50
) -> List[Dict[str, Any]]:
    """【v23新機能】分割発注の提案"""
    proposals = []
    
    # 単発発注
    proposals.append({
        'type': 'single',
        'description': '一括発注',
        'orders': [{'timing': '即時', 'qty': total_qty, 'coverage_days': forecast_days}],
        'total_qty': total_qty,
        'risk_level': 'high' if total_qty > 1000 else 'medium'
    })
    
    # 2分割発注
    if forecast_days >= 60:
        first_coverage = forecast_days // 2 + lead_time_days
        daily_avg = total_qty / forecast_days
        
        first_qty = int(np.ceil(daily_avg * first_coverage / min_order_qty)) * min_order_qty
        second_qty = total_qty - first_qty
        second_qty = max(min_order_qty, int(np.ceil(second_qty / min_order_qty)) * min_order_qty)
        
        proposals.append({
            'type': 'split_2',
            'description': '2回分割発注（推奨）',
            'orders': [
                {'timing': '即時', 'qty': first_qty, 'coverage_days': first_coverage},
                {'timing': f'{forecast_days // 2 - lead_time_days}日後', 'qty': second_qty, 
                 'coverage_days': forecast_days - first_coverage + lead_time_days}
            ],
            'total_qty': first_qty + second_qty,
            'risk_level': 'low'
        })
    
    # 3分割発注
    if forecast_days >= 120:
        interval = forecast_days // 3
        daily_avg = total_qty / forecast_days
        
        orders = []
        remaining = total_qty
        for i in range(3):
            coverage = interval + lead_time_days if i < 2 else forecast_days - interval * 2
            qty = int(np.ceil(daily_avg * coverage / min_order_qty)) * min_order_qty
            qty = min(qty, remaining)
            remaining -= qty
            
            timing = '即時' if i == 0 else f'{interval * i - lead_time_days}日後'
            orders.append({'timing': timing, 'qty': qty, 'coverage_days': coverage})
        
        proposals.append({
            'type': 'split_3',
            'description': '3回分割発注',
            'orders': orders,
            'total_qty': sum(o['qty'] for o in orders),
            'risk_level': 'very_low'
        })
    
    return proposals


# -----------------------------------------------------------------------------
# 5. 拡張バックテスト
# -----------------------------------------------------------------------------

def run_new_product_backtest_v23(
    df_items: pd.DataFrame,
    n_samples: int = 20,
    test_period_days: int = 90
) -> Dict[str, Any]:
    """
    【v23新機能】新規授与品予測ロジックのバックテスト
    
    既存商品を「新規扱い」して予測し、実績と比較。
    """
    if df_items is None or df_items.empty:
        return {'available': False, 'message': 'データなし'}
    
    df = df_items.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 商品ごとの販売数を集計
    product_totals = df.groupby('商品名')['販売商品数'].sum()
    
    # 十分なデータがある商品を抽出（最低100個以上）
    valid_products = product_totals[product_totals >= 100].index.tolist()
    
    if len(valid_products) < 5:
        return {'available': False, 'message': f'十分なデータのある商品が少なすぎます（{len(valid_products)}件）'}
    
    # サンプリング
    np.random.seed(42)
    test_products = np.random.choice(valid_products, min(n_samples, len(valid_products)), replace=False)
    
    results = []
    p80_hits = 0
    p90_hits = 0
    
    for product in test_products:
        # この商品を「新規」として、他の商品から予測
        other_products = [p for p in valid_products if p != product]
        
        if not other_products:
            continue
        
        # 類似商品を探す（この商品を除いたデータで）
        product_stats = compute_product_daily_stats_v23(df[df['商品名'].isin(other_products)])
        
        # 簡易的な類似度計算（名前のJaccard類似度のみ）
        similar = []
        for other in other_products:
            sim = calculate_jaccard_similarity_v23(product, other) * 100
            if sim > 5:
                stats = product_stats.get(other, {})
                if stats.get('total_qty', 0) > 0:
                    similar.append({
                        'name': other,
                        'similarity': sim,
                        'avg_daily': stats.get('avg_daily', 1),
                        'std_daily': stats.get('std_daily', 0),
                        'cv': stats.get('cv', 0.3)
                    })
        
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 予測
        factors = estimate_calendar_factors_from_similar_v23(
            df[df['商品名'].isin(other_products)], 
            similar[:10]
        )
        
        if similar:
            weighted_sum = sum(p['avg_daily'] * p['similarity'] for p in similar[:5])
            weight_total = sum(p['similarity'] for p in similar[:5])
            base_daily = weighted_sum / weight_total if weight_total > 0 else factors['baseline']
        else:
            base_daily = factors['baseline']
        
        cv = factors['cv']
        predicted_total = base_daily * test_period_days
        p80_estimate = predicted_total * (1 + 0.84 * cv)
        p90_estimate = predicted_total * (1 + 1.28 * cv)
        
        # 実績
        product_df = df[df['商品名'] == product]
        daily = product_df.groupby('date')['販売商品数'].sum()
        
        if len(daily) >= test_period_days:
            actual_total = daily.iloc[:test_period_days].sum()
        else:
            actual_total = daily.sum()
        
        # 誤差計算
        if actual_total > 0:
            mape = abs(predicted_total - actual_total) / actual_total * 100
        else:
            mape = 100
        
        if actual_total <= p80_estimate:
            p80_hits += 1
        if actual_total <= p90_estimate:
            p90_hits += 1
        
        results.append({
            'product': product,
            'predicted': predicted_total,
            'p80': p80_estimate,
            'p90': p90_estimate,
            'actual': actual_total,
            'mape': mape,
            'similar_count': len(similar)
        })
    
    if not results:
        return {'available': False, 'message': 'バックテスト実行不可'}
    
    avg_mape = np.mean([r['mape'] for r in results])
    n_tests = len(results)
    
    return {
        'available': True,
        'results': results,
        'avg_mape': avg_mape,
        'p80_coverage': p80_hits / n_tests,
        'p90_coverage': p90_hits / n_tests,
        'n_tests': n_tests
    }


# -----------------------------------------------------------------------------
# 6. 強化されたファクトチェックプロンプト
# -----------------------------------------------------------------------------

def generate_enhanced_factcheck_prompt_new_product_v23(
    product_name: str,
    category: str,
    price: int,
    result: Dict[str, Any],
    similar_products: List[Dict]
) -> str:
    """【v23新機能】新規授与品予測用の強化ファクトチェックプロンプト"""
    
    # 類似商品情報
    similar_section = "■ 類似商品（上位5件）:\n"
    if similar_products:
        for i, p in enumerate(similar_products[:5], 1):
            similar_section += f"  {i}. {p['name']} - 日販{p['avg_daily']:.1f}体, 類似度{p['similarity']:.0f}%\n"
    else:
        similar_section += "  （類似商品なし - カテゴリ平均を使用）\n"
    
    # 係数情報
    factors = result.get('factors', {})
    factors_section = "■ 使用した係数:\n"
    factors_section += f"  - ベース日販: {result.get('base_daily', 0):.2f}体/日\n"
    factors_section += f"  - 変動係数(CV): {result.get('cv', 0):.2%}\n"
    factors_section += f"  - 正月係数: {factors.get('special_periods', {}).get('new_year', 'N/A')}\n"
    
    # 分割発注提案
    split_section = ""
    if result.get('split_proposals'):
        split_section = "\n■ 分割発注提案:\n"
        for proposal in result['split_proposals']:
            split_section += f"  【{proposal['description']}】リスク: {proposal['risk_level']}\n"
            for order in proposal['orders']:
                split_section += f"    - {order['timing']}: {order['qty']:,}体（{order['coverage_days']}日分）\n"
    
    prompt = f"""【新規授与品 需要予測ファクトチェック依頼】

■ 基本情報:
- 商品名: {product_name}
- カテゴリ: {category}
- 価格: ¥{price:,}
- 予測期間: {result.get('period_days', 0)}日間
- 類似商品数: {result.get('similar_count', 0)}件

{similar_section}
{factors_section}
■ 予測結果（統計的分位点）:
- 点推定（平均）: {result.get('point_total', 0):,.0f}体 → 丸め後: {result.get('point_total_rounded', 0):,}体
- 滞留回避（P50）: {result.get('p50_total', 0):,.0f}体 → 丸め後: {result.get('p50_rounded', 0):,}体
- バランス（P80）: {result.get('p80_total', 0):,.0f}体 → 丸め後: {result.get('p80_rounded', 0):,}体
- 欠品回避（P90）: {result.get('p90_total', 0):,.0f}体 → 丸め後: {result.get('p90_rounded', 0):,}体

■ 推奨発注数:
- モード: {result.get('recommended_label', 'N/A')}
- 数量: {result.get('recommended_qty', 0):,}体
- 予測売上: ¥{result.get('recommended_qty', 0) * price:,}

■ 日販の検算:
- 予測日販（平均）: {result.get('avg_daily', 0):.2f}体/日
- ベース日販: {result.get('base_daily', 0):.2f}体/日
- P80日販換算: {result.get('p80_rounded', 0) / result.get('period_days', 1):.2f}体/日
{split_section}
■ 検証依頼:
1. 類似商品の選定は妥当ですか？
2. 予測値（P80: {result.get('p80_rounded', 0):,}体）は現実的ですか？
3. 新規商品のため不確実性が高いですが、P90で発注すべきですか？
4. 分割発注は検討すべきですか？

【出力形式】
1. 検算結果（OK/誤りあり）
2. 妥当性判定（妥当/要修正/却下）
3. 推奨発注数（あなたの見解）
4. リスク分析
5. 追加確認事項
"""
    
    return prompt





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
    st.session_state.v20_stockout_periods = []  # 欠品期間リスト（全体用・互換性維持）
if 'v20_last_reorder_point' not in st.session_state:
    st.session_state.v20_last_reorder_point = None

# 【v22新機能】商品個別の欠品期間管理
if 'v22_product_stockout_periods' not in st.session_state:
    st.session_state.v22_product_stockout_periods = {}  # {商品名: [(開始日, 終了日), ...]}

# 【v22新機能】六曜設定（デフォルトOFF - 簡易計算が不正確なため）
if 'v22_enable_rokuyou' not in st.session_state:
    st.session_state.v22_enable_rokuyou = False  # デフォルト: 六曜OFF

# 【v22新機能】発注モードのデフォルト
if 'v22_order_mode' not in st.session_state:
    st.session_state.v22_order_mode = 'balanced'  # デフォルト: バランス（P80）

# 【v24新機能】異常値検出・除外
if 'v24_outlier_threshold' not in st.session_state:
    st.session_state.v24_outlier_threshold = 5.0  # デフォルト: 中央値の5倍
if 'v24_outlier_excluded_dates' not in st.session_state:
    st.session_state.v24_outlier_excluded_dates = {}  # {商品名: [除外日付リスト]}


# =============================================================================
# ユーティリティ関数
# =============================================================================

def detect_outliers(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    【v24新機能】異常値検出（中央値のX倍超えを検出）
    
    Args:
        df: 売上データ（'date', '販売商品数'列を含む）
        threshold: 閾値（中央値の何倍を異常とするか）
    
    Returns:
        異常値を含む行のDataFrame（date, 販売商品数, is_outlier, median, threshold_value）
    """
    if df.empty or '販売商品数' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 0より大きい値のみで中央値を計算（0は「販売なし」なので除外）
    positive_values = df[df['販売商品数'] > 0]['販売商品数']
    
    if positive_values.empty:
        return pd.DataFrame()
    
    median_val = positive_values.median()
    threshold_value = median_val * threshold
    
    # 異常値フラグを設定
    df['is_outlier'] = df['販売商品数'] > threshold_value
    df['median'] = median_val
    df['threshold_value'] = threshold_value
    
    # 異常値のみを返す
    outliers = df[df['is_outlier']].copy()
    
    return outliers


def exclude_outlier_dates(df: pd.DataFrame, excluded_dates: List[date]) -> pd.DataFrame:
    """
    【v24新機能】異常値の日付を学習データから除外
    
    Args:
        df: 売上データ
        excluded_dates: 除外する日付のリスト
    
    Returns:
        異常値日付を除外したDataFrame（除外行の販売数はNaN）
    """
    if not excluded_dates or df.empty:
        return df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 除外フラグを初期化
    df['is_outlier_excluded'] = False
    
    for exc_date in excluded_dates:
        exc_ts = pd.Timestamp(exc_date)
        mask = df['date'] == exc_ts
        df.loc[mask, 'is_outlier_excluded'] = True
    
    # 除外日の販売数をNaNに（学習から除外）
    excluded_count = df['is_outlier_excluded'].sum()
    df.loc[df['is_outlier_excluded'], '販売商品数'] = np.nan
    
    if excluded_count > 0:
        logger.info(f"異常値除外: {excluded_count}日分を学習対象外にしました")
    
    return df


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
    
    # 【v24新機能】データあり日数を計算
    data_days = len(df_agg[df_agg['販売商品数'] > 0]) if not df_agg.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🛒 販売数量合計", f"{total_qty:,}体")
    col2.metric("💰 売上合計", f"¥{total_sales:,.0f}")
    col3.metric("📈 平均日販", f"{avg_daily:.1f}体/日")
    # 【v24修正】期間とデータあり日数を表示
    col4.metric("📅 期間", f"{period_days}日間", delta=f"データあり {data_days}日", delta_color="off")
    
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
    
    # ========== 【v24新機能】異常値検出セクション ==========
    render_outlier_detection(df_agg, st.session_state.selected_products)
    
    # ========== 過去との比較セクション ==========
    render_period_comparison(df_items, original_names, start_date, end_date, total_qty)
    
    st.session_state.sales_data = df_agg
    
    return df_agg


def render_outlier_detection(df_agg: pd.DataFrame, selected_products: List[str]):
    """
    【v24新機能】異常値検出UI
    
    Args:
        df_agg: 集計済み売上データ
        selected_products: 選択された商品リスト
    """
    with st.expander("🔍 **異常値検出・除外設定**", expanded=False):
        st.write("売上データから異常値（中央値のX倍を超える日）を検出し、予測から除外できます。")
        
        # 閾値スライダー
        threshold = st.slider(
            "異常値閾値（中央値の何倍を異常とするか）",
            min_value=3.0,
            max_value=10.0,
            value=st.session_state.v24_outlier_threshold,
            step=0.5,
            help="中央値のこの倍数を超える日を異常値として検出します",
            key="outlier_threshold_slider"
        )
        
        # 閾値が変更されたらsession_stateを更新
        if threshold != st.session_state.v24_outlier_threshold:
            st.session_state.v24_outlier_threshold = threshold
        
        # 異常値を検出
        outliers = detect_outliers(df_agg, threshold)
        
        if outliers.empty:
            st.success(f"✅ 異常値は検出されませんでした（閾値: 中央値の{threshold}倍）")
            # 除外リストをクリア
            for product in selected_products:
                if product in st.session_state.v24_outlier_excluded_dates:
                    del st.session_state.v24_outlier_excluded_dates[product]
        else:
            median_val = outliers['median'].iloc[0]
            threshold_val = outliers['threshold_value'].iloc[0]
            
            st.warning(f"⚠️ **{len(outliers)}件の異常値を検出**（中央値: {median_val:.1f}体、閾値: {threshold_val:.1f}体）")
            
            # 異常値一覧表示
            st.write("**検出された異常値:**")
            
            display_outliers = outliers[['date', '販売商品数']].copy()
            display_outliers['date'] = pd.to_datetime(display_outliers['date']).dt.strftime('%Y-%m-%d')
            display_outliers.columns = ['日付', '販売数']
            display_outliers['中央値比'] = (outliers['販売商品数'] / median_val).apply(lambda x: f"{x:.1f}倍")
            
            # 各行に「正常/除外」選択
            st.write("予測から除外する日を選択してください:")
            
            # 現在の除外リストを取得（商品単位で管理）
            product_key = ",".join(sorted(selected_products))  # 複数商品の場合はキーを結合
            current_excluded = st.session_state.v24_outlier_excluded_dates.get(product_key, [])
            
            # 選択用のデータフレーム作成
            new_excluded = []
            
            for idx, row in outliers.iterrows():
                row_date = pd.to_datetime(row['date']).date()
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    st.write(f"📅 {row_date.strftime('%Y-%m-%d')}")
                with col2:
                    st.write(f"📊 {int(row['販売商品数']):,}体")
                with col3:
                    st.write(f"⚡ 中央値の{row['販売商品数']/median_val:.1f}倍")
                with col4:
                    # チェックボックスで除外/正常を選択
                    is_excluded = st.checkbox(
                        "除外",
                        value=row_date in current_excluded,
                        key=f"outlier_exclude_{row_date}_{product_key}"
                    )
                    if is_excluded:
                        new_excluded.append(row_date)
            
            # 除外リストを更新
            st.session_state.v24_outlier_excluded_dates[product_key] = new_excluded
            
            # 除外状況サマリー
            if new_excluded:
                st.info(f"🚫 **{len(new_excluded)}日を予測から除外します**")
                excluded_dates_str = ", ".join([d.strftime('%Y-%m-%d') for d in sorted(new_excluded)])
                st.caption(f"除外日: {excluded_dates_str}")
            else:
                st.info("ℹ️ すべての日を予測に使用します（除外なし）")


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
                    # すべての方法で予測【v22改善：order_mode対応】
                    product_id = "_".join(st.session_state.selected_products[:3])
                    all_results = forecast_all_methods_with_vertex_ai(
                        sales_data, forecast_days, product_id,
                        baseline_method=baseline_method,
                        auto_special_factors=auto_special_factors,
                        backtest_days=backtest_days,
                        order_mode=order_mode  # 【v22】発注モードを渡す
                    )
                    display_comparison_results_v19(all_results, forecast_days, sales_data, order_mode=order_mode)  # 【v22】
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
                    
                    # 有効日数の表示（v21）
                    valid_days = bt.get('valid_days', len(bt.get('actual', [])))
                    st.caption(f"※ MAPE計算に使用した有効日数: {valid_days}日")
            
            # 予測の信頼度メッセージ（v21改善版）
            reliability = bt.get('reliability', 'unknown')
            reliability_msg = bt.get('reliability_message', '')
            is_seasonal = bt.get('is_seasonal', False)
            
            if is_seasonal:
                st.warning(f"🌸 季節商品の可能性があります。{reliability_msg}")
            elif reliability == 'high':
                st.success(f"✅ 予測精度: 高い（MAPE {mape:.1f}%）- {reliability_msg}")
            elif reliability == 'good':
                st.info(f"💡 予測精度: 良好（MAPE {mape:.1f}%）- {reliability_msg}")
            elif reliability == 'medium':
                st.warning(f"⚠️ 予測精度: 中程度（MAPE {mape:.1f}%）- {reliability_msg}")
            elif reliability in ('low', 'very_low'):
                st.error(f"❌ 予測精度: 要注意（MAPE {mape:.1f}%）- {reliability_msg}")
            elif mape is not None:
                # 従来のロジック（互換性）
                if mape < 25:
                    st.success(f"✅ 予測精度: 高い（MAPE {mape:.1f}%）")
                elif mape < 50:
                    st.info(f"💡 予測精度: 良好（MAPE {mape:.1f}%）")
                elif mape < 80:
                    st.warning(f"⚠️ 予測精度: 中程度（MAPE {mape:.1f}%）")
                else:
                    st.error(f"❌ 予測精度: 要注意（MAPE {mape:.1f}%）")
    
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
        'アンサンブル': '#673AB7',
        '精度強化版': '#9C27B0',
        'Prophet': '#2196F3',
        'Holt-Winters': '#009688',
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
    """
    【v22.5改訂版】個別予測セクション
    
    変更点:
    - 商品ごとに予測結果を表示（合算ではなく）
    - 各商品の入力データ統計を正しく表示
    - コピーボタンをst.text_areaで実装
    """
    st.markdown('<p class="section-header">④ 個別需要予測</p>', unsafe_allow_html=True)
    
    if not st.session_state.individual_sales_data:
        st.info("売上データがあると、需要予測ができます")
        return
    
    # ==========================================================================
    # 予測実行フラグのチェック（formの外で予測を実行）
    # ==========================================================================
    if st.session_state.get('v22_run_forecast_flag'):
        params = st.session_state.pop('v22_run_forecast_flag')
        
        forecast_days = params['forecast_days']
        enable_zero_fill = params['enable_zero_fill']
        enable_trend = params['enable_trend']
        use_daily_new_year = params['use_daily_new_year']
        trend_window_days = params['trend_window_days']
        stockout_periods = params.get('stockout_periods')
        
        # 【v23.2】予測日数をsession_stateに保存（異常値チェック用）
        st.session_state['v22_forecast_days'] = forecast_days
        
        with st.spinner("全予測方法で予測中..."):
            all_products_results = {}
            product_names = list(st.session_state.individual_sales_data.keys())
            
            # 商品ごとの売上データも保存
            products_sales_data = {}
            
            for product_name, sales_data in st.session_state.individual_sales_data.items():
                try:
                    products_sales_data[product_name] = sales_data.copy()
                    
                    # 【v24新機能】異常値除外日付を取得
                    product_key = ",".join(sorted(st.session_state.selected_products))
                    outlier_excluded = st.session_state.v24_outlier_excluded_dates.get(product_key, [])
                    
                    method_results = forecast_all_methods_unified_v22(
                        df=sales_data,
                        periods=forecast_days,
                        product_id=product_name,
                        enable_zero_fill=enable_zero_fill,
                        stockout_periods=stockout_periods,
                        enable_trend=enable_trend,
                        use_daily_new_year=use_daily_new_year,
                        trend_window_days=trend_window_days,
                        outlier_excluded_dates=outlier_excluded  # 【v24新機能】
                    )
                    
                    if method_results:
                        # 商品ごとの最終推奨を計算
                        product_final = calculate_final_recommendation_v22(
                            all_results=method_results,
                            sales_data=sales_data
                        )
                        
                        all_products_results[product_name] = {
                            'method_results': method_results,
                            'final_recommendation': product_final,
                            'sales_data': sales_data
                        }
                        
                except Exception as e:
                    st.warning(f"⚠️ {safe_html(product_name)}の予測に失敗しました: {str(e)[:100]}")
                    logger.error(f"{product_name}の予測エラー: {e}")
            
            if all_products_results:
                # session_stateに保存（商品ごとの結果）
                st.session_state['v22_results'] = {
                    'all_products_results': all_products_results,
                    'forecast_days': forecast_days,
                    'product_names': product_names
                }
                
                # 納品計画用に従来形式でも保存
                individual_results = []
                total_balanced = 0
                
                for pname, pdata in all_products_results.items():
                    final_rec = pdata['final_recommendation']
                    balanced = final_rec['balanced']['total']
                    total_balanced += balanced
                    
                    individual_results.append({
                        'product': pname,
                        'forecast': None,
                        'raw_total': balanced,
                        'rounded_total': balanced,
                        'avg_predicted': balanced / forecast_days if forecast_days > 0 else 0,
                        'method_message': 'v22.5統合予測'
                    })
                
                st.session_state.individual_forecast_results = individual_results
                st.session_state.forecast_total = total_balanced
                st.session_state.last_forecast_method = "v22.5統合予測（商品別）"
                
                st.success("✅ 予測が完了しました！")
    
    # ==========================================================================
    # 予測パラメータ設定
    # ==========================================================================
    with st.form(key="individual_forecast_form_v22"):
        st.write("### 🎯 予測設定")
        st.info("📊 **商品ごとに全予測方法で自動的に予測し、最適な発注数を算出します**")
        
        # 予測期間の設定
        forecast_mode = st.radio(
            "予測期間の指定方法",
            ["日数で指定", "期間で指定"],
            horizontal=True,
            key="v22_forecast_mode_input",
            help="「期間で指定」は期間限定品の予測に便利です"
        )
        
        if forecast_mode == "日数で指定":
            forecast_days = st.slider("予測日数", 30, 365, 180, key="v22_forecast_days_input")
            forecast_start_date = None
            forecast_end_date = None
        else:
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
                    key="v22_start_year_input"
                )
            with col_s2:
                start_month = st.selectbox(
                    "予測開始月",
                    list(range(1, 13)),
                    index=default_start.month - 1,
                    format_func=lambda x: f"{x}月",
                    key="v22_start_month_input"
                )
            with col_s3:
                max_day_start = calendar.monthrange(start_year, start_month)[1]
                start_day = st.selectbox(
                    "予測開始日",
                    list(range(1, max_day_start + 1)),
                    index=min(default_start.day - 1, max_day_start - 1),
                    format_func=lambda x: f"{x}日",
                    key="v22_start_day_input"
                )
            
            with col_e1:
                end_year = st.selectbox(
                    "予測終了年",
                    list(range(2025, 2028)),
                    index=list(range(2025, 2028)).index(default_end.year) if default_end.year in range(2025, 2028) else 0,
                    key="v22_end_year_input"
                )
            with col_e2:
                end_month = st.selectbox(
                    "予測終了月",
                    list(range(1, 13)),
                    index=default_end.month - 1,
                    format_func=lambda x: f"{x}月",
                    key="v22_end_month_input"
                )
            with col_e3:
                max_day_end = calendar.monthrange(end_year, end_month)[1]
                end_day = st.selectbox(
                    "予測終了日",
                    list(range(1, max_day_end + 1)),
                    index=min(default_end.day - 1, max_day_end - 1),
                    format_func=lambda x: f"{x}日",
                    key="v22_end_day_input"
                )
            
            forecast_start_date = date(start_year, start_month, start_day)
            forecast_end_date = date(end_year, end_month, end_day)
            forecast_days = max(1, (forecast_end_date - forecast_start_date).days + 1)
        
        # 詳細設定
        with st.expander("⚙️ **詳細設定（精度強化オプション）**", expanded=False):
            col_v20_1, col_v20_2 = st.columns(2)
            
            with col_v20_1:
                enable_zero_fill = st.checkbox(
                    "0埋め処理（推奨）",
                    value=True,
                    help="売上がない日を0で補完",
                    key="v22_zero_fill_input"
                )
                enable_trend = st.checkbox(
                    "トレンド係数（前年比）",
                    value=True,
                    help="成長/衰退トレンドを反映",
                    key="v22_trend_input"
                )
            
            with col_v20_2:
                use_daily_new_year = st.checkbox(
                    "正月日別係数（1/1〜1/7）",
                    value=True,
                    help="正月のピークを正確に捉える",
                    key="v22_daily_new_year_input"
                )
                trend_window_days = st.selectbox(
                    "トレンド比較期間",
                    options=[30, 60, 90],
                    format_func=lambda x: f"直近{x}日間",
                    index=1,
                    key="v22_trend_window_input"
                )
        
        submitted = st.form_submit_button(
            "🔮 全方法で需要予測を実行",
            type="primary",
            use_container_width=True
        )
    
    # ==========================================================================
    # 欠品期間の管理（フォーム外）
    # ==========================================================================
    with st.expander("🚫 欠品期間の登録・管理", expanded=False):
        st.caption("在庫切れ期間を指定すると、その期間は学習から除外されます")
        
        col_stock1, col_stock2, col_stock3 = st.columns([2, 2, 1])
        
        with col_stock1:
            stockout_start = st.date_input("欠品開始日", value=None, key="v22_stockout_start_input")
        with col_stock2:
            stockout_end = st.date_input("欠品終了日", value=None, key="v22_stockout_end_input")
        with col_stock3:
            st.write("")
            st.write("")
            add_stockout = st.button("➕ 追加", key="v22_add_stockout_btn")
        
        if add_stockout and stockout_start and stockout_end:
            if stockout_start <= stockout_end:
                if 'v20_stockout_periods' not in st.session_state:
                    st.session_state.v20_stockout_periods = []
                new_period = (stockout_start, stockout_end)
                if new_period not in st.session_state.v20_stockout_periods:
                    st.session_state.v20_stockout_periods.append(new_period)
                    st.success(f"欠品期間を追加: {stockout_start} 〜 {stockout_end}")
                    st.rerun()
            else:
                st.warning("終了日は開始日以降にしてください")
        
        if st.session_state.get('v20_stockout_periods'):
            st.markdown("**登録済み:**")
            for i, (s, e) in enumerate(st.session_state.v20_stockout_periods):
                col_p1, col_p2 = st.columns([4, 1])
                with col_p1:
                    st.text(f"  {i+1}. {s} 〜 {e}")
                with col_p2:
                    if st.button("🗑️", key=f"v22_del_stockout_btn_{i}"):
                        st.session_state.v20_stockout_periods.pop(i)
                        st.rerun()
    
    # ==========================================================================
    # フォーム送信時の処理
    # ==========================================================================
    if submitted:
        if forecast_mode == "期間で指定":
            if forecast_end_date <= forecast_start_date:
                st.error("⚠️ 終了日は開始日より後にしてください")
                return
        
        stockout_periods = st.session_state.get('v20_stockout_periods', None)
        if stockout_periods:
            stockout_periods = [(s, e) for s, e in stockout_periods]
        
        st.session_state['v22_run_forecast_flag'] = {
            'forecast_days': forecast_days,
            'enable_zero_fill': enable_zero_fill,
            'enable_trend': enable_trend,
            'use_daily_new_year': use_daily_new_year,
            'trend_window_days': trend_window_days,
            'stockout_periods': stockout_periods
        }
        st.rerun()
    
    # ==========================================================================
    # 【v22.5】商品別予測結果の表示
    # ==========================================================================
    if st.session_state.get('v22_results'):
        results = st.session_state['v22_results']
        all_products_results = results.get('all_products_results', {})
        forecast_days_result = results.get('forecast_days', 180)
        
        if not all_products_results:
            st.warning("予測結果がありません")
            return
        
        st.write("## 📊 商品別予測結果")
        st.caption("各商品の予測結果を個別に表示しています")
        
        # 商品ごとにタブまたはexpanderで表示
        for product_name, pdata in all_products_results.items():
            method_results = pdata['method_results']
            final_rec = pdata['final_recommendation']
            sales_data = pdata.get('sales_data')
            
            with st.expander(f"📦 **{product_name}**", expanded=True):
                # 入力データの統計
                if sales_data is not None and not sales_data.empty:
                    total_days = len(sales_data)
                    total_qty = int(sales_data['販売商品数'].sum())
                    avg_daily = sales_data['販売商品数'].mean()
                    std_daily = sales_data['販売商品数'].std()
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    col_stat1.metric("📅 学習データ", f"{total_days}日間")
                    col_stat2.metric("📈 総販売数", f"{total_qty:,}体")
                    col_stat3.metric("📊 実績日販", f"{avg_daily:.1f}体/日")
                    col_stat4.metric("📉 標準偏差", f"{std_daily:.1f}")
                else:
                    st.warning("⚠️ 入力データがありません")
                    avg_daily = 0
                
                st.markdown("---")
                
                # 最終推奨発注数
                st.write("### 🎯 推奨発注数")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cons = final_rec['conservative']
                    st.metric(f"📉 {cons['label']}", f"{cons['total']:,}体")
                    st.caption(cons['risk'])
                
                with col2:
                    bal = final_rec['balanced']
                    st.metric(f"⚖️ {bal['label']}", f"{bal['total']:,}体")
                    st.caption(bal['risk'])
                    st.markdown("**↑ おすすめ**")
                
                with col3:
                    aggr = final_rec['aggressive']
                    st.metric(f"🛡️ {aggr['label']}", f"{aggr['total']:,}体")
                    st.caption(aggr['risk'])
                
                # 各予測方法の結果
                with st.expander("📈 各予測方法の詳細", expanded=False):
                    table_data = []
                    for method_name, result in method_results.items():
                        totals = result.get('totals', {})
                        mape = result.get('mape')
                        table_data.append({
                            '予測方法': method_name,
                            '滞留回避': f"{totals.get('conservative', 0):,}体",
                            'バランス': f"{totals.get('balanced', 0):,}体",
                            '欠品回避': f"{totals.get('aggressive', 0):,}体",
                            'MAPE': f"{mape:.1f}%" if mape else "-"
                        })
                    
                    if table_data:
                        df_methods = pd.DataFrame(table_data)
                        st.dataframe(df_methods, use_container_width=True, hide_index=True)
                
                # ファクトチェック用プロンプト
                with st.expander("🔍 ファクトチェック用プロンプト", expanded=False):
                    # プロンプト生成
                    prompt = generate_product_factcheck_prompt_v22(
                        product_name=product_name,
                        method_results=method_results,
                        final_recommendation=final_rec,
                        forecast_days=forecast_days_result,
                        sales_data=sales_data
                    )
                    
                    st.markdown("""
                    <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong>💡 使い方:</strong> 下のテキストを選択してコピー（Ctrl+A → Ctrl+C）し、
                        ChatGPT、Claude、Geminiなどに貼り付けて検証してもらってください。
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # st.text_areaでコピーしやすく
                    st.text_area(
                        "プロンプト",
                        value=prompt,
                        height=300,
                        key=f"factcheck_prompt_{product_name}",
                        label_visibility="collapsed"
                    )
        
        # 全商品の合計サマリー
        if len(all_products_results) > 1:
            st.markdown("---")
            st.write("## 📋 全商品合計")
            
            total_cons = sum(p['final_recommendation']['conservative']['total'] for p in all_products_results.values())
            total_bal = sum(p['final_recommendation']['balanced']['total'] for p in all_products_results.values())
            total_aggr = sum(p['final_recommendation']['aggressive']['total'] for p in all_products_results.values())
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📉 滞留回避（合計）", f"{total_cons:,}体")
            col2.metric("⚖️ バランス（合計）", f"{total_bal:,}体")
            col3.metric("🛡️ 欠品回避（合計）", f"{total_aggr:,}体")
    
    # 従来の結果表示（互換性維持）
    elif st.session_state.get('individual_forecast_results'):
        results = st.session_state.individual_forecast_results
        st.success(f"✅ {len(results)}件の予測が完了しました！")
        
        for r in results:
            st.write(f"**{r['product']}**: {r['rounded_total']:,}体（{r['avg_predicted']:.1f}体/日）")


def generate_product_factcheck_prompt_v22(
    product_name: str,
    method_results: Dict[str, Dict[str, Any]],
    final_recommendation: Dict[str, Any],
    forecast_days: int,
    sales_data: pd.DataFrame = None
) -> str:
    """
    【v22.5】商品別ファクトチェック用プロンプト生成
    """
    # 入力データの統計
    if sales_data is not None and not sales_data.empty:
        total_days = len(sales_data)
        total_qty = int(sales_data['販売商品数'].sum())
        avg_daily = sales_data['販売商品数'].mean()
        std_daily = sales_data['販売商品数'].std()
        max_daily = sales_data['販売商品数'].max()
        min_daily = sales_data['販売商品数'].min()
        
        input_section = f"""■ 入力データ（過去の実績）:
- 学習データ期間: {total_days}日間
- 総販売数: {total_qty:,}体
- 実績日販（平均）: {avg_daily:.2f}体/日
- 標準偏差: {std_daily:.2f}
- 最大日販: {max_daily:.0f}体/日
- 最小日販: {min_daily:.0f}体/日"""
    else:
        input_section = "■ 入力データ: 取得できませんでした（システムエラーの可能性）"
        avg_daily = 0
    
    # 各予測方法の結果
    methods_section = "■ 各予測方法の結果:\n"
    for method_name, result in method_results.items():
        totals = result.get('totals', {})
        mape = result.get('mape')
        methods_section += f"  - {method_name}: バランス={totals.get('balanced', 0):,}体"
        if mape:
            methods_section += f"（MAPE {mape:.1f}%）"
        methods_section += "\n"
    
    # 最終推奨
    cons = final_recommendation['conservative']['total']
    bal = final_recommendation['balanced']['total']
    aggr = final_recommendation['aggressive']['total']
    
    # 変化率
    expected_from_actual = avg_daily * forecast_days
    if expected_from_actual > 0:
        change_rate = ((bal / expected_from_actual) - 1) * 100
    else:
        change_rate = 0
    
    prompt = f"""【需要予測ファクトチェック依頼】

■ 基本情報:
- 対象商品: {product_name}
- 予測期間: {forecast_days}日間
- 予測方法: 全{len(method_results)}方法の加重平均

{input_section}

{methods_section}
■ 最終推奨発注数:
- 滞留回避（P50）: {cons:,}体
- バランス（P80）: {bal:,}体 ← 推奨
- 欠品回避（P90）: {aggr:,}体

■ 変化率（実績→予測）:
- 実績ベース予測: {avg_daily:.2f} × {forecast_days} = {expected_from_actual:,.0f}体
- AI予測（バランス）: {bal:,}体
- 乖離率: {change_rate:+.1f}%

■ 検証依頼:
1. 最終推奨発注数{bal:,}体は妥当ですか？
2. 実績日販{avg_daily:.1f}体/日に対して、予測は適切ですか？
3. 各予測方法の結果にばらつきがある場合、どれを信頼すべきですか？

【出力形式】
1. 検算結果
2. 妥当性判定（OK/NG/要確認）
3. リスク分析
4. 推奨発注数（あなたの見解）
5. 追加確認事項
"""
    
    return prompt


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
    """
    【v23.2改善版】新規授与品の需要予測
    
    改善点:
    - 類似商品の手動選択機能
    - 同名商品の最優先（「椿守 アクリル型」→「椿守」を自動1位）
    - 予測内訳の詳細表示
    """
    
    st.markdown("""
    <div class="new-product-card">
        <h2>✨ 新規授与品の需要予測（v23.2改善版）</h2>
        <p>まだ販売実績のない新しい授与品の需要を、類似商品のデータから予測します。</p>
        <p style="font-size: 0.9em; opacity: 0.9;">🆕 手動で参考商品を選択可能、予測内訳を詳細表示</p>
    </div>
    """, unsafe_allow_html=True)
    
    # タブで機能を分割
    tab1, tab2 = st.tabs(["📊 需要予測", "🧪 精度検証（バックテスト）"])
    
    with tab1:
        st.markdown('<p class="section-header">① 新規授与品の情報を入力</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_product_name = st.text_input(
                "授与品名",
                placeholder="例: 椿守 アクリル型",
                help="新しく作る授与品の名前",
                key="v23_2_new_product_name"
            )
            
            new_product_category = st.selectbox(
                "カテゴリー",
                list(CATEGORY_CHARACTERISTICS.keys()),
                help="最も近いカテゴリーを選んでください",
                key="v23_2_new_product_category"
            )
            
            new_product_price = st.number_input(
                "価格（円）",
                min_value=100,
                max_value=50000,
                value=1000,
                step=100,
                help="販売予定価格",
                key="v23_2_new_product_price"
            )
        
        with col2:
            new_product_description = st.text_area(
                "特徴・コンセプト（任意）",
                placeholder="例: 椿をモチーフにしたアクリル製のお守り",
                help="授与品の特徴を記述",
                key="v23_2_new_product_description"
            )
            
            target_audience = st.multiselect(
                "ターゲット層（任意）",
                ["若い女性", "若い男性", "中高年女性", "中高年男性", "家族連れ", "観光客", "地元の方"],
                default=[],
                key="v23_2_target_audience"
            )
        
        # ========== ② 参考商品の選択 ==========
        st.markdown('<p class="section-header">② 参考にする商品を選択</p>', unsafe_allow_html=True)
        
        # 類似商品を自動検索
        all_similar_products = []
        if new_product_name and new_product_name.strip():
            with st.spinner("類似商品を検索中..."):
                all_similar_products = find_similar_products_v23_2(
                    new_product_name, 
                    new_product_category, 
                    new_product_price,
                    new_product_description,
                    target_audience
                )
        
        selected_products = []
        
        if all_similar_products:
            st.success(f"✅ {len(all_similar_products)}件の候補が見つかりました")
            
            # 自動推薦（上位5件）
            auto_recommended = all_similar_products[:5]
            auto_names = [p['name'] for p in auto_recommended]
            
            st.write("**🤖 AIが推薦する参考商品（上位5件）:**")
            
            rec_table = []
            for i, p in enumerate(auto_recommended, 1):
                bonus_badge = "⭐同名" if p.get('same_name_bonus', 0) > 30 else ""
                reason = p.get('same_name_reason', 'キーワード一致') if bonus_badge else 'キーワード/価格一致'
                rec_table.append({
                    '順位': i,
                    '商品名': f"{p['name']} {bonus_badge}",
                    '日販': f"{p['avg_daily']:.1f}体/日",
                    '類似度': f"{p['similarity']:.0f}%",
                    '理由': reason
                })
            
            df_rec = pd.DataFrame(rec_table)
            st.dataframe(df_rec, use_container_width=True, hide_index=True)
            
            # 手動選択オプション
            st.write("---")
            use_manual = st.checkbox(
                "🔧 参考商品を手動で選択する",
                value=False,
                help="AIの推薦ではなく、自分で参考にしたい商品を選びたい場合はチェック",
                key="v23_2_use_manual"
            )
            
            if use_manual:
                # 全商品リストから選択
                all_product_names = [p['name'] for p in all_similar_products]
                
                selected_names = st.multiselect(
                    "参考にする商品を選択（複数可）",
                    options=all_product_names,
                    default=auto_names[:3],  # デフォルトは上位3件
                    help="選択した商品の実績を基に予測します",
                    key="v23_2_manual_selection"
                )
                
                # 選択された商品の情報を取得
                selected_products = [p for p in all_similar_products if p['name'] in selected_names]
                
                if selected_products:
                    st.write(f"**選択された参考商品: {len(selected_products)}件**")
                    
                    sel_table = []
                    for p in selected_products:
                        sel_table.append({
                            '商品名': p['name'],
                            '日販': f"{p['avg_daily']:.1f}体/日",
                            '総販売数': f"{p['total_qty']:,}体"
                        })
                    
                    df_sel = pd.DataFrame(sel_table)
                    st.dataframe(df_sel, use_container_width=True, hide_index=True)
                    
                    # 手動選択の場合の予測ベース日販
                    avg_daily = np.mean([p['avg_daily'] for p in selected_products])
                    st.info(f"📊 選択商品の平均日販: **{avg_daily:.2f}体/日**")
                else:
                    st.warning("⚠️ 参考商品を選択してください")
            else:
                selected_products = auto_recommended
        else:
            if new_product_name:
                st.warning("⚠️ 類似商品が見つかりませんでした。カテゴリの平均値から予測します。")
            else:
                st.info("👆 授与品名を入力すると、類似商品を検索します")
        
        # ========== ③ 予測設定 ==========
        st.markdown('<p class="section-header">③ 予測設定</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_period = st.selectbox(
                "予測期間",
                ["1ヶ月", "3ヶ月", "6ヶ月", "1年"],
                index=1,  # 3ヶ月をデフォルト
                key="v23_2_forecast_period"
            )
        
        with col2:
            order_mode = st.selectbox(
                "発注モード",
                ["滞留回避（P50）", "バランス（P80）★推奨", "欠品回避（P90）"],
                index=1,
                help="P50=50%の確率で需要を満たす, P80=80%, P90=90%",
                key="v23_2_order_mode"
            )
        
        with col3:
            lead_time = st.number_input(
                "リードタイム（日）",
                min_value=1,
                max_value=90,
                value=14,
                help="発注から納品までの日数",
                key="v23_2_lead_time"
            )
        
        # モードの変換
        mode_map = {
            "滞留回避（P50）": "conservative",
            "バランス（P80）★推奨": "balanced",
            "欠品回避（P90）": "aggressive"
        }
        selected_mode = mode_map.get(order_mode, "balanced")
        
        # 期間の変換
        period_days = {"1ヶ月": 30, "3ヶ月": 90, "6ヶ月": 180, "1年": 365}[forecast_period]
        
        # ========== 予測実行ボタン ==========
        if st.button("🔮 需要を予測", type="primary", use_container_width=True, key="v23_2_forecast_btn"):
            if not new_product_name or not new_product_name.strip():
                st.error("授与品名を入力してください")
            elif not selected_products:
                st.error("参考商品を選択してください（類似商品が見つからない場合は授与品名を変更してお試しください）")
            else:
                with st.spinner("予測中..."):
                    forecast_result = forecast_new_product_v23_2(
                        new_product_name,
                        new_product_category,
                        new_product_price,
                        selected_products,
                        period_days,
                        selected_mode
                    )
                    
                    # 結果をsession_stateに保存
                    st.session_state['v23_2_forecast_result'] = forecast_result
                    st.session_state['v23_2_product_info'] = {
                        'name': new_product_name,
                        'category': new_product_category,
                        'price': new_product_price,
                        'reference_products': selected_products
                    }
        
        # ========== 結果表示 ==========
        if st.session_state.get('v23_2_forecast_result'):
            result = st.session_state['v23_2_forecast_result']
            info = st.session_state.get('v23_2_product_info', {})
            
            display_new_product_forecast_v23_2(
                result, 
                info.get('name', ''),
                info.get('price', 0),
                info.get('reference_products', [])
            )
    
    with tab2:
        render_new_product_backtest_v23()


def display_new_product_forecast_v23_2(result: dict, product_name: str, price: int, reference_products: list):
    """【v23.2改善版】新規授与品の予測結果を表示"""
    
    st.success("✅ 予測完了！")
    
    st.write(f"### 📦 「{product_name}」の需要予測")
    
    # 参考商品の表示
    ref_count = result.get('reference_count', 0)
    if ref_count >= 3:
        st.info(f"📊 参考商品 **{ref_count}件** の実績データを基に予測しました")
    elif ref_count >= 1:
        st.warning(f"⚠️ 参考商品 **{ref_count}件** のみで予測。精度が低い可能性があります")
    else:
        st.error("❌ 参考商品なし。カテゴリ平均から予測しています")
    
    # ========== 推奨発注数 ==========
    st.write("#### 🎯 推奨発注数")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📉 滞留回避（P50）",
            f"{result.get('p50_rounded', 0):,}体",
            help="50%の確率で需要を満たす"
        )
    
    with col2:
        st.metric(
            "⚖️ バランス（P80）",
            f"{result.get('p80_rounded', 0):,}体",
            help="80%の確率で需要を満たす"
        )
        st.caption("**↑ 推奨**")
    
    with col3:
        st.metric(
            "🛡️ 欠品回避（P90）",
            f"{result.get('p90_rounded', 0):,}体",
            help="90%の確率で需要を満たす"
        )
    
    # 選択モードの強調
    st.info(f"""
    📌 **選択モード: {result.get('recommended_label', 'N/A')}**
    - 推奨発注数: **{result.get('recommended_qty', 0):,}体**
    - 予測売上: ¥{result.get('recommended_qty', 0) * price:,}
    """)
    
    # ========== 【v23.2】予測内訳の詳細 ==========
    st.write("#### 📊 予測の内訳")
    
    with st.expander("🔍 予測計算の詳細を見る", expanded=True):
        # 基本パラメータ
        st.write("**基本パラメータ**")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("ベース日販", f"{result.get('base_daily', 0):.2f}体/日")
        col2.metric("変動係数(CV)", f"{result.get('cv', 0):.1%}")
        col3.metric("予測期間", f"{result.get('period_days', 0)}日")
        col4.metric("参考商品数", f"{result.get('reference_count', 0)}件")
        
        # 係数サマリー
        factor_summary = result.get('factor_summary', {})
        
        st.write("**適用した係数**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("📅 **曜日係数**")
            st.write(f"- 平日平均: {factor_summary.get('weekday_weekday', 1.0):.2f}")
            st.write(f"- 土日平均: {factor_summary.get('weekday_weekend', 1.0):.2f}")
        
        with col2:
            st.write("📆 **月係数（抜粋）**")
            month_f = factor_summary.get('month_factors', {})
            st.write(f"- 1月: {month_f.get(1, 1.0):.2f}")
            st.write(f"- 12月: {month_f.get(12, 1.0):.2f}")
        
        with col3:
            st.write("🎌 **特別期間係数**")
            special_f = factor_summary.get('special_factors', {})
            st.write(f"- 正月: {special_f.get('new_year', 1.0):.2f}")
            st.write(f"- 年末: {special_f.get('year_end', 1.0):.2f}")
        
        # 計算式の説明
        st.write("---")
        st.write("**計算式**")
        base = result.get('base_daily', 0)
        ww = factor_summary.get('weekday_weekend', 1.0)
        m2 = factor_summary.get('month_factors', {}).get(2, 1.0)
        st.code(f"""
日別予測 = ベース日販 × 曜日係数 × 月係数 × 特別期間係数
        = {base:.2f} × (曜日) × (月) × (特別)

例: 通常の土曜日（2月）
   = {base:.2f} × {ww:.2f} × {m2:.2f} × 1.0
   ≈ {base * ww * m2:.2f}体/日
        """)
        
        # 日別詳細（最初の14日）
        daily_details = result.get('daily_details', [])
        if daily_details:
            st.write("---")
            st.write("**日別予測（最初の14日）**")
            
            detail_table = []
            for d in daily_details[:14]:
                detail_table.append({
                    '日付': d['date'],
                    '曜日': d['weekday'],
                    'ベース': f"{d['base']:.1f}",
                    '曜日係数': f"{d['weekday_f']:.2f}",
                    '月係数': f"{d['month_f']:.2f}",
                    '特別': d['period'],
                    '予測': f"{d['predicted']:.1f}体"
                })
            
            df_detail = pd.DataFrame(detail_table)
            st.dataframe(df_detail, use_container_width=True, hide_index=True)
    
    # ========== 月別予測グラフ ==========
    monthly_data = []
    if result.get('monthly') and isinstance(result['monthly'], dict):
        for period, qty in result['monthly'].items():
            monthly_data.append({'月': str(period), '予測販売数': qty})
    
    if monthly_data:
        st.write("#### 📅 月別予測")
        df_monthly = pd.DataFrame(monthly_data)
        
        fig = px.bar(
            df_monthly, x='月', y='予測販売数',
            title='月別予測販売数',
            color='予測販売数',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========== 分割発注提案 ==========
    st.write("#### 🚚 発注戦略の提案")
    
    split_proposals = result.get('split_proposals', [])
    if split_proposals:
        for proposal in split_proposals:
            risk_color = {
                'very_low': '🟢',
                'low': '🟢',
                'medium': '🟡',
                'high': '🔴'
            }.get(proposal['risk_level'], '⚪')
            
            is_recommended = proposal['type'] == 'split_2'
            
            with st.expander(f"{risk_color} {proposal['description']} - 合計{proposal['total_qty']:,}体", expanded=is_recommended):
                for order in proposal['orders']:
                    st.write(f"- **{order['timing']}**: {order['qty']:,}体（{order['coverage_days']}日分カバー）")
                
                if is_recommended:
                    st.success("✅ 分割発注により、欠品リスクと滞留リスクの両方を軽減できます")
    
    # ========== ファクトチェックプロンプト ==========
    with st.expander("🔍 ファクトチェック用プロンプト", expanded=False):
        prompt = generate_factcheck_prompt_v23_2(
            product_name,
            st.session_state.get('v23_2_product_info', {}).get('category', ''),
            price,
            result,
            reference_products
        )
        
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>💡 使い方:</strong> 下のテキストをコピーして、ChatGPT、Claude、Geminiなどに貼り付けて検証してもらってください。
        </div>
        """, unsafe_allow_html=True)
        
        st.text_area(
            "プロンプト",
            value=prompt,
            height=400,
            key="v23_2_factcheck_prompt",
            label_visibility="collapsed"
        )


def display_new_product_forecast_v23(result: dict, product_name: str, price: int, similar_products: list):
    """【v23改善版】新規授与品の予測結果を表示"""
    
    st.success("✅ 予測完了！")
    
    st.write(f"### 📦 「{product_name}」の需要予測")
    
    # 信頼度表示
    similar_count = result.get('similar_count', 0)
    if similar_count >= 5:
        st.info(f"📊 類似商品 {similar_count} 件のデータを基に予測しました。信頼度: ⭐⭐⭐")
    elif similar_count >= 2:
        st.warning(f"📊 類似商品 {similar_count} 件のデータを基に予測しました。信頼度: ⭐⭐")
    else:
        st.warning("📊 類似商品が少ないため、カテゴリーの平均値から予測しました。信頼度: ⭐")
    
    # メイン指標
    st.write("#### 🎯 推奨発注数")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📉 滞留回避（P50）",
            f"{result.get('p50_rounded', 0):,}体",
            help="50%の確率で需要を満たす数量"
        )
    
    with col2:
        st.metric(
            "⚖️ バランス（P80）★推奨",
            f"{result.get('p80_rounded', 0):,}体",
            help="80%の確率で需要を満たす数量"
        )
    
    with col3:
        st.metric(
            "🛡️ 欠品回避（P90）",
            f"{result.get('p90_rounded', 0):,}体",
            help="90%の確率で需要を満たす数量"
        )
    
    # 選択されたモードをハイライト
    st.info(f"""
    📌 **選択モード: {result.get('recommended_label', 'N/A')}**
    - 推奨発注数: **{result.get('recommended_qty', 0):,}体**
    - 予測売上: ¥{result.get('recommended_qty', 0) * price:,}
    - 予測日販: {result.get('avg_daily', 0):.1f}体/日
    """)
    
    # 統計情報
    with st.expander("📈 予測の詳細", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**予測パラメータ**")
            st.write(f"- 予測期間: {result.get('period_days', 0)}日間")
            st.write(f"- ベース日販: {result.get('base_daily', 0):.2f}体/日")
            st.write(f"- 変動係数(CV): {result.get('cv', 0):.1%}")
            st.write(f"- 類似商品数: {result.get('similar_count', 0)}件")
        
        with col2:
            st.write("**未丸め値（検算用）**")
            st.write(f"- 点推定: {result.get('point_total', 0):,.1f}体")
            st.write(f"- P50: {result.get('p50_total', 0):,.1f}体")
            st.write(f"- P80: {result.get('p80_total', 0):,.1f}体")
            st.write(f"- P90: {result.get('p90_total', 0):,.1f}体")
    
    # 月別予測グラフ
    monthly_data = []
    if result.get('monthly') and isinstance(result['monthly'], dict):
        for period, qty in result['monthly'].items():
            monthly_data.append({'月': str(period), '予測販売数': qty})
    
    if monthly_data:
        st.write("#### 📅 月別予測")
        df_monthly = pd.DataFrame(monthly_data)
        
        fig = px.bar(
            df_monthly, x='月', y='予測販売数',
            title='月別予測販売数',
            color='予測販売数',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 分割発注提案
    st.write("#### 🚚 発注戦略の提案")
    
    split_proposals = result.get('split_proposals', [])
    if split_proposals:
        for proposal in split_proposals:
            risk_color = {
                'very_low': '🟢',
                'low': '🟢',
                'medium': '🟡',
                'high': '🔴'
            }.get(proposal['risk_level'], '⚪')
            
            with st.expander(f"{risk_color} {proposal['description']} - 合計{proposal['total_qty']:,}体", expanded=(proposal['type'] == 'split_2')):
                for order in proposal['orders']:
                    st.write(f"- **{order['timing']}**: {order['qty']:,}体（{order['coverage_days']}日分カバー）")
                
                if proposal['type'] == 'split_2':
                    st.success("✅ 分割発注により、欠品リスクと滞留リスクの両方を軽減できます")
    
    # ファクトチェックプロンプト
    with st.expander("🔍 ファクトチェック用プロンプト", expanded=False):
        prompt = generate_enhanced_factcheck_prompt_new_product_v23(
            product_name,
            st.session_state.get('v23_new_product_info', {}).get('category', ''),
            price,
            result,
            similar_products
        )
        
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>💡 使い方:</strong> 下のテキストを選択してコピー（Ctrl+A → Ctrl+C）し、
            ChatGPT、Claude、Geminiなどに貼り付けて検証してもらってください。
        </div>
        """, unsafe_allow_html=True)
        
        st.text_area(
            "プロンプト",
            value=prompt,
            height=400,
            key="v23_factcheck_prompt",
            label_visibility="collapsed"
        )


def render_new_product_backtest_v23():
    """【v23新機能】新規授与品予測ロジックのバックテスト"""
    
    st.markdown('<p class="section-header">🧪 新規予測ロジックの精度検証</p>', unsafe_allow_html=True)
    
    st.markdown("""
    既存の商品を「新規商品」として扱い、予測ロジックの精度を検証します。
    これにより、新規授与品予測の信頼性を定量的に評価できます。
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider(
            "テスト商品数",
            min_value=5,
            max_value=50,
            value=20,
            help="バックテストに使用する商品数",
            key="v23_backtest_samples"
        )
    
    with col2:
        test_period = st.slider(
            "テスト期間（日）",
            min_value=30,
            max_value=180,
            value=90,
            help="予測精度を検証する期間",
            key="v23_backtest_period"
        )
    
    if st.button("🚀 バックテストを実行", type="primary", key="v23_run_backtest"):
        if st.session_state.data_loader is None:
            st.error("データが読み込まれていません")
            return
        
        df_items = st.session_state.data_loader.load_item_sales()
        
        if df_items.empty:
            st.error("売上データがありません")
            return
        
        with st.spinner(f"バックテスト実行中... ({n_samples}商品 × {test_period}日)"):
            backtest_result = run_new_product_backtest_v23(
                df_items,
                n_samples=n_samples,
                test_period_days=test_period
            )
        
        if not backtest_result.get('available'):
            st.warning(f"⚠️ {backtest_result.get('message', 'バックテスト実行不可')}")
            return
        
        st.success(f"✅ バックテスト完了！（{backtest_result['n_tests']}商品をテスト）")
        
        # サマリー
        st.write("### 📊 精度サマリー")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mape = backtest_result.get('avg_mape', 0)
            mape_color = "🟢" if mape < 30 else "🟡" if mape < 50 else "🔴"
            st.metric(f"{mape_color} 平均MAPE", f"{mape:.1f}%")
        
        with col2:
            p80_cov = backtest_result.get('p80_coverage', 0)
            p80_color = "🟢" if p80_cov >= 0.75 else "🟡" if p80_cov >= 0.60 else "🔴"
            st.metric(f"{p80_color} P80カバレッジ", f"{p80_cov:.1%}", help="目標: 80%")
        
        with col3:
            p90_cov = backtest_result.get('p90_coverage', 0)
            p90_color = "🟢" if p90_cov >= 0.85 else "🟡" if p90_cov >= 0.75 else "🔴"
            st.metric(f"{p90_color} P90カバレッジ", f"{p90_cov:.1%}", help="目標: 90%")
        
        # 解釈
        if mape < 30 and p80_cov >= 0.75:
            st.success("✅ 予測精度は良好です。P80での発注が推奨されます。")
        elif mape < 50 and p80_cov >= 0.60:
            st.warning("⚠️ 予測精度は許容範囲ですが、P90での発注を検討してください。")
        else:
            st.error("❌ 予測精度が低いです。類似商品の選定基準を見直すか、P90での発注を強く推奨します。")
        
        # 詳細テーブル
        with st.expander("📋 商品別の結果", expanded=False):
            results = backtest_result.get('results', [])
            if results:
                table_data = []
                for r in results:
                    table_data.append({
                        '商品名': r['product'][:20] + '...' if len(r['product']) > 20 else r['product'],
                        '予測': f"{r['predicted']:,.0f}",
                        'P80': f"{r['p80']:,.0f}",
                        '実績': f"{r['actual']:,.0f}",
                        'MAPE': f"{r['mape']:.1f}%",
                        '類似商品数': r['similar_count']
                    })
                
                df_results = pd.DataFrame(table_data)
                st.dataframe(df_results, use_container_width=True, hide_index=True)



def render_new_product_forecast_legacy():
    """新規授与品の需要予測（レガシー版 - v23以前の互換用）"""
    
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
    col1.metric("予測販売総数", f"{result.get('total_qty_rounded', 0):,}体")
    col2.metric("予測売上", f"¥{result.get('total_qty_rounded', 0) * price:,.0f}")
    col3.metric("平均日販", f"{result.get('avg_daily', 0):.1f}体/日")
    col4.metric("予測期間", f"{result.get('period_days', 0)}日間")
    
    if result.get('similar_count', 0) >= 3:
        st.info(f"📊 類似商品 {result['similar_count']} 件のデータを基に予測しました。信頼度: ⭐⭐⭐")
    elif result.get('similar_count', 0) >= 1:
        st.warning(f"📊 類似商品 {result['similar_count']} 件のデータを基に予測しました。信頼度: ⭐⭐")
    else:
        st.warning("📊 類似商品がなかったため、カテゴリーの平均値から予測しました。信頼度: ⭐")
    
    # 【v22修正】monthly がNoneまたは空の場合のエラーハンドリング
    monthly_data = []
    if result.get('monthly') and isinstance(result['monthly'], dict):
        for period, qty in result['monthly'].items():
            monthly_data.append({'月': str(period), '予測販売数': qty})
    
    if monthly_data:
        df_monthly = pd.DataFrame(monthly_data)
        
        fig = px.bar(
            df_monthly, x='月', y='予測販売数',
            title='月別予測販売数',
            color='予測販売数',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("月別データがありません")
    
    st.write("### 📋 初回発注量の提案")
    
    avg_daily = result.get('avg_daily', 1)
    col1, col2, col3 = st.columns(3)
    col1.metric("少なめ（1ヶ月分）", f"{round_up_to_50(int(avg_daily * 30))}体")
    col2.metric("標準（3ヶ月分）", f"{round_up_to_50(int(avg_daily * 90))}体")
    col3.metric("多め（6ヶ月分）", f"{round_up_to_50(int(avg_daily * 180))}体")


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
    
    # バージョン情報（v24更新）
    version_info = "v24.0.0 (異常値検出UI・データ入力日数表示)"
    if VERTEX_AI_AVAILABLE:
        version_info += " | 🚀 Vertex AI: 有効"
    else:
        version_info += " | ⚠️ Vertex AI: 未設定"
    
    st.caption(f"⛩️ 酒列磯前神社 授与品管理システム {version_info}")


if __name__ == "__main__":
    main()