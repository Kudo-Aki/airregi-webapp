# =============================================================================
# Airレジ 売上分析・需要予測 Webアプリ 設定ファイル
# =============================================================================

# Google Sheets設定
SPREADSHEET_ID = "1wbx8zfP-n-mDnzVshIaFulinpFj-uoIGmNIsI_QTEVQ"
SALES_SUMMARY_SHEET = "daily_sales_summary"
ITEM_SALES_SHEET = "daily_item_sales"
CALENDAR_SHEET = "calendar_data"  # 新規作成するシート

# 認証設定
ALLOWED_DOMAINS = ["gmail.com"]  # 許可するGoogleアカウントのドメイン

# キャッシュ設定（秒）
CACHE_TTL = 600  # 5分

# 商品名正規化設定
NORMALIZE_OPTIONS = {
    "remove_prefix_numbers": True,  # 先頭の数字を除去
    "normalize_width": True,        # 全角半角を統一
    "remove_brackets": True,        # 【】の中身を除去
}

# 需要予測設定
FORECAST_PARAMS = {
    "default_periods": 365,          # デフォルト予測期間（日）
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
}

# 納品計画設定
DELIVERY_PARAMS = {
    "default_lead_time": 30,         # デフォルトリードタイム（日）
    "default_safety_stock_days": 14, # 安全在庫日数
    "default_order_lot": 100,        # 発注ロット単位
}

# =============================================================================
# v12追加: Vertex AI AutoML Forecasting 設定
# =============================================================================
# 
# 【設定方法】
# 1. Google Cloud Console でプロジェクトを作成または選択
# 2. Vertex AI API を有効化
# 3. AutoML Forecasting でモデルをトレーニング・デプロイ
# 4. サービスアカウントを作成し、JSONキーをダウンロード
# 5. 以下の値を設定
#
# 【環境変数での設定も可能】
# - VERTEX_AI_PROJECT_ID
# - VERTEX_AI_LOCATION
# - VERTEX_AI_ENDPOINT_ID
# - VERTEX_AI_SERVICE_ACCOUNT_FILE
# =============================================================================

# Google CloudプロジェクトID（例: "my-project-123456"）
VERTEX_AI_PROJECT_ID = ""

# リージョン（日本の場合は asia-northeast1 推奨）
VERTEX_AI_LOCATION = "asia-northeast1"

# AutoML ForecastingエンドポイントID
# デプロイ後にVertex AI Consoleで確認できます（例: "1234567890123456789"）
VERTEX_AI_ENDPOINT_ID = ""

# サービスアカウントJSONファイルのパス
# プロジェクトルートからの相対パス、または絶対パス
VERTEX_AI_SERVICE_ACCOUNT_FILE = "service_account.json"
