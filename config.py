# =============================================================================
# Airレジ 売上分析・需要予測 Webアプリ 設定ファイル
# =============================================================================

# Google Sheets設定
SPREADSHEET_ID = "1wbx8zfP-n-mDnzVshIaFulinpFj-uoIGmNIsI_QTEVQ"
SALES_SUMMARY_SHEET = "daily_sales_summary"
ITEM_SALES_SHEET = "daily_item_sales"
CALENDAR_SHEET = "calendar_data"  # 新規作成するシート

# =============================================================================
# v18追加: 郵送フォーム（Googleフォーム回答）の設定
# =============================================================================
# 郵送依頼フォームの回答スプレッドシートID
# フォームの回答シートのURLからIDを取得してください
# 例: https://docs.google.com/spreadsheets/d/XXXXXX/edit → XXXXXX の部分
MAIL_ORDER_SPREADSHEET_ID = ""  # ここに郵送フォームのスプレッドシートIDを設定

# 郵送フォームのシート名
MAIL_ORDER_SHEET_NAME = "フォームの回答 1"

# 郵送フォームの商品列マッピング
# フォームの列名 → Airレジの商品名
MAIL_ORDER_PRODUCT_COLUMNS = {
    # 馬守シリーズ（郵送フォーム専用？）
    "駆馬守": "駆馬守",
    "跳馬守": "跳馬守",
    "福馬守": "福馬守",
    "うまくいく守": "うまくいく守",
    
    # 金運系
    "金運守": "金運守",
    "願い貝守": "願い貝守",
    "金運セット": "金運セット",
    
    # 健康系
    "健康守 ": "健康守",  # 末尾スペースあり
    "水琴鈴健康守（亀）": "水琴鈴健康守　亀",
    "水琴鈴健康守（椿）": "水琴鈴健康守　椿",
    "病気平癒守（袋型）": "病気平癒守",
    "病気平癒守（カード型）": "病気平癒カード",
    "病気平癒守（桐箱型）": "病気平癒桐箱",
    
    # その他のお守り
    "仕事守": "仕事守",
    "開運厄除守": "開運厄除守",
    "子宝守 ": "子宝守",  # 末尾スペースあり
    "安産守": "安産守",
    "勝守 ": "勝守",  # 末尾スペースあり
    "椿守": "椿守",
    "魔除守": "魔除守",
    "学業守": "学業守",
    "合格祈願守": "合格祈願守",
    
    # 交通安全系
    "交通安全守（赤）": "交通安全　赤",
    "交通安全守（白）": "交通安全　白",
    "交通安全守（紫）": "交通安全　紫",
    "交通安全守（木札）": "交通安全　木札",
    "交通安全守（ステッカー）": "交通安全　ステッカー",
    
    # カード守
    "カード守（白）": "カード守　白",
    "カード守（赤）": "カード守　赤",
    "カード守（緑)": "カード守　緑",  # 閉じ括弧が半角
    
    # 子ども守
    "子ども守（青）": "子供守　青",
    "子ども守（ピンク）": "子供守　ピンク",
    
    # 縁結び（フォームは「縁結守」、Airレジは色別）
    "縁結守": "縁結び　青",  # デフォルトで青にマッピング（複数色ある場合は要調整）
}

# 認証設定
ALLOWED_DOMAINS = ["gmail.com"]  # 許可するGoogleアカウントのドメイン

# キャッシュ設定（秒）
CACHE_TTL = 300  # 5分

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
