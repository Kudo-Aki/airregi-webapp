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
