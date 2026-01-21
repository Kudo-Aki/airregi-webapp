#!/usr/bin/env python3
"""
毎朝の需要予測スクリプト

実行タイミング: 毎日 6:00（GitHub Actions）
機能:
  - 向こう7日間の売上・商品別販売数を予測
  - 予測データをスプレッドシート（forecast_log）に保存
  - 学習済み係数を適用
"""

import os
import sys
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# 設定
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "1wbx8zfP-n-mDnzVshIaFulinpFj-uoIGmNIsI_QTEVQ")
SERVICE_ACCOUNT_FILE = "airregi-csv-automation-d19ec6c116ff.json"

# ひたちなか市の座標
LATITUDE = 36.3833
LONGITUDE = 140.6167


def get_service():
    """Google Sheets APIサービスを取得"""
    # 環境変数からサービスアカウント情報を取得（GitHub Actions用）
    sa_info = os.environ.get("GCP_SERVICE_ACCOUNT")
    
    if sa_info:
        import json
        creds = Credentials.from_service_account_info(
            json.loads(sa_info),
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
    elif Path(SERVICE_ACCOUNT_FILE).exists():
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
    else:
        raise FileNotFoundError("サービスアカウントキーが見つかりません")
    
    return build('sheets', 'v4', credentials=creds)


def fetch_weather_forecast(days: int = 7) -> list:
    """Open-Meteo APIから天気予報を取得"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Tokyo",
        "forecast_days": days
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        weather_codes = {
            0: "晴れ", 1: "晴れ", 2: "曇り", 3: "曇り",
            45: "霧", 48: "霧",
            51: "小雨", 53: "雨", 55: "雨",
            61: "雨", 63: "雨", 65: "大雨",
            71: "雪", 73: "雪", 75: "大雪",
            80: "にわか雨", 81: "にわか雨", 82: "大雨",
            95: "雷雨", 96: "雷雨", 99: "雷雨"
        }
        
        forecasts = []
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        codes = daily.get("weather_code", [])
        
        for i, d in enumerate(dates):
            code = codes[i] if i < len(codes) else 0
            weather = weather_codes.get(code, "不明")
            forecasts.append({
                "date": d,
                "weather": weather,
                "weather_code": code
            })
        
        return forecasts
    
    except Exception as e:
        print(f"天気予報取得エラー: {e}")
        # デフォルト値を返す
        return [{"date": (date.today() + timedelta(days=i)).isoformat(), "weather": "不明", "weather_code": 0} for i in range(days)]


def load_sales_data(service) -> pd.DataFrame:
    """売上データを読み込み"""
    result = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range="'daily_item_sales'!A:G"
    ).execute()
    
    values = result.get('values', [])
    if not values:
        return pd.DataFrame()
    
    df = pd.DataFrame(values[1:], columns=values[0])
    
    # 日付変換
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 数値変換
    for col in ['販売商品数', '販売総売上']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def load_calendar_data(service) -> pd.DataFrame:
    """カレンダーデータを読み込み"""
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="'calendar_data'!A:U"
        ).execute()
        
        values = result.get('values', [])
        if not values:
            return pd.DataFrame()
        
        df = pd.DataFrame(values[1:], columns=values[0])
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df
    except:
        return pd.DataFrame()


def load_learning_coefficients(service) -> dict:
    """学習済み係数を読み込み"""
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="'learning_coefficients'!A:E"
        ).execute()
        
        values = result.get('values', [])
        if len(values) <= 1:
            return get_default_coefficients()
        
        coefficients = {}
        for row in values[1:]:
            if len(row) >= 4:
                factor_type = row[0]
                factor_value = row[1]
                learned = float(row[3]) if row[3] else 1.0
                
                if factor_type not in coefficients:
                    coefficients[factor_type] = {}
                coefficients[factor_type][factor_value] = learned
        
        return coefficients if coefficients else get_default_coefficients()
    
    except Exception as e:
        print(f"係数読み込みエラー: {e}")
        return get_default_coefficients()


def get_default_coefficients() -> dict:
    """デフォルトの係数"""
    return {
        "weekday": {"0": 0.9, "1": 0.95, "2": 0.95, "3": 1.0, "4": 1.1, "5": 1.4, "6": 1.6},
        "month": {"1": 3.0, "2": 0.8, "3": 1.0, "4": 1.0, "5": 1.1, "6": 0.9, 
                  "7": 1.0, "8": 1.1, "9": 1.0, "10": 1.1, "11": 1.3, "12": 1.5},
        "weather": {"晴れ": 1.0, "曇り": 0.95, "雨": 0.7, "雪": 0.5, "不明": 1.0},
        "rokuyou": {"大安": 1.3, "友引": 1.1, "先勝": 1.05, "先負": 0.95, "赤口": 0.9, "仏滅": 0.85},
        "special": {"年末年始": 3.0, "GW": 1.5, "お盆": 1.8, "お彼岸": 1.5, "七五三": 2.0}
    }


def get_active_products(df_sales: pd.DataFrame) -> list:
    """アクティブな商品リストを取得（過去90日で売上がある商品）"""
    cutoff = datetime.now() - timedelta(days=90)
    
    if 'date' not in df_sales.columns or '商品名' not in df_sales.columns:
        return []
    
    recent = df_sales[df_sales['date'] >= cutoff]
    
    # 商品ごとの売上集計
    product_col = '商品名'
    qty_col = '販売商品数'
    
    if qty_col not in recent.columns:
        return []
    
    product_sales = recent.groupby(product_col)[qty_col].sum()
    
    # 売上が1以上の商品を返す
    active = product_sales[product_sales > 0].index.tolist()
    
    return active[:50]  # 最大50商品


def forecast_product(df_sales: pd.DataFrame, product_name: str, target_date: date, 
                     coefficients: dict, weather: str, df_calendar: pd.DataFrame) -> dict:
    """商品の需要を予測"""
    
    # 商品の過去データを抽出
    product_col = '商品名'
    qty_col = '販売商品数'
    sales_col = '販売総売上'
    
    product_data = df_sales[df_sales[product_col] == product_name].copy()
    
    if product_data.empty:
        return {"qty": 0, "sales": 0}
    
    # 基本統計
    overall_mean = product_data[qty_col].mean()
    
    if pd.isna(overall_mean) or overall_mean == 0:
        overall_mean = 1
    
    # 平均単価
    total_qty = product_data[qty_col].sum()
    total_sales = product_data[sales_col].sum()
    unit_price = total_sales / total_qty if total_qty > 0 else 1000
    
    # 基本予測値
    base_prediction = overall_mean
    
    # 係数を適用
    adjusted = base_prediction
    
    # 曜日係数
    weekday_coef = coefficients.get("weekday", {}).get(str(target_date.weekday()), 1.0)
    adjusted *= weekday_coef
    
    # 月係数
    month_coef = coefficients.get("month", {}).get(str(target_date.month), 1.0)
    adjusted *= month_coef
    
    # 天気係数
    weather_coef = coefficients.get("weather", {}).get(weather, 1.0)
    adjusted *= weather_coef
    
    # 六曜・特別期間（カレンダーデータがある場合）
    if not df_calendar.empty:
        target_dt = pd.Timestamp(target_date)
        cal_row = df_calendar[df_calendar['date'] == target_dt]
        
        if not cal_row.empty:
            # 六曜
            if 'rokuyou' in cal_row.columns:
                rokuyou = cal_row['rokuyou'].iloc[0]
                if pd.notna(rokuyou):
                    rokuyou_coef = coefficients.get("rokuyou", {}).get(rokuyou, 1.0)
                    adjusted *= rokuyou_coef
            
            # 特別期間
            if 'special_period' in cal_row.columns:
                special = cal_row['special_period'].iloc[0]
                if pd.notna(special) and special:
                    special_coef = coefficients.get("special", {}).get(special, 1.0)
                    adjusted *= special_coef
    
    # 最低でも0.1は確保
    adjusted = max(0.1, adjusted)
    
    return {
        "qty": round(adjusted),
        "sales": round(adjusted * unit_price)
    }


def save_forecasts(service, forecasts: list):
    """予測データをスプレッドシートに保存"""
    
    # シートが存在するか確認、なければ作成
    try:
        service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="'forecast_log'!A1"
        ).execute()
    except:
        # シートを作成
        body = {
            'requests': [{
                'addSheet': {
                    'properties': {'title': 'forecast_log'}
                }
            }]
        }
        service.spreadsheets().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body).execute()
        
        # ヘッダーを追加
        headers = [["date", "created_at", "product_name", "predicted_qty", "predicted_sales", "method", "weather_forecast"]]
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range="'forecast_log'!A1",
            valueInputOption='RAW',
            body={'values': headers}
        ).execute()
    
    # データを追加
    rows = []
    for f in forecasts:
        rows.append([
            f['date'].isoformat() if isinstance(f['date'], date) else f['date'],
            f['created_at'].isoformat() if isinstance(f['created_at'], datetime) else f['created_at'],
            f['product_name'],
            f['predicted_qty'],
            f['predicted_sales'],
            f['method'],
            f['weather']
        ])
    
    if rows:
        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range="'forecast_log'!A:G",
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',
            body={'values': rows}
        ).execute()
    
    print(f"予測データを保存しました: {len(rows)}件")


def main():
    print("=" * 60)
    print(f"毎朝の需要予測スクリプト - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. サービス初期化
    service = get_service()
    
    # 2. データ読み込み
    print("データを読み込み中...")
    df_sales = load_sales_data(service)
    df_calendar = load_calendar_data(service)
    coefficients = load_learning_coefficients(service)
    
    if df_sales.empty:
        print("売上データがありません")
        return
    
    print(f"  売上データ: {len(df_sales)}件")
    print(f"  カレンダーデータ: {len(df_calendar)}件")
    
    # 3. 天気予報を取得
    print("天気予報を取得中...")
    weather_forecast = fetch_weather_forecast(days=7)
    
    for w in weather_forecast:
        print(f"  {w['date']}: {w['weather']}")
    
    # 4. アクティブな商品を取得
    products = get_active_products(df_sales)
    print(f"予測対象商品: {len(products)}件")
    
    if not products:
        print("予測対象の商品がありません")
        return
    
    # 5. 予測を実行
    print("予測を実行中...")
    forecasts = []
    created_at = datetime.now()
    
    for product in products:
        for i, w in enumerate(weather_forecast):
            target_date = date.today() + timedelta(days=i)
            
            prediction = forecast_product(
                df_sales, product, target_date,
                coefficients, w['weather'], df_calendar
            )
            
            forecasts.append({
                'date': target_date,
                'created_at': created_at,
                'product_name': product,
                'predicted_qty': prediction['qty'],
                'predicted_sales': prediction['sales'],
                'method': 'seasonal_learned',
                'weather': w['weather']
            })
    
    print(f"予測件数: {len(forecasts)}件")
    
    # 6. 保存
    save_forecasts(service, forecasts)
    
    # 7. サマリー出力
    total_qty = sum(f['predicted_qty'] for f in forecasts if f['date'] == date.today())
    total_sales = sum(f['predicted_sales'] for f in forecasts if f['date'] == date.today())
    
    print("\n本日の予測サマリー:")
    print(f"  合計販売数: {total_qty}体")
    print(f"  合計売上: ¥{total_sales:,}")
    
    print("\n完了！")


if __name__ == "__main__":
    main()
