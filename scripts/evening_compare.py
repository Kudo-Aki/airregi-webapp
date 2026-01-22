#!/usr/bin/env python3
"""
予測vs実績の比較・学習・メールレポートスクリプト

実行タイミング: 毎日 22:00（GitHub Actions）
機能:
  1. 当日の予測と実績を比較
  2. 差異の原因を分析
  3. 学習係数を更新
  4. 差異レポートをメール送信
"""

import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# 設定
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "1wbx8zfP-n-mDnzVshIaFulinpFj-uoIGmNIsI_QTEVQ")
# ローカル用: サービスアカウントキーは環境変数またはカレントディレクトリから
SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

# メール設定
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")  # Gmailの場合はアプリパスワード
EMAIL_TO = os.environ.get("EMAIL_TO", "")

# 学習設定
LEARNING_RATE = 0.1  # 学習率（0.1 = 10%ずつ調整）
COEFFICIENT_MIN = 0.3  # 係数の最小値
COEFFICIENT_MAX = 3.0  # 係数の最大値


def get_service():
    """Google Sheets APIサービスを取得"""
    sa_info = os.environ.get("GCP_SERVICE_ACCOUNT")
    
    if sa_info:
        creds = Credentials.from_service_account_info(
            json.loads(sa_info),
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
    elif SERVICE_ACCOUNT_FILE and Path(SERVICE_ACCOUNT_FILE).exists():
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
    else:
        raise FileNotFoundError("サービスアカウントキーが見つかりません。環境変数 GCP_SERVICE_ACCOUNT または GOOGLE_APPLICATION_CREDENTIALS を設定してください。")
    
    return build('sheets', 'v4', credentials=creds)


def load_predictions(service, target_date: date) -> list:
    """当日の予測データを読み込み"""
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="'forecast_log'!A:G"
        ).execute()
        
        values = result.get('values', [])
        if len(values) <= 1:
            return []
        
        predictions = []
        headers = values[0]
        
        for row in values[1:]:
            if len(row) >= 4:
                row_date = row[0]
                if row_date == target_date.isoformat():
                    predictions.append({
                        'product_name': row[2] if len(row) > 2 else '',
                        'predicted_qty': int(float(row[3])) if len(row) > 3 and row[3] else 0,
                        'predicted_sales': int(float(row[4])) if len(row) > 4 and row[4] else 0,
                        'weather': row[6] if len(row) > 6 else ''
                    })
        
        return predictions
    
    except Exception as e:
        print(f"予測データ読み込みエラー: {e}")
        return []


def load_actuals(service, target_date: date) -> list:
    """当日の実績データを読み込み"""
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="'daily_item_sales'!A:G"
        ).execute()
        
        values = result.get('values', [])
        if len(values) <= 1:
            return []
        
        actuals = []
        headers = values[0]
        
        # 列インデックスを特定
        date_idx = headers.index('date') if 'date' in headers else 0
        product_idx = headers.index('商品名') if '商品名' in headers else 2
        qty_idx = headers.index('販売商品数') if '販売商品数' in headers else 4
        sales_idx = headers.index('販売総売上') if '販売総売上' in headers else 5
        
        for row in values[1:]:
            if len(row) > date_idx:
                row_date = row[date_idx]
                if row_date == target_date.isoformat():
                    actuals.append({
                        'product_name': row[product_idx] if len(row) > product_idx else '',
                        'actual_qty': int(float(row[qty_idx])) if len(row) > qty_idx and row[qty_idx] else 0,
                        'actual_sales': int(float(row[sales_idx])) if len(row) > sales_idx and row[sales_idx] else 0
                    })
        
        return actuals
    
    except Exception as e:
        print(f"実績データ読み込みエラー: {e}")
        return []


def load_calendar_info(service, target_date: date) -> dict:
    """当日のカレンダー情報を取得"""
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="'calendar_data'!A:U"
        ).execute()
        
        values = result.get('values', [])
        if len(values) <= 1:
            return {}
        
        headers = values[0]
        
        for row in values[1:]:
            if len(row) > 0 and row[0] == target_date.isoformat():
                info = {}
                for i, h in enumerate(headers):
                    if i < len(row):
                        info[h] = row[i]
                return info
        
        return {}
    
    except Exception as e:
        print(f"カレンダー情報読み込みエラー: {e}")
        return {}


def load_weather_actual(service, target_date: date) -> str:
    """当日の実際の天気を取得"""
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="'calendar_data'!A:U"
        ).execute()
        
        values = result.get('values', [])
        if len(values) <= 1:
            return "不明"
        
        headers = values[0]
        weather_idx = headers.index('weather') if 'weather' in headers else -1
        
        if weather_idx < 0:
            return "不明"
        
        for row in values[1:]:
            if len(row) > 0 and row[0] == target_date.isoformat():
                return row[weather_idx] if len(row) > weather_idx else "不明"
        
        return "不明"
    
    except:
        return "不明"


def compare_predictions_and_actuals(predictions: list, actuals: list, 
                                     target_date: date, calendar_info: dict,
                                     weather_actual: str) -> list:
    """予測と実績を比較"""
    
    # 実績を辞書に変換
    actual_dict = {a['product_name']: a for a in actuals}
    
    comparisons = []
    
    for pred in predictions:
        product = pred['product_name']
        actual = actual_dict.get(product, {'actual_qty': 0, 'actual_sales': 0})
        
        predicted_qty = pred['predicted_qty']
        actual_qty = actual['actual_qty']
        
        diff = actual_qty - predicted_qty
        diff_pct = (diff / predicted_qty * 100) if predicted_qty > 0 else 0
        
        # 要因を収集
        factors = []
        factors.append(f"weekday_{target_date.weekday()}")
        factors.append(f"month_{target_date.month}")
        
        if weather_actual and weather_actual != "不明":
            factors.append(f"weather_{weather_actual}")
        
        if calendar_info:
            rokuyou = calendar_info.get('rokuyou', '')
            if rokuyou:
                factors.append(f"rokuyou_{rokuyou}")
            
            special = calendar_info.get('special_period', '')
            if special:
                factors.append(f"special_{special}")
        
        comparisons.append({
            'date': target_date,
            'product_name': product,
            'predicted_qty': predicted_qty,
            'actual_qty': actual_qty,
            'diff': diff,
            'diff_pct': diff_pct,
            'weather_actual': weather_actual,
            'factors': factors
        })
    
    return comparisons


def save_comparisons(service, comparisons: list):
    """比較結果をスプレッドシートに保存"""
    
    # シートが存在するか確認、なければ作成
    try:
        service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="'forecast_accuracy'!A1"
        ).execute()
    except:
        body = {
            'requests': [{
                'addSheet': {
                    'properties': {'title': 'forecast_accuracy'}
                }
            }]
        }
        service.spreadsheets().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body).execute()
        
        headers = [["date", "product_name", "predicted_qty", "actual_qty", "diff", "diff_pct", "weather_actual", "factors"]]
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range="'forecast_accuracy'!A1",
            valueInputOption='RAW',
            body={'values': headers}
        ).execute()
    
    # データを追加
    rows = []
    for c in comparisons:
        rows.append([
            c['date'].isoformat(),
            c['product_name'],
            c['predicted_qty'],
            c['actual_qty'],
            c['diff'],
            round(c['diff_pct'], 1),
            c['weather_actual'],
            ','.join(c['factors'])
        ])
    
    if rows:
        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range="'forecast_accuracy'!A:H",
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',
            body={'values': rows}
        ).execute()
    
    print(f"比較結果を保存しました: {len(rows)}件")


def load_learning_coefficients(service) -> dict:
    """学習係数を読み込み"""
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
    
    except:
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


def update_learning_coefficients(service, comparisons: list, coefficients: dict):
    """学習係数を更新"""
    
    updated = False
    
    for comp in comparisons:
        if comp['diff_pct'] == 0 or comp['predicted_qty'] == 0:
            continue
        
        # 調整量を計算（最大±20%）
        adjustment = comp['diff_pct'] / 100 * LEARNING_RATE
        adjustment = max(-0.2, min(0.2, adjustment))
        
        # 各要因の係数を調整
        for factor in comp['factors']:
            parts = factor.split('_', 1)
            if len(parts) != 2:
                continue
            
            factor_type, factor_value = parts
            
            if factor_type not in coefficients:
                coefficients[factor_type] = {}
            
            current = coefficients[factor_type].get(factor_value, 1.0)
            new_value = current * (1 + adjustment)
            
            # 係数の範囲を制限
            new_value = max(COEFFICIENT_MIN, min(COEFFICIENT_MAX, new_value))
            
            coefficients[factor_type][factor_value] = round(new_value, 3)
            updated = True
    
    if updated:
        save_learning_coefficients(service, coefficients)
    
    return coefficients


def save_learning_coefficients(service, coefficients: dict):
    """学習係数を保存"""
    
    # シートが存在するか確認、なければ作成
    try:
        service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="'learning_coefficients'!A1"
        ).execute()
    except:
        body = {
            'requests': [{
                'addSheet': {
                    'properties': {'title': 'learning_coefficients'}
                }
            }]
        }
        service.spreadsheets().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body).execute()
    
    # データを作成
    rows = [["factor_type", "factor_value", "base_coefficient", "learned_adjustment", "updated_at"]]
    
    updated_at = datetime.now().isoformat()
    
    for factor_type, values in coefficients.items():
        for factor_value, learned in values.items():
            rows.append([factor_type, factor_value, "1.0", str(learned), updated_at])
    
    # 全データを上書き
    service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range="'learning_coefficients'!A1",
        valueInputOption='RAW',
        body={'values': rows}
    ).execute()
    
    print(f"学習係数を更新しました: {len(rows)-1}件")


def generate_report(comparisons: list, target_date: date) -> str:
    """差異レポートを生成"""
    
    if not comparisons:
        return f"""
酒列磯前神社 需要予測レポート
日付: {target_date.strftime('%Y年%m月%d日')}

本日の予測データまたは実績データがありませんでした。
"""
    
    # 集計
    total_predicted = sum(c['predicted_qty'] for c in comparisons)
    total_actual = sum(c['actual_qty'] for c in comparisons)
    total_diff = total_actual - total_predicted
    total_diff_pct = (total_diff / total_predicted * 100) if total_predicted > 0 else 0
    
    # 差異が大きい商品TOP5
    sorted_by_diff = sorted(comparisons, key=lambda x: abs(x['diff']), reverse=True)[:5]
    
    # 曜日名
    weekday_names = ['月', '火', '水', '木', '金', '土', '日']
    weekday = weekday_names[target_date.weekday()]
    
    # 天気
    weather = comparisons[0]['weather_actual'] if comparisons else '不明'
    
    report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⛩️ 酒列磯前神社 需要予測 差異レポート
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 日付: {target_date.strftime('%Y年%m月%d日')}（{weekday}曜日）
🌤️ 天気: {weather}

────────────────────────────────────────────────────
📊 本日のサマリー
────────────────────────────────────────────────────

  予測合計:   {total_predicted:>6}体
  実績合計:   {total_actual:>6}体
  ────────────────────
  差異:       {total_diff:>+6}体 ({total_diff_pct:+.1f}%)

"""
    
    # 評価
    if abs(total_diff_pct) < 10:
        report += "  📈 評価: ⭐⭐⭐ 優秀（誤差10%未満）\n"
    elif abs(total_diff_pct) < 20:
        report += "  📈 評価: ⭐⭐ 良好（誤差20%未満）\n"
    elif abs(total_diff_pct) < 30:
        report += "  📈 評価: ⭐ 普通（誤差30%未満）\n"
    else:
        report += "  📈 評価: 要改善（誤差30%以上）\n"
    
    report += """
────────────────────────────────────────────────────
📋 差異が大きかった商品 TOP5
────────────────────────────────────────────────────

"""
    
    for i, c in enumerate(sorted_by_diff, 1):
        sign = "+" if c['diff'] >= 0 else ""
        report += f"  {i}. {c['product_name'][:20]}\n"
        report += f"     予測: {c['predicted_qty']}体 → 実績: {c['actual_qty']}体 ({sign}{c['diff']}体, {c['diff_pct']:+.1f}%)\n\n"
    
    report += """
────────────────────────────────────────────────────
🔄 学習状況
────────────────────────────────────────────────────

本日のデータを学習し、予測モデルを自動調整しました。
明日以降の予測精度が向上します。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
このメールは自動送信されています。
"""
    
    return report


def send_email_report(report: str, target_date: date):
    """メールでレポートを送信"""
    
    if not SMTP_USER or not SMTP_PASSWORD or not EMAIL_TO:
        print("メール設定がありません。スキップします。")
        return
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = EMAIL_TO
        msg['Subject'] = f"⛩️ 需要予測レポート {target_date.strftime('%Y/%m/%d')}"
        
        msg.attach(MIMEText(report, 'plain', 'utf-8'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        print(f"メールを送信しました: {EMAIL_TO}")
    
    except Exception as e:
        print(f"メール送信エラー: {e}")


def main():
    print("=" * 60)
    print(f"予測vs実績 比較・学習スクリプト - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    target_date = date.today()
    
    # 1. サービス初期化
    service = get_service()
    
    # 2. 予測データを読み込み
    print(f"対象日: {target_date}")
    predictions = load_predictions(service, target_date)
    print(f"予測データ: {len(predictions)}件")
    
    if not predictions:
        print("予測データがありません。終了します。")
        return
    
    # 3. 実績データを読み込み
    actuals = load_actuals(service, target_date)
    print(f"実績データ: {len(actuals)}件")
    
    # 4. カレンダー情報と天気を取得
    calendar_info = load_calendar_info(service, target_date)
    weather_actual = load_weather_actual(service, target_date)
    print(f"天気（実績）: {weather_actual}")
    
    # 5. 比較
    comparisons = compare_predictions_and_actuals(
        predictions, actuals, target_date, calendar_info, weather_actual
    )
    print(f"比較件数: {len(comparisons)}件")
    
    # 6. 比較結果を保存
    save_comparisons(service, comparisons)
    
    # 7. 学習係数を更新
    coefficients = load_learning_coefficients(service)
    coefficients = update_learning_coefficients(service, comparisons, coefficients)
    
    # 8. レポート生成
    report = generate_report(comparisons, target_date)
    print("\n" + report)
    
    # 9. メール送信
    send_email_report(report, target_date)
    
    print("\n完了！")


if __name__ == "__main__":
    main()
