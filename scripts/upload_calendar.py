#!/usr/bin/env python3
"""
カレンダーデータをGoogle Sheetsにアップロード

使用前に:
1. サービスアカウントキーを同じフォルダに配置
2. スプレッドシートをサービスアカウントに共有
"""

import csv
import json
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build

# 設定
SPREADSHEET_ID = "1wbx8zfP-n-mDnzVshIaFulinpFj-uoIGmNIsI_QTEVQ"
SHEET_NAME = "calendar_data"
SERVICE_ACCOUNT_KEY = "airregi-csv-automation-d19ec6c116ff.json"
CSV_FILE = "calendar_data_v2.csv"


def upload_calendar_data():
    """カレンダーデータをアップロード"""
    
    # 認証
    key_path = Path(SERVICE_ACCOUNT_KEY)
    if not key_path.exists():
        print(f"エラー: サービスアカウントキーが見つかりません: {SERVICE_ACCOUNT_KEY}")
        return
    
    credentials = service_account.Credentials.from_service_account_file(
        str(key_path),
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    service = build('sheets', 'v4', credentials=credentials)
    
    # CSVを読み込み
    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        print(f"エラー: CSVファイルが見つかりません: {CSV_FILE}")
        print("先に generate_calendar.py を実行してください")
        return
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        values = list(reader)
    
    print(f"読み込み完了: {len(values)}行")
    
    # シートが存在するか確認、なければ作成
    try:
        sheet_metadata = service.spreadsheets().get(
            spreadsheetId=SPREADSHEET_ID
        ).execute()
        
        sheet_exists = False
        for sheet in sheet_metadata.get('sheets', []):
            if sheet['properties']['title'] == SHEET_NAME:
                sheet_exists = True
                break
        
        if not sheet_exists:
            # シートを作成（2000行を確保）
            request_body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': SHEET_NAME,
                            'gridProperties': {
                                'rowCount': 2000,
                                'columnCount': 30
                            }
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body=request_body
            ).execute()
            print(f"シート '{SHEET_NAME}' を作成しました（2000行）")
        else:
            # 既存シートの行数を拡張
            request_body = {
                'requests': [{
                    'updateSheetProperties': {
                        'properties': {
                            'sheetId': next(
                                s['properties']['sheetId'] 
                                for s in sheet_metadata['sheets'] 
                                if s['properties']['title'] == SHEET_NAME
                            ),
                            'gridProperties': {
                                'rowCount': 2000,
                                'columnCount': 30
                            }
                        },
                        'fields': 'gridProperties(rowCount,columnCount)'
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body=request_body
            ).execute()
            print(f"シート '{SHEET_NAME}' の行数を拡張しました")
    except Exception as e:
        print(f"シート確認エラー: {e}")
        return
    
    # 既存データをクリア
    try:
        service.spreadsheets().values().clear(
            spreadsheetId=SPREADSHEET_ID,
            range=f"'{SHEET_NAME}'!A:Z"
        ).execute()
        print("既存データをクリアしました")
    except:
        pass
    
    # データをアップロード（バッチ処理）
    batch_size = 500
    total = len(values)
    
    for i in range(0, total, batch_size):
        batch = values[i:i + batch_size]
        
        if i == 0:
            # 最初のバッチはA1から
            range_notation = f"'{SHEET_NAME}'!A1"
        else:
            # 以降は追加
            range_notation = f"'{SHEET_NAME}'!A{i + 1}"
        
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=range_notation,
            valueInputOption='RAW',
            body={'values': batch}
        ).execute()
        
        print(f"アップロード: {min(i + batch_size, total)}/{total}行")
    
    print(f"\n完了！スプレッドシートの '{SHEET_NAME}' シートにデータをアップロードしました")
    print(f"URL: https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════╗
║     カレンダーデータ アップロードスクリプト                    ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    upload_calendar_data()
