"""
Google Sheets データ読み込みモジュール

機能:
- Google Sheets APIを使用したデータ取得
- キャッシュ機能
- データの前処理
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import config


class SheetsDataLoader:
    """Google Sheetsデータ読み込みクラス"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Args:
            credentials_path: サービスアカウントキーのパス（指定しない場合は環境変数から）
        """
        self.credentials = self._get_credentials(credentials_path)
        self.service = build('sheets', 'v4', credentials=self.credentials)
    
    def _get_credentials(self, credentials_path: Optional[str] = None):
        """認証情報を取得"""
        # Streamlit Cloudの場合はSecretsから
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            return service_account.Credentials.from_service_account_info(
                st.secrets['gcp_service_account'],
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
        
        # ローカルの場合はファイルから
        if credentials_path and Path(credentials_path).exists():
            return service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
        
        # 環境変数から
        key_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if key_path and Path(key_path).exists():
            return service_account.Credentials.from_service_account_file(
                key_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
        
        raise ValueError("認証情報が見つかりません")
    
    def _read_sheet(self, sheet_name: str, range_notation: str = "") -> pd.DataFrame:
        """シートからデータを読み込み"""
        try:
            full_range = f"'{sheet_name}'"
            if range_notation:
                full_range += f"!{range_notation}"
            
            result = self.service.spreadsheets().values().get(
                spreadsheetId=config.SPREADSHEET_ID,
                range=full_range
            ).execute()
            
            values = result.get('values', [])
            
            if not values:
                return pd.DataFrame()
            
            # 最初の行をヘッダーとして使用
            df = pd.DataFrame(values[1:], columns=values[0])
            
            return df
            
        except HttpError as e:
            st.error(f"シート読み込みエラー: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def load_sales_summary(_self) -> pd.DataFrame:
        """日別売上データを読み込み"""
        df = _self._read_sheet(config.SALES_SUMMARY_SHEET)
        
        if df.empty:
            return df
        
        # データ型変換
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        numeric_columns = ['売上', '会計数', '会計単価', '客数', '客単価', '商品数']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def load_item_sales(_self) -> pd.DataFrame:
        """商品別売上データを読み込み"""
        df = _self._read_sheet(config.ITEM_SALES_SHEET)
        
        if df.empty:
            return df
        
        # データ型変換
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        numeric_columns = ['販売総売上', '販売商品数', '返品商品数']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def load_calendar(_self) -> pd.DataFrame:
        """カレンダーデータを読み込み"""
        df = _self._read_sheet(config.CALENDAR_SHEET)
        
        if df.empty:
            return df
        
        # データ型変換
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Boolean変換
        bool_columns = ['is_weekend', 'is_holiday', 'is_taian', 
                       'is_ichiryumanbai', 'is_tensha', 'is_tora_no_hi', 'is_mi_no_hi']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})
        
        # 数値変換
        numeric_columns = ['temp_max', 'temp_min', 'temp_mean', 'precipitation']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_unique_products(self) -> List[str]:
        """ユニークな商品名リストを取得"""
        df = self.load_item_sales()
        if '商品名' in df.columns:
            return df['商品名'].dropna().unique().tolist()
        return []
    
    def get_date_range(self) -> tuple:
        """データの日付範囲を取得"""
        df = self.load_item_sales()
        if 'date' in df.columns and not df.empty:
            min_date = df['date'].min()
            max_date = df['date'].max()
            return min_date, max_date
        return None, None
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def load_mail_orders(_self) -> pd.DataFrame:
        """郵送フォームデータを読み込み"""
        mail_order_id = getattr(config, 'MAIL_ORDER_SPREADSHEET_ID', '')
        
        if not mail_order_id:
            return pd.DataFrame()
        
        try:
            result = _self.service.spreadsheets().values().get(
                spreadsheetId=mail_order_id,
                range="'フォームの回答 1'!A:BZ"  # 十分広い範囲
            ).execute()
            
            values = result.get('values', [])
            if not values or len(values) < 2:
                return pd.DataFrame()
            
            # ヘッダーとデータを取得
            headers = values[0]
            data = values[1:]
            
            # 行の長さを揃える
            max_cols = len(headers)
            normalized_data = []
            for row in data:
                if len(row) < max_cols:
                    row = row + [''] * (max_cols - len(row))
                normalized_data.append(row[:max_cols])
            
            df = pd.DataFrame(normalized_data, columns=headers)
            
            return df
        
        except HttpError as e:
            st.warning(f"郵送フォームデータの読み込みエラー: {e}")
            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()
    
    def get_mail_order_summary(_self) -> pd.DataFrame:
        """郵送フォームから商品別・日付別の販売数を集計"""
        df = _self.load_mail_orders()
        
        if df.empty:
            return pd.DataFrame()
        
        product_columns = getattr(config, 'MAIL_ORDER_PRODUCT_COLUMNS', {})
        
        if not product_columns:
            return pd.DataFrame()
        
        # タイムスタンプ列を探す
        timestamp_col = None
        for col in df.columns:
            if 'タイムスタンプ' in col or 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        if timestamp_col is None and len(df.columns) > 0:
            timestamp_col = df.columns[0]  # 最初の列をタイムスタンプとして使用
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                # 日付を解析
                if timestamp_col and pd.notna(row.get(timestamp_col)):
                    timestamp = pd.to_datetime(row[timestamp_col], errors='coerce')
                    if pd.isna(timestamp):
                        continue
                    order_date = timestamp.date()
                else:
                    continue
                
                # 各商品の数量を取得
                for form_col, system_name in product_columns.items():
                    if system_name is None:
                        continue
                    
                    # フォームの列を探す
                    matched_col = None
                    for col in df.columns:
                        if form_col in col:
                            matched_col = col
                            break
                    
                    if matched_col is None:
                        continue
                    
                    qty_str = row.get(matched_col, '')
                    if pd.isna(qty_str) or qty_str == '':
                        continue
                    
                    # 数量を解析（数字のみ抽出）
                    import re
                    numbers = re.findall(r'\d+', str(qty_str))
                    if numbers:
                        qty = int(numbers[0])
                        if qty > 0:
                            results.append({
                                'date': pd.Timestamp(order_date),
                                '商品名': system_name,
                                '販売商品数': qty,
                                '販売総売上': 0,  # 価格情報がないため0
                                '返品商品数': 0,
                                'source': 'mail_order'
                            })
            
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(results)
        
        # 日付・商品名でグループ化して集計
        summary = result_df.groupby(['date', '商品名']).agg({
            '販売商品数': 'sum',
            '販売総売上': 'sum',
            '返品商品数': 'sum'
        }).reset_index()
        
        summary['source'] = 'mail_order'
        
        return summary


def merge_with_calendar(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """売上データとカレンダーデータをマージ"""
    if sales_df.empty or calendar_df.empty:
        return sales_df
    
    # date列で結合
    merged = sales_df.merge(
        calendar_df,
        on='date',
        how='left'
    )
    
    return merged


def aggregate_by_products(df: pd.DataFrame, product_names: List[str], 
                         aggregate: bool = True) -> pd.DataFrame:
    """
    指定商品のデータを集計
    
    Args:
        df: 商品別売上データ
        product_names: 対象商品名リスト
        aggregate: Trueなら合算、Falseなら個別
    """
    if df.empty or not product_names:
        return pd.DataFrame()
    
    # 対象商品をフィルタ
    filtered = df[df['商品名'].isin(product_names)].copy()
    
    if filtered.empty:
        return pd.DataFrame()
    
    if aggregate:
        # 日付ごとに合算
        result = filtered.groupby('date').agg({
            '販売商品数': 'sum',
            '販売総売上': 'sum',
            '返品商品数': 'sum',
        }).reset_index()
        result['商品名'] = '合計'
    else:
        # 商品別・日付別に集計
        result = filtered.groupby(['date', '商品名']).agg({
            '販売商品数': 'sum',
            '販売総売上': 'sum',
            '返品商品数': 'sum',
        }).reset_index()
    
    return result
