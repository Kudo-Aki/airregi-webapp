"""
Google Sheets データ読み込みモジュール

機能:
- Google Sheets APIを使用したデータ取得
- キャッシュ機能（@st.cache_data + @st.cache_resource）
- データの前処理

【パフォーマンス改善】
- @st.cache_resource で認証情報とAPIサービスを全セッション間で共有
- 新規セッション開始時の build() 再実行を防止
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


# =============================================================================
# 【パフォーマンス改善】認証情報・APIサービスのキャッシュ
# =============================================================================
# @st.cache_resource はプロセス全体で共有されるため、
# 新規セッション開始時にも build() が再実行されない。
# Community Cloud では通常1プロセスなので全ユーザーで共有される。
# =============================================================================

@st.cache_resource
def _get_cached_credentials():
    """
    認証情報を取得してキャッシュ（全セッション共有）
    
    @st.cache_resource により、プロセス内で1回だけ実行される。
    """
    # Streamlit Cloudの場合はSecretsから
    if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
        return service_account.Credentials.from_service_account_info(
            st.secrets['gcp_service_account'],
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


@st.cache_resource
def _get_cached_sheets_service():
    """
    Google Sheets APIサービスをキャッシュ（全セッション共有）
    
    build() はHTTPクライアント初期化＋ディスカバリドキュメント取得を含む
    重い処理のため、@st.cache_resource でキャッシュする。
    
    注意: Google APIクライアントは完全なスレッドセーフではないが、
    Community Cloud では同時アクセスユーザーが限られるため実用上問題ない。
    """
    credentials = _get_cached_credentials()
    return build('sheets', 'v4', credentials=credentials)


class SheetsDataLoader:
    """Google Sheetsデータ読み込みクラス"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Args:
            credentials_path: サービスアカウントキーのパス（指定しない場合は環境変数から）
        
        【パフォーマンス改善】
        認証情報とサービスの初期化を @st.cache_resource 経由に変更。
        credentials_path 引数は後方互換性のため残すが、
        キャッシュ済みの場合は無視される（st.secrets または環境変数が優先）。
        """
        if credentials_path and Path(credentials_path).exists():
            # 明示的にパスが指定された場合は従来通り直接生成
            self.credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
            self.service = build('sheets', 'v4', credentials=self.credentials)
        else:
            # 通常パス: キャッシュ済みサービスを使用
            self.credentials = _get_cached_credentials()
            self.service = _get_cached_sheets_service()
    
    def _get_credentials(self, credentials_path: Optional[str] = None):
        """認証情報を取得（後方互換性のため残す）"""
        return _get_cached_credentials()
    
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
            # シート名を設定から取得
            sheet_name = getattr(config, 'MAIL_ORDER_SHEET_NAME', 'フォームの回答 1')
            
            result = _self.service.spreadsheets().values().get(
                spreadsheetId=mail_order_id,
                range=f"'{sheet_name}'!A:BZ"  # 十分広い範囲
            ).execute()
            
            values = result.get('values', [])
            if not values or len(values) < 2:
                return pd.DataFrame()
            
            # ヘッダーとデータを分離
            headers = values[0]
            data = values[1:]
            
            df = pd.DataFrame(data, columns=headers)
            
            return df
            
        except HttpError as e:
            # 郵送データの読み込みエラーは警告に留める
            st.warning(f"郵送フォームデータの読み込みエラー: {e}")
            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()
    
    def get_mail_order_summary(self) -> pd.DataFrame:
        """
        郵送フォームデータを商品別に集計
        
        Returns:
            集計DataFrame（date, 商品名, 販売商品数, 販売総売上）
        """
        df_raw = self.load_mail_orders()
        
        if df_raw.empty:
            return pd.DataFrame()
        
        try:
            # タイムスタンプ列を特定
            timestamp_col = None
            for col in df_raw.columns:
                if 'タイムスタンプ' in col or 'timestamp' in col.lower():
                    timestamp_col = col
                    break
            
            if timestamp_col is None and len(df_raw.columns) > 0:
                timestamp_col = df_raw.columns[0]
            
            if timestamp_col is None:
                return pd.DataFrame()
            
            # 商品列マッピングを取得
            product_columns = getattr(config, 'MAIL_ORDER_PRODUCT_COLUMNS', {})
            
            if not product_columns:
                return pd.DataFrame()
            
            rows = []
            
            for _, record in df_raw.iterrows():
                # 日付を取得
                try:
                    date_val = pd.to_datetime(record[timestamp_col], errors='coerce')
                    if pd.isna(date_val):
                        continue
                except:
                    continue
                
                # 各商品列をチェック
                for form_col, airregi_name in product_columns.items():
                    if form_col in record.index:
                        qty_str = str(record[form_col]).strip()
                        if qty_str and qty_str != '' and qty_str != 'nan' and qty_str != '0':
                            try:
                                qty = int(float(qty_str))
                                if qty > 0:
                                    rows.append({
                                        'date': date_val.normalize(),
                                        '商品名': airregi_name,
                                        '販売商品数': qty,
                                        '販売総売上': 0,  # 郵送は売上不明（後で単価統合）
                                        'source': 'mail_order'
                                    })
                            except (ValueError, TypeError):
                                continue
            
            if not rows:
                return pd.DataFrame()
            
            df_result = pd.DataFrame(rows)
            
            # 同日・同商品を集約
            df_result = df_result.groupby(['date', '商品名', 'source']).agg({
                '販売商品数': 'sum',
                '販売総売上': 'sum'
            }).reset_index()
            
            return df_result
            
        except Exception as e:
            return pd.DataFrame()


def aggregate_by_products(df: pd.DataFrame, product_names: list, aggregate: bool = True) -> pd.DataFrame:
    """商品名でフィルタリング＆集約"""
    if df.empty or not product_names:
        return pd.DataFrame()
    
    if '商品名' not in df.columns:
        return pd.DataFrame()
    
    # 商品名でフィルタ
    mask = df['商品名'].isin(product_names)
    df_filtered = df[mask].copy()
    
    if df_filtered.empty:
        return pd.DataFrame()
    
    if aggregate and 'date' in df_filtered.columns:
        # 日別に集約
        agg_dict = {}
        if '販売商品数' in df_filtered.columns:
            agg_dict['販売商品数'] = 'sum'
        if '販売総売上' in df_filtered.columns:
            agg_dict['販売総売上'] = 'sum'
        if '返品商品数' in df_filtered.columns:
            agg_dict['返品商品数'] = 'sum'
        
        if agg_dict:
            df_filtered = df_filtered.groupby('date').agg(agg_dict).reset_index()
    
    return df_filtered


def merge_with_calendar(df_sales: pd.DataFrame, df_calendar: pd.DataFrame) -> pd.DataFrame:
    """売上データとカレンダーデータをマージ"""
    if df_sales.empty or df_calendar.empty:
        return df_sales
    
    if 'date' not in df_sales.columns or 'date' not in df_calendar.columns:
        return df_sales
    
    df_merged = pd.merge(df_sales, df_calendar, on='date', how='left')
    
    return df_merged
