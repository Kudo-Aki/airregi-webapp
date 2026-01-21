"""
商品名正規化・あいまい検索モジュール

機能:
- 全角半角の統一
- 先頭の数字除去
- 【】内のテキスト処理
- あいまい検索（部分一致）
- 類似商品のグループ化
"""

import re
import unicodedata
from typing import List, Dict, Set, Tuple
from difflib import SequenceMatcher

import pandas as pd


class ProductNormalizer:
    """商品名正規化クラス"""
    
    def __init__(self, 
                 remove_prefix_numbers: bool = True,
                 normalize_width: bool = True,
                 remove_brackets: bool = False):
        """
        Args:
            remove_prefix_numbers: 先頭の数字を除去するか
            normalize_width: 全角半角を統一するか
            remove_brackets: 【】内を除去するか
        """
        self.remove_prefix_numbers = remove_prefix_numbers
        self.normalize_width = normalize_width
        self.remove_brackets = remove_brackets
        
        # 商品マスタ（初期化後にセット）
        self.product_master: pd.DataFrame = None
        self.normalized_names: Dict[str, str] = {}  # 正規化後 → 元の名前
    
    def normalize(self, name: str) -> str:
        """商品名を正規化"""
        if not name or pd.isna(name):
            return ""
        
        result = str(name).strip()
        
        # 全角半角統一（NFKCで正規化）
        if self.normalize_width:
            result = unicodedata.normalize('NFKC', result)
        
        # 先頭の数字を除去（例: "1【月替】初詣" → "【月替】初詣"）
        if self.remove_prefix_numbers:
            result = re.sub(r'^[\d０-９]+\s*', '', result)
        
        # 【】内を除去（オプション）
        if self.remove_brackets:
            result = re.sub(r'【[^】]*】', '', result)
        
        return result.strip()
    
    def normalize_for_search(self, name: str) -> str:
        """検索用に正規化（より緩い正規化）"""
        result = self.normalize(name)
        
        # 検索用にさらに正規化
        result = result.lower()
        result = re.sub(r'\s+', '', result)  # スペース除去
        result = re.sub(r'[【】\[\]（）\(\)]', '', result)  # 括弧除去
        
        return result
    
    def extract_bracket_content(self, name: str) -> Tuple[str, str]:
        """【】内のテキストと残りを分離"""
        match = re.search(r'【([^】]*)】', name)
        if match:
            bracket_content = match.group(1)
            remaining = re.sub(r'【[^】]*】', '', name).strip()
            return bracket_content, remaining
        return "", name
    
    def build_master(self, df: pd.DataFrame, name_column: str = "商品名"):
        """商品マスタを構築"""
        if name_column not in df.columns:
            raise ValueError(f"列 '{name_column}' が見つかりません")
        
        # ユニークな商品名を取得
        unique_names = df[name_column].dropna().unique()
        
        # 正規化マッピングを作成
        self.normalized_names = {}
        for name in unique_names:
            normalized = self.normalize(name)
            if normalized:
                if normalized not in self.normalized_names:
                    self.normalized_names[normalized] = []
                self.normalized_names[normalized].append(name)
        
        # マスタDataFrameを作成
        master_data = []
        for normalized, original_names in self.normalized_names.items():
            bracket, base = self.extract_bracket_content(original_names[0])
            master_data.append({
                "normalized_name": normalized,
                "original_names": original_names,
                "bracket_content": bracket,
                "base_name": self.normalize(base),
                "search_key": self.normalize_for_search(normalized),
            })
        
        self.product_master = pd.DataFrame(master_data)
        return self.product_master
    
    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """
        あいまい検索
        
        Args:
            query: 検索クエリ
            limit: 最大結果数
            
        Returns:
            マッチした商品のリスト（スコア順）
        """
        if self.product_master is None or self.product_master.empty:
            return []
        
        query_normalized = self.normalize_for_search(query)
        
        if not query_normalized:
            # クエリが空の場合は全件返す（上限あり）
            return self.product_master.head(limit).to_dict('records')
        
        results = []
        
        for _, row in self.product_master.iterrows():
            search_key = row['search_key']
            normalized_name = row['normalized_name']
            
            # スコア計算
            score = 0
            
            # 完全一致
            if query_normalized == search_key:
                score = 100
            # 前方一致
            elif search_key.startswith(query_normalized):
                score = 90
            # 後方一致
            elif search_key.endswith(query_normalized):
                score = 80
            # 部分一致
            elif query_normalized in search_key:
                score = 70
            # 逆部分一致（クエリの方が長い場合）
            elif search_key in query_normalized:
                score = 60
            # 類似度（編集距離ベース）
            else:
                similarity = SequenceMatcher(None, query_normalized, search_key).ratio()
                if similarity > 0.5:
                    score = int(similarity * 50)
            
            if score > 0:
                results.append({
                    'normalized_name': normalized_name,
                    'original_names': row['original_names'],
                    'bracket_content': row['bracket_content'],
                    'base_name': row['base_name'],
                    'score': score,
                })
        
        # スコア順でソート
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:limit]
    
    def get_all_original_names(self, normalized_names: List[str]) -> List[str]:
        """正規化名から元の商品名リストを取得"""
        all_names = []
        for norm_name in normalized_names:
            matches = self.product_master[
                self.product_master['normalized_name'] == norm_name
            ]
            if not matches.empty:
                all_names.extend(matches.iloc[0]['original_names'])
        return all_names


class ProductGroupManager:
    """商品グループ管理クラス"""
    
    def __init__(self):
        self.groups: Dict[str, Set[str]] = {
            'A': set(),  # 合算グループ
            'B': set(),  # 個別グループ
        }
    
    def add_to_group(self, group_name: str, product_name: str):
        """商品をグループに追加"""
        if group_name in self.groups:
            # 他のグループから削除
            for g in self.groups:
                if g != group_name:
                    self.groups[g].discard(product_name)
            # 指定グループに追加
            self.groups[group_name].add(product_name)
    
    def remove_from_group(self, group_name: str, product_name: str):
        """商品をグループから削除"""
        if group_name in self.groups:
            self.groups[group_name].discard(product_name)
    
    def get_group(self, group_name: str) -> Set[str]:
        """グループの商品を取得"""
        return self.groups.get(group_name, set())
    
    def clear_all(self):
        """全グループをクリア"""
        for g in self.groups:
            self.groups[g] = set()
    
    def to_dict(self) -> Dict[str, List[str]]:
        """辞書形式で取得"""
        return {k: list(v) for k, v in self.groups.items()}
    
    def from_dict(self, data: Dict[str, List[str]]):
        """辞書から復元"""
        for k, v in data.items():
            if k in self.groups:
                self.groups[k] = set(v)
