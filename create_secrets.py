#!/usr/bin/env python3
"""
サービスアカウントJSONからsecrets.tomlを生成（修正版）

使い方:
  python create_secrets.py airregi-csv-automation-d19ec6c116ff.json
"""

import json
import sys
from pathlib import Path


def create_secrets_toml(json_path: str):
    """JSONファイルからsecrets.tomlを生成"""
    
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"エラー: ファイルが見つかりません: {json_path}")
        return
    
    # JSONを読み込み
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # secrets.toml形式に変換
    toml_lines = ["[gcp_service_account]"]
    
    for key, value in data.items():
        if isinstance(value, str):
            if '\n' in value:
                # 改行を含む文字列（private_key等）はトリプルクォートを使用
                toml_lines.append(f'{key} = """{value}"""')
            else:
                # 通常の文字列
                toml_lines.append(f'{key} = "{value}"')
        else:
            toml_lines.append(f'{key} = {json.dumps(value)}')
    
    toml_content = '\n'.join(toml_lines)
    
    # 出力先
    output_dir = Path(".streamlit")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "secrets.toml"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(toml_content)
    
    print(f"✅ 作成完了: {output_path}")
    print(f"\n注意: このファイルは機密情報を含むため、Gitにコミットしないでください！")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python create_secrets.py <サービスアカウントキー.json>")
        print("例: python create_secrets.py airregi-csv-automation-d19ec6c116ff.json")
    else:
        create_secrets_toml(sys.argv[1])
