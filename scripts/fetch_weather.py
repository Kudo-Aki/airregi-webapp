#!/usr/bin/env python3
"""
天気データ取得スクリプト（Open-Meteo API）

ひたちなか市の過去の天気データを取得して calendar_data.csv に追加します。
"""

import csv
import time
from datetime import date, datetime
from pathlib import Path
import requests


def weathercode_to_text(code: int) -> str:
    """天気コードをテキストに変換"""
    weather_codes = {
        0: "晴れ",
        1: "晴れ", 2: "晴れ", 3: "曇り",
        45: "霧", 48: "霧",
        51: "小雨", 53: "雨", 55: "雨",
        56: "雨", 57: "雨",
        61: "雨", 63: "雨", 65: "大雨",
        66: "雨", 67: "雨",
        71: "雪", 73: "雪", 75: "大雪",
        77: "雪",
        80: "にわか雨", 81: "にわか雨", 82: "にわか雨",
        85: "雪", 86: "雪",
        95: "雷雨", 96: "雷雨", 99: "雷雨",
    }
    return weather_codes.get(code, "不明")


def get_weather_data(lat: float, lon: float, start_date: date, end_date: date):
    """Open-Meteo APIから過去の天気データを取得"""
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,weathercode",
        "timezone": "Asia/Tokyo"
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"エラー: {e}")
        return None


def update_calendar_with_weather(csv_file: str = "calendar_data.csv"):
    """calendar_data.csvに天気データを追加"""
    
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"エラー: {csv_file} が見つかりません")
        print("先に generate_calendar.py を実行してください")
        return
    
    # 既存データを読み込み
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    print(f"読み込み完了: {len(rows)}行")
    
    # 日付範囲を取得
    dates = [row['date'] for row in rows]
    start_date = date.fromisoformat(min(dates))
    end_date = date.fromisoformat(max(dates))
    
    # ひたちなか市の緯度経度
    lat = 36.3833
    lon = 140.6167
    
    # 天気データを取得（1年ずつ）
    print("天気データを取得中...")
    weather_data = {}
    
    current = start_date
    while current <= end_date:
        chunk_end = min(date(current.year, 12, 31), end_date)
        
        print(f"  {current.year}年を取得中...")
        
        data = get_weather_data(lat, lon, current, chunk_end)
        
        if data and "daily" in data:
            dates_list = data["daily"]["time"]
            temps_max = data["daily"]["temperature_2m_max"]
            temps_min = data["daily"]["temperature_2m_min"]
            temps_mean = data["daily"]["temperature_2m_mean"]
            precip = data["daily"]["precipitation_sum"]
            weather_codes = data["daily"]["weathercode"]
            
            for i, d in enumerate(dates_list):
                weather_data[d] = {
                    "temp_max": temps_max[i] if temps_max[i] is not None else "",
                    "temp_min": temps_min[i] if temps_min[i] is not None else "",
                    "temp_mean": temps_mean[i] if temps_mean[i] is not None else "",
                    "precipitation": precip[i] if precip[i] is not None else "",
                    "weather": weathercode_to_text(weather_codes[i]) if weather_codes[i] is not None else ""
                }
            
            print(f"    完了: {len(dates_list)}日分")
        else:
            print(f"    取得失敗")
        
        current = date(current.year + 1, 1, 1)
        time.sleep(1)  # API制限対策
    
    # 天気データをマージ
    print("\nデータをマージ中...")
    updated = 0
    for row in rows:
        d = row['date']
        if d in weather_data:
            row['temp_max'] = weather_data[d]['temp_max']
            row['temp_min'] = weather_data[d]['temp_min']
            row['temp_mean'] = weather_data[d]['temp_mean']
            row['precipitation'] = weather_data[d]['precipitation']
            row['weather'] = weather_data[d]['weather']
            updated += 1
    
    # 保存
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n完了！{updated}日分の天気データを追加しました")
    print(f"ファイル: {csv_path}")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════╗
║     天気データ取得スクリプト                                   ║
╠════════════════════════════════════════════════════════════════╣
║  Open-Meteo APIから過去の天気データを取得します                ║
║  対象: ひたちなか市（緯度36.3833, 経度140.6167）               ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    update_calendar_with_weather()
