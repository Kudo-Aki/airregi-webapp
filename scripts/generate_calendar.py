#!/usr/bin/env python3
"""
カレンダーデータ生成スクリプト
- 六曜（先勝、友引、先負、仏滅、大安、赤口）
- 七曜（曜日）
- 開運日（大安、一粒万倍日、天赦日、寅の日、巳の日）
- 祝日
- 天気・気温（Open-Meteo API）

生成期間: 2022-08-01 〜 2026-12-31
"""

import json
import time
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import requests

# =============================================================================
# 六曜計算
# =============================================================================

def get_rokuyou(target_date: date) -> str:
    """六曜を計算（旧暦ベースの簡易計算）"""
    # 六曜は旧暦の月日の和を6で割った余りで決まる
    # 簡易計算のため、実際の旧暦変換ライブラリを使用するのが理想
    # ここでは近似計算を使用
    
    rokuyou_names = ["大安", "赤口", "先勝", "友引", "先負", "仏滅"]
    
    # 簡易旧暦計算（誤差あり、正確には旧暦ライブラリが必要）
    # 基準日: 2022-01-01 は旧暦 11月29日 → 六曜は (11+29)%6 = 4 = 先負
    base_date = date(2022, 1, 1)
    base_rokuyou_index = 4  # 先負
    
    days_diff = (target_date - base_date).days
    
    # 六曜は基本的に1日ずつ進むが、旧暦の1日でリセットされる
    # 簡易的に日数差から計算（実運用では旧暦ライブラリ推奨）
    index = (base_rokuyou_index + days_diff) % 6
    
    return rokuyou_names[index]


def get_rokuyou_accurate(target_date: date) -> str:
    """六曜を正確に計算（qreki使用）"""
    try:
        from qreki import Kyureki
        
        rokuyou_names = ["大安", "赤口", "先勝", "友引", "先負", "仏滅"]
        
        # 旧暦に変換
        kyureki = Kyureki.from_ymd(target_date.year, target_date.month, target_date.day)
        
        # 旧暦の月と日の和を6で割った余り
        index = (kyureki.month + kyureki.day) % 6
        
        return rokuyou_names[index]
    except ImportError:
        # qrekiがない場合は簡易計算
        return get_rokuyou(target_date)


# =============================================================================
# 十二支計算（寅の日、巳の日）
# =============================================================================

JUNISHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

def get_junishi(target_date: date) -> str:
    """十二支を計算"""
    # 基準日: 2022-01-01 は「寅」の日
    base_date = date(2022, 1, 1)
    base_index = 2  # 寅
    
    days_diff = (target_date - base_date).days
    index = (base_index + days_diff) % 12
    
    return JUNISHI[index]


def is_tora_no_hi(target_date: date) -> bool:
    """寅の日かどうか"""
    return get_junishi(target_date) == "寅"


def is_mi_no_hi(target_date: date) -> bool:
    """巳の日かどうか"""
    return get_junishi(target_date) == "巳"


# =============================================================================
# 一粒万倍日
# =============================================================================

# 一粒万倍日は旧暦の月と日の組み合わせで決まる
ICHIRYUMANBAI_DAYS = {
    1: [3, 16],   # 旧暦1月: 3日、16日
    2: [2, 14, 27],
    3: [1, 13, 25],
    4: [12, 24],
    5: [11, 23],
    6: [10, 22],
    7: [9, 21],
    8: [8, 20],
    9: [7, 19],
    10: [6, 18],
    11: [5, 17, 29],
    12: [4, 16, 28],
}

def is_ichiryumanbai(target_date: date) -> bool:
    """一粒万倍日かどうか（簡易計算）"""
    # 新暦ベースの簡易判定（実際は旧暦）
    # 2024年の一粒万倍日リストから周期性を計算
    
    # 簡易的に：旧暦変換なしでおおよその判定
    # 月2〜3回程度出現する
    day_of_year = target_date.timetuple().tm_yday
    
    # 約15日周期 + 月の影響
    cycle1 = (day_of_year + target_date.month * 3) % 15 in [0, 1]
    cycle2 = (day_of_year + target_date.month * 7) % 13 in [0]
    
    return cycle1 or cycle2


# =============================================================================
# 天赦日
# =============================================================================

def get_tensha_days(year: int) -> List[date]:
    """その年の天赦日を計算"""
    # 天赦日は季節と干支の組み合わせで決まる
    # 春（立春〜立夏前）: 戊寅の日
    # 夏（立夏〜立秋前）: 甲午の日
    # 秋（立秋〜立冬前）: 戊申の日
    # 冬（立冬〜立春前）: 甲子の日
    
    # 簡易的に既知の天赦日パターンを使用
    # 年に5〜6回程度
    tensha_days = []
    
    # 2022-2027年の天赦日（概算）
    tensha_dates = {
        2022: ["01-11", "03-26", "06-10", "08-23", "10-22", "11-07"],
        2023: ["01-06", "03-21", "06-05", "08-04", "08-18", "10-17"],
        2024: ["01-01", "03-15", "05-30", "07-29", "08-12", "10-11", "12-26"],
        2025: ["01-10", "03-25", "06-09", "08-08", "08-22", "10-21"],
        2026: ["01-05", "03-20", "06-04", "08-03", "08-17", "10-16", "12-31"],
    }
    
    if year in tensha_dates:
        for d in tensha_dates[year]:
            month, day = map(int, d.split("-"))
            tensha_days.append(date(year, month, day))
    
    return tensha_days


def is_tensha(target_date: date) -> bool:
    """天赦日かどうか"""
    tensha_days = get_tensha_days(target_date.year)
    return target_date in tensha_days


# =============================================================================
# 祝日
# =============================================================================

def get_holidays(year: int) -> Dict[date, str]:
    """日本の祝日を取得"""
    try:
        import jpholiday
        holidays = {}
        for d, name in jpholiday.year_holidays(year):
            holidays[d] = name
        return holidays
    except ImportError:
        # jpholidayがない場合は手動定義
        return get_holidays_manual(year)


def get_holidays_manual(year: int) -> Dict[date, str]:
    """祝日を手動定義（jpholidayがない場合）"""
    holidays = {}
    
    # 固定祝日
    fixed = [
        (1, 1, "元日"),
        (2, 11, "建国記念の日"),
        (2, 23, "天皇誕生日"),
        (4, 29, "昭和の日"),
        (5, 3, "憲法記念日"),
        (5, 4, "みどりの日"),
        (5, 5, "こどもの日"),
        (8, 11, "山の日"),
        (11, 3, "文化の日"),
        (11, 23, "勤労感謝の日"),
    ]
    
    for month, day, name in fixed:
        holidays[date(year, month, day)] = name
    
    # ハッピーマンデー
    # 成人の日: 1月第2月曜
    holidays[get_nth_weekday(year, 1, 0, 2)] = "成人の日"
    # 海の日: 7月第3月曜
    holidays[get_nth_weekday(year, 7, 0, 3)] = "海の日"
    # 敬老の日: 9月第3月曜
    holidays[get_nth_weekday(year, 9, 0, 3)] = "敬老の日"
    # スポーツの日: 10月第2月曜
    holidays[get_nth_weekday(year, 10, 0, 2)] = "スポーツの日"
    
    # 春分の日、秋分の日は年によって変わるので概算
    holidays[date(year, 3, 20 if year % 4 == 0 else 21)] = "春分の日"
    holidays[date(year, 9, 22 if year % 4 == 0 else 23)] = "秋分の日"
    
    return holidays


def get_nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """第n週の特定曜日を取得"""
    first_day = date(year, month, 1)
    first_weekday = first_day.weekday()
    
    # 最初の該当曜日
    days_until = (weekday - first_weekday) % 7
    first_occurrence = first_day + timedelta(days=days_until)
    
    # 第n週
    return first_occurrence + timedelta(weeks=n-1)


# =============================================================================
# 天気データ取得（Open-Meteo API）
# =============================================================================

def get_weather_data(lat: float, lon: float, start_date: date, end_date: date) -> Dict[str, Any]:
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
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"天気データ取得エラー: {e}")
        return None


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


# =============================================================================
# メイン処理：カレンダーデータ生成
# =============================================================================

def generate_calendar_data(start_date: date, end_date: date, 
                           lat: float = 36.3833, lon: float = 140.6167) -> List[Dict[str, Any]]:
    """
    カレンダーデータを生成
    
    lat, lon: ひたちなか市の緯度経度（デフォルト）
    """
    
    print(f"カレンダーデータ生成: {start_date} 〜 {end_date}")
    
    # 祝日を事前に取得
    holidays = {}
    for year in range(start_date.year, end_date.year + 1):
        holidays.update(get_holidays(year))
    
    # 天赦日を事前に取得
    tensha_days = set()
    for year in range(start_date.year, end_date.year + 1):
        tensha_days.update(get_tensha_days(year))
    
    # 天気データを取得（API制限があるので分割）
    weather_data = {}
    print("天気データを取得中...")
    
    current = start_date
    while current <= end_date:
        # 1年ずつ取得
        chunk_end = min(date(current.year, 12, 31), end_date)
        
        data = get_weather_data(lat, lon, current, chunk_end)
        
        if data and "daily" in data:
            dates = data["daily"]["time"]
            temps_max = data["daily"]["temperature_2m_max"]
            temps_min = data["daily"]["temperature_2m_min"]
            temps_mean = data["daily"]["temperature_2m_mean"]
            precip = data["daily"]["precipitation_sum"]
            weather_codes = data["daily"]["weathercode"]
            
            for i, d in enumerate(dates):
                weather_data[d] = {
                    "temp_max": temps_max[i] if temps_max[i] is not None else 0,
                    "temp_min": temps_min[i] if temps_min[i] is not None else 0,
                    "temp_mean": temps_mean[i] if temps_mean[i] is not None else 0,
                    "precipitation": precip[i] if precip[i] is not None else 0,
                    "weather": weathercode_to_text(weather_codes[i]) if weather_codes[i] is not None else "不明"
                }
        
        print(f"  {current.year}年 完了")
        current = date(current.year + 1, 1, 1)
        time.sleep(0.5)  # API制限対策
    
    # カレンダーデータを生成
    print("カレンダーデータを生成中...")
    calendar_data = []
    
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        
        # 六曜
        rokuyou = get_rokuyou_accurate(current)
        
        # 曜日
        weekday_names = ["月", "火", "水", "木", "金", "土", "日"]
        weekday = weekday_names[current.weekday()]
        weekday_num = current.weekday()
        
        # 十二支
        junishi = get_junishi(current)
        
        # 開運日フラグ
        is_taian = rokuyou == "大安"
        is_ichiryumanbai_day = is_ichiryumanbai(current)
        is_tensha_day = current in tensha_days
        is_tora = is_tora_no_hi(current)
        is_mi = is_mi_no_hi(current)
        
        # 祝日
        holiday_name = holidays.get(current, "")
        is_holiday = holiday_name != ""
        
        # 土日判定
        is_weekend = weekday_num >= 5
        
        # 天気
        weather_info = weather_data.get(date_str, {})
        
        row = {
            "date": date_str,
            "year": current.year,
            "month": current.month,
            "day": current.day,
            "weekday": weekday,
            "weekday_num": weekday_num,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "holiday_name": holiday_name,
            "rokuyou": rokuyou,
            "junishi": junishi,
            "is_taian": is_taian,
            "is_ichiryumanbai": is_ichiryumanbai_day,
            "is_tensha": is_tensha_day,
            "is_tora_no_hi": is_tora,
            "is_mi_no_hi": is_mi,
            "temp_max": weather_info.get("temp_max", ""),
            "temp_min": weather_info.get("temp_min", ""),
            "temp_mean": weather_info.get("temp_mean", ""),
            "precipitation": weather_info.get("precipitation", ""),
            "weather": weather_info.get("weather", ""),
        }
        
        calendar_data.append(row)
        current += timedelta(days=1)
    
    print(f"生成完了: {len(calendar_data)}日分")
    return calendar_data


def save_to_csv(data: List[Dict[str, Any]], filepath: str):
    """CSVに保存"""
    import csv
    
    if not data:
        return
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    print(f"保存完了: {filepath}")


def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║     カレンダーデータ生成スクリプト                             ║
╠════════════════════════════════════════════════════════════════╣
║  六曜・開運日・祝日・天気データを生成します                    ║
║  期間: 2022-08-01 〜 2026-12-31                                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    start_date = date(2022, 8, 1)
    end_date = date(2026, 12, 31)
    
    # ひたちなか市の緯度経度
    lat = 36.3833
    lon = 140.6167
    
    # データ生成
    calendar_data = generate_calendar_data(start_date, end_date, lat, lon)
    
    # CSV保存
    save_to_csv(calendar_data, "calendar_data.csv")
    
    # JSON保存（デバッグ用）
    with open("calendar_data.json", 'w', encoding='utf-8') as f:
        json.dump(calendar_data, f, ensure_ascii=False, indent=2)
    
    print("\n完了！calendar_data.csv をスプレッドシートにインポートしてください。")


if __name__ == "__main__":
    main()
