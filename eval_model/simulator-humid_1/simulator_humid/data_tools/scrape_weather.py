if __package__ in (None, ""):
    import pathlib
    import sys

    _pkg_dir = pathlib.Path(__file__).resolve().parent
    while _pkg_dir.name != "simulator_humid" and _pkg_dir.parent != _pkg_dir:
        _pkg_dir = _pkg_dir.parent
    if _pkg_dir.name == "simulator_humid":
        _project_root = _pkg_dir.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        del _project_root
    del _pkg_dir

import os
import ssl
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from simulator_humid.data_tools.absolute_humidity import (
    calculate_absolute_humidity_g_per_kg,
)
from simulator_humid.utils.paths import WEATHER_DATA_DIR

# SSL証明書の検証を無効にする（開発環境でのみ使用）
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# urllibのデフォルトSSLコンテキストを設定
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))

# weather_dataディレクトリの作成
weather_data_dir = WEATHER_DATA_DIR
weather_data_dir.mkdir(exist_ok=True)

# 気象データを取得（7月の全31日分）
for day in range(1, 32):
    try:
        print(f"7月{day}日のデータを取得中...")
        df = pd.read_html(f"https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no=44&block_no=47662&year=2025&month=7&day={day}&view=")
        
        # データフレームから気温と相対湿度を取得
        temp_c = df[0]["気温 (℃)"]["気温 (℃)"]
        relative_humidity = df[0]["相対湿度 (％)"]["相対湿度 (％)"]
        time_data = df[0]["時分"]["時分"]
        
        # 1日分のデータを格納するリスト
        daily_data = []
        
        # データを処理
        for i in range(len(temp_c)):
            if pd.notna(temp_c.iloc[i]) and pd.notna(relative_humidity.iloc[i]):
                # 時間の処理
                time_str = str(time_data.iloc[i])
                if len(time_str) == 4:  # HHMM形式
                    time_formatted = f"{time_str[:2]}:{time_str[2:]}"
                else:
                    time_formatted = time_str
                
                # 分の計算（10分間隔）
                minutes = i * 10
                
                # 絶対湿度の計算（add_absolute_humidity.pyの関数を使用）
                abs_humidity = calculate_absolute_humidity_g_per_kg(temp_c.iloc[i], relative_humidity.iloc[i])
                
                daily_data.append({
                    'time': time_formatted,
                    'temp_c': temp_c.iloc[i],
                    'minutes': minutes,
                    'relative_humidity': relative_humidity.iloc[i],
                    'absolute_humidity_g_per_kg': abs_humidity
                })
        
        # 1日分のデータフレームを作成
        daily_df = pd.DataFrame(daily_data)
        
        # ファイル名を生成（例：outdoor_temp_20250701.csv）
        filename = f"outdoor_temp_202507{day:02d}.csv"
        filepath = weather_data_dir / filename
        
        # CSVファイルに保存
        daily_df.to_csv(filepath, index=False)
        print(f"  → {filename} に保存完了（{len(daily_df)}行）")
        
    except Exception as e:
        print(f"7月{day}日のデータ取得でエラー: {e}")
        continue

print(f"\nデータ取得完了！")
print(f"保存先ディレクトリ: {weather_data_dir}")
print(f"保存されたファイル数: {len(list(weather_data_dir.glob('*.csv')))}")
