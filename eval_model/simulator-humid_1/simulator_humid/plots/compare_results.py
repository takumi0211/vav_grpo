from __future__ import annotations

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

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

plt.switch_backend("Agg")


def plot_comparison(
    output_path: Path,
    *,
    baseline_csv: Path,
    llm_csv: Path,
    column_names: list[str] | None = None,
    figsize: tuple[float, float] = (12, 16),
    comparison_label: str = "LLM",
) -> None:
    """2つのシミュレーション結果のCSVから指定された列を比較してプロット。
    
    Parameters
    ----------
    output_path : Path
        出力画像のパス
    baseline_csv : Path
        ベースラインシミュレーション結果のCSVファイルパス
    llm_csv : Path
        比較対象のシミュレーション結果のCSVファイルパス
    column_names : list[str] | None
        比較する列名のリスト（デフォルト: ["fan_inv", "supply_temp", "fan_power_kw", "chw_pump_power_kw", "co2_avg_ppm", "co2_max_ppm"]）
    figsize : tuple[float, float]
        図のサイズ
    comparison_label : str
        比較対象のラベル名（デフォルト: "LLM"）
    """
    if column_names is None:
        column_names = ["fan_inv", "supply_temp", "fan_power_kw", "chw_pump_power_kw", "co2_avg_ppm", "co2_max_ppm"]
    
    # CSVファイルを読み込む
    df_baseline = pd.read_csv(baseline_csv)
    df_llm = pd.read_csv(llm_csv)
    
    # CO2濃度の平均値と最大値を計算
    zone_co2_columns = ["zone1_co2_ppm", "zone2_co2_ppm", "zone3_co2_ppm", "zone4_co2_ppm"]
    if all(col in df_baseline.columns for col in zone_co2_columns):
        df_baseline["co2_avg_ppm"] = df_baseline[zone_co2_columns].mean(axis=1)
        df_baseline["co2_max_ppm"] = df_baseline[zone_co2_columns].max(axis=1)
    if all(col in df_llm.columns for col in zone_co2_columns):
        df_llm["co2_avg_ppm"] = df_llm[zone_co2_columns].mean(axis=1)
        df_llm["co2_max_ppm"] = df_llm[zone_co2_columns].max(axis=1)
    
    # 指定された列が存在するか確認
    for column_name in column_names:
        if column_name not in df_baseline.columns:
            raise ValueError(f"列 '{column_name}' がベースラインCSVに見つかりません")
        if column_name not in df_llm.columns:
            raise ValueError(f"列 '{column_name}' がLLM CSVに見つかりません")
    
    # time列をdatetime型に変換
    # LLMのCSVにtime列がある場合はそれを使用
    if "time" in df_llm.columns:
        df_llm["time"] = pd.to_datetime(df_llm["time"])
        time_data = df_llm["time"]
        
        # ベースラインにtime列がない場合でも、同じ長さならLLMのtime列を使用
        if "time" in df_baseline.columns:
            df_baseline["time"] = pd.to_datetime(df_baseline["time"])
            time_baseline = df_baseline["time"]
        elif len(df_baseline) == len(df_llm):
            time_baseline = time_data
        else:
            time_baseline = range(len(df_baseline))
        
        time_llm = time_data
    elif "time" in df_baseline.columns:
        df_baseline["time"] = pd.to_datetime(df_baseline["time"])
        time_baseline = df_baseline["time"]
        time_llm = time_baseline if len(df_llm) == len(df_baseline) else range(len(df_llm))
    else:
        time_baseline = range(len(df_baseline))
        time_llm = range(len(df_llm))
    
    # サブプロット数を決定
    n_plots = len(column_names)
    
    # プロット作成
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    
    # 1つの列の場合でもリストとして扱えるようにする
    if n_plots == 1:
        axes = [axes]
    
    # 横軸が時刻かどうかを判定
    use_datetime = (isinstance(time_baseline, pd.Series) and pd.api.types.is_datetime64_any_dtype(time_baseline)) or \
                   (isinstance(time_llm, pd.Series) and pd.api.types.is_datetime64_any_dtype(time_llm))
    
    for idx, column_name in enumerate(column_names):
        ax = axes[idx]
        
        ax.plot(
            time_baseline,
            df_baseline[column_name],
            label="Baseline",
            alpha=0.8,
            linewidth=1.2,
        )
        ax.plot(
            time_llm,
            df_llm[column_name],
            label=comparison_label,
            alpha=0.8,
            linewidth=1.2,
        )
        
        # 横軸がdatetimeの場合、フォーマットを設定
        if use_datetime:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.set_xlabel("Time")
            
            # 横軸の範囲を0-24時に固定
            if isinstance(time_baseline, pd.Series) and len(time_baseline) > 0:
                start_date = time_baseline.iloc[0].replace(hour=0, minute=0, second=0)
                end_date = start_date + pd.Timedelta(hours=24)
                ax.set_xlim(start_date, end_date)
        else:
            ax.set_xlabel("Time Step")
        
        ax.set_ylabel(column_name)
        ax.set_title(f"Comparison: {column_name}")
        ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)
        
        # CO2濃度のプロットには1000ppmの基準線を追加
        if "co2" in column_name.lower() and "ppm" in column_name.lower():
            ax.axhline(y=1000, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='1000 ppm threshold')
            ax.legend()
        else:
            ax.legend()
    
    # サブプロット間のスペースを調整
    fig.tight_layout(h_pad=3.0)
    
    # 日付軸の場合、ラベルを斜めに回転（tight_layoutの後に実行）
    if use_datetime:
        for ax in axes:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
    
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # 統計情報を表示
    print(f"比較プロットを保存しました: {output_path}")
    print(f"\n統計情報:")
    for column_name in column_names:
        baseline_mean = df_baseline[column_name].mean()
        llm_mean = df_llm[column_name].mean()
        baseline_std = df_baseline[column_name].std()
        llm_std = df_llm[column_name].std()
        
        print(f"\n{column_name}:")
        print(f"  Baseline: 平均 = {baseline_mean:.4f}, 標準偏差 = {baseline_std:.4f}")
        print(f"  {comparison_label}:      平均 = {llm_mean:.4f}, 標準偏差 = {llm_std:.4f}")


def plot_cumulative_kwh_comparison(
    output_path: Path,
    *,
    baseline_csv: Path,
    llm_csv: Path,
    figsize: tuple[float, float] = (8, 13),
    comparison_label: str = "LLM",
) -> None:
    """2つのシミュレーション結果のCSVからファン、ポンプ、チラーの積算負荷kWhを比較してプロット。
    
    Parameters
    ----------
    output_path : Path
        出力画像のパス
    baseline_csv : Path
        ベースラインシミュレーション結果のCSVファイルパス
    llm_csv : Path
        比較対象のシミュレーション結果のCSVファイルパス
    figsize : tuple[float, float]
        図のサイズ
    comparison_label : str
        比較対象のラベル名（デフォルト: "LLM"）
    """
    # CSVファイルを読み込む
    df_baseline = pd.read_csv(baseline_csv)
    df_llm = pd.read_csv(llm_csv)
    
    # 電力列のリスト
    power_columns = {
        "fan_power_kw": "Fan",
        "chw_pump_power_kw": "Pump",
        "chiller_power_kw": "Chiller"
    }
    
    # 列が存在するか確認
    for column_name in power_columns.keys():
        if column_name not in df_baseline.columns:
            raise ValueError(f"列 '{column_name}' がベースラインCSVに見つかりません")
        if column_name not in df_llm.columns:
            raise ValueError(f"列 '{column_name}' がLLM CSVに見つかりません")
    
    # 積算値を計算（kW * 1min / 60 = kWh）
    # 1分間隔と仮定
    time_step_hours = 1.0 / 60.0
    
    for column_name in power_columns.keys():
        df_baseline[f"{column_name}_cumsum"] = (df_baseline[column_name] * time_step_hours).cumsum()
        df_llm[f"{column_name}_cumsum"] = (df_llm[column_name] * time_step_hours).cumsum()
    
    # time列をdatetime型に変換
    if "time" in df_llm.columns:
        df_llm["time"] = pd.to_datetime(df_llm["time"])
        time_data = df_llm["time"]
        
        if "time" in df_baseline.columns:
            df_baseline["time"] = pd.to_datetime(df_baseline["time"])
            time_baseline = df_baseline["time"]
        elif len(df_baseline) == len(df_llm):
            time_baseline = time_data
        else:
            time_baseline = range(len(df_baseline))
        
        time_llm = time_data
    elif "time" in df_baseline.columns:
        df_baseline["time"] = pd.to_datetime(df_baseline["time"])
        time_baseline = df_baseline["time"]
        time_llm = time_baseline if len(df_llm) == len(df_baseline) else range(len(df_llm))
    else:
        time_baseline = range(len(df_baseline))
        time_llm = range(len(df_llm))
    
    # プロット作成（3つの時系列 + 1つの棒グラフ）
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # 横軸が時刻かどうかを判定
    use_datetime = (isinstance(time_baseline, pd.Series) and pd.api.types.is_datetime64_any_dtype(time_baseline)) or \
                   (isinstance(time_llm, pd.Series) and pd.api.types.is_datetime64_any_dtype(time_llm))
    
    # 時系列プロット（上3つ）
    for idx, (column_name, label) in enumerate(power_columns.items()):
        ax = axes[idx]
        cumsum_column = f"{column_name}_cumsum"
        
        ax.plot(
            time_baseline,
            df_baseline[cumsum_column],
            label="Baseline",
            alpha=0.8,
            linewidth=1.5,
        )
        ax.plot(
            time_llm,
            df_llm[cumsum_column],
            label=comparison_label,
            alpha=0.8,
            linewidth=1.5,
        )
        
        # 横軸がdatetimeの場合、フォーマットを設定
        if use_datetime:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.set_xlabel("Time")
            
            # 横軸の範囲を08:00-18:00に固定
            if isinstance(time_baseline, pd.Series) and len(time_baseline) > 0:
                start_date = time_baseline.iloc[0].replace(hour=8, minute=0, second=0)
                end_date = time_baseline.iloc[0].replace(hour=18, minute=0, second=0)
                ax.set_xlim(start_date, end_date)
        else:
            ax.set_xlabel("Time Step")
        
        ax.set_ylabel("Cumulative Energy (kWh)")
        ax.set_title(f"{label} Cumulative Energy Consumption")
        ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.legend()
    
    # 棒グラフ（一番下）
    ax_bar = axes[3]
    
    # 合計値を計算
    totals_baseline = []
    totals_llm = []
    labels = []
    
    for column_name, label in power_columns.items():
        cumsum_column = f"{column_name}_cumsum"
        totals_baseline.append(df_baseline[cumsum_column].iloc[-1])
        totals_llm.append(df_llm[cumsum_column].iloc[-1])
        labels.append(label)
    
    # 合計を追加
    totals_baseline.append(sum(totals_baseline))
    totals_llm.append(sum(totals_llm))
    labels.append("Total")
    
    import numpy as np
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, totals_baseline, width, label='Baseline', alpha=0.8)
    bars2 = ax_bar.bar(x + width/2, totals_llm, width, label=comparison_label, alpha=0.8)
    
    ax_bar.set_xlabel('Equipment')
    ax_bar.set_ylabel('Total Energy (kWh)')
    ax_bar.set_title('Total Energy Consumption Comparison (24h)')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.legend()
    ax_bar.grid(True, axis='y', linestyle=":", linewidth=0.8, alpha=0.6)
    ax_bar.set_ylim(0, 120)
    
    # 棒グラフの上に数値を表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
    
    # サブプロット間のスペースを調整
    fig.tight_layout(h_pad=3.0)
    
    # 日付軸の場合、ラベルを斜めに回転（tight_layoutの後に実行）
    if use_datetime:
        for ax in axes[:3]:  # 時系列プロットのみ
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
    
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # 統計情報を表示
    print(f"積算負荷kWh比較プロットを保存しました: {output_path}")
    print(f"\n統計情報 (24時間合計):")
    for column_name, label in power_columns.items():
        cumsum_column = f"{column_name}_cumsum"
        baseline_total = df_baseline[cumsum_column].iloc[-1]
        llm_total = df_llm[cumsum_column].iloc[-1]
        diff = llm_total - baseline_total
        diff_pct = (diff / baseline_total * 100) if baseline_total != 0 else 0
        
        print(f"\n{label}:")
        print(f"  Baseline: {baseline_total:.2f} kWh")
        print(f"  {comparison_label}:      {llm_total:.2f} kWh")
        print(f"  差分:     {diff:+.2f} kWh ({diff_pct:+.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2つのシミュレーション結果CSVファイルから指定された列を比較してプロット"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/baseline/baseline_pid/simulation_results.csv"),
        help="ベースラインシミュレーション結果のCSVファイルパス",
    )
    parser.add_argument(
        "--llm",
        type=Path,
        default=Path("outputs/rl/td3_eval/td3_simulation_results.csv"),
        help="LLMシミュレーション結果のCSVファイルパス",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="比較する列名（複数指定可能、デフォルト: fan_inv supply_temp fan_power_kw chw_pump_power_kw co2_avg_ppm co2_max_ppm）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/comparison_td3.png"),
        help="出力画像ファイルパス",
    )
    parser.add_argument(
        "--kwh",
        type=lambda x: x.lower() == 'true',
        default=False,
        help="積算負荷kWhの比較プロットを生成 (true/false)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="TD3",
        help="比較対象のラベル名（デフォルト: LLM）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 相対パスを絶対パスに変換（プロジェクトルートからの相対パス）
    project_root = Path(__file__).resolve().parent.parent.parent
    
    baseline_path = args.baseline if args.baseline.is_absolute() else project_root / args.baseline
    llm_path = args.llm if args.llm.is_absolute() else project_root / args.llm
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    
    # 出力ディレクトリが存在しない場合は作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.kwh:
        # 積算kWhのプロットを生成
        # 出力ファイル名を変更（_kwh を追加）
        kwh_output_path = output_path.parent / output_path.name.replace(".png", "_kwh.png")
        if kwh_output_path == output_path:
            kwh_output_path = output_path.parent / (output_path.stem + "_kwh.png")
        
        plot_cumulative_kwh_comparison(
            kwh_output_path,
            baseline_csv=baseline_path,
            llm_csv=llm_path,
            comparison_label=args.label,
        )
    else:
        plot_comparison(
            output_path,
            baseline_csv=baseline_path,
            llm_csv=llm_path,
            column_names=args.columns,
            comparison_label=args.label,
        )


if __name__ == "__main__":
    main()

