# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 10:20:11 2025

@author: 高文萱
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 10:20:11 2025

@author: 高文萱
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 0) 相对路径设置 =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))              # 当前脚本所在目录（EAN-11523777-am）
PARQUET_PATH = os.path.join(BASE_DIR, "all_years_cleaned.parquet") # 同目录下的 parquet
OUTPUT_DIR = os.path.join(BASE_DIR, "model_outputs")               # 输出到同目录下 model_outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= 1) 读取 =========
df = pd.read_parquet(PARQUET_PATH)

# 若索引不是时间类型，则尝试自动识别一个“时间”列并设为索引
if not isinstance(df.index, pd.DatetimeIndex):
    time_col = None
    for c in df.columns:
        cl = c.lower()
        if "time" in cl or "date" in cl or "datetime" in cl:
            time_col = c
            break
    if time_col is None:
        raise ValueError("未找到时间列。请确认 parquet 中是否包含 datetime 列或已经是 DatetimeIndex。")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

print("Loaded shape:", df.shape)
print("Columns:", list(df.columns))

# ========= 2) 统一列名（优先选择带单位的原始列；若没有则用简名） =========
def pick_col(df, candidates):
    """从候选列名中返回第一个存在的列名；若都不存在，返回 None"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

col_ch4_val = pick_col(df, ["CH4 (ppm)", "CH4"])
col_co2_val = pick_col(df, ["CO2 (ppm)", "CO2"])
col_co_val  = pick_col(df, ["CO (ppb)", "CO"])
col_h2o_val = pick_col(df, ["H2O (ppm)", "H2O"])

col_ch4_qc = pick_col(df, ["CH4_qc_flags", "ch4_qc_flags"])
col_co2_qc = pick_col(df, ["CO2_qc_flags", "co2_qc_flags"])
col_co_qc  = pick_col(df, ["CO_qc_flags", "co_qc_flags"])
col_h2o_qc = pick_col(df, ["H2O_qc_flags", "h2o_qc_flags"])

# 取出需要的列（有则取），并复制一份干净表
keep_cols = [c for c in [col_ch4_val, col_co2_val, col_co_val, col_h2o_val,
                         col_ch4_qc, col_co2_qc, col_co_qc, col_h2o_qc] if c is not None]
df_use = df[keep_cols].copy()

# ========= 3) 按 QC=1 过滤有效值 =========
def apply_qc(df_in, val_col, qc_col):
    if (val_col is not None) and (qc_col is not None) and (val_col in df_in.columns) and (qc_col in df_in.columns):
        df_in.loc[df_in[qc_col] != 1, val_col] = np.nan

apply_qc(df_use, col_ch4_val, col_ch4_qc)
apply_qc(df_use, col_co2_val, col_co2_qc)
apply_qc(df_use, col_co_val,  col_co_qc)
apply_qc(df_use, col_h2o_val, col_h2o_qc)

# ========= 4) 重命名为简短统一列名 =========
rename_map = {}
if col_ch4_val: rename_map[col_ch4_val] = "CH4"
if col_co2_val: rename_map[col_co2_val] = "CO2"
if col_co_val:  rename_map[col_co_val]  = "CO"
if col_h2o_val: rename_map[col_h2o_val] = "H2O"
df_use = df_use.rename(columns=rename_map)

# 仅保留观测值列（CH4, CO2, CO, H2O）
obs_cols = [c for c in ["CH4","CO2","CO","H2O"] if c in df_use.columns]
df_obs = df_use[obs_cols]

# ========= 5) 计算日均、月均 =========
daily   = df_obs.resample("D").mean()
monthly = df_obs.resample("M").mean()

# ========= 6) 作图并保存 & 显示（只三张图） =========
def save_and_show(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved figure:", path)
    plt.show()
    plt.close()

# 6.1 CH4 日均
if "CH4" in daily.columns:
    plt.figure(figsize=(12, 4))
    daily["CH4"].plot()
    plt.title("Daily CH₄ Concentration (QC-filtered)")
    plt.xlabel("Date")
    plt.ylabel("CH₄ (ppm)")
    save_and_show(os.path.join(OUTPUT_DIR, "daily_CH4.png"))
else:
    print("警告：未找到 CH4 列，跳过 CH4 日均图。")

# 6.2 CO2 日均
if "CO2" in daily.columns:
    plt.figure(figsize=(12, 4))
    daily["CO2"].plot()
    plt.title("Daily CO₂ Concentration (QC-filtered)")
    plt.xlabel("Date")
    plt.ylabel("CO₂ (ppm)")
    save_and_show(os.path.join(OUTPUT_DIR, "daily_CO2.png"))
else:
    print("警告：未找到 CO2 列，跳过 CO2 日均图。")

# 6.3 月均双轴（CO2 左轴，CH4 右轴）
if ("CO2" in monthly.columns) and ("CH4" in monthly.columns):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    l1 = ax1.plot(monthly.index, monthly["CO2"], linestyle="-")[0]
    ax1.set_ylabel("CO₂ (ppm)")

    ax2 = ax1.twinx()
    l2 = ax2.plot(monthly.index, monthly["CH4"], linestyle="--")[0]
    ax2.set_ylabel("CH₄ (ppm)")

    ax1.set_title("Monthly Mean CO₂ (left) and CH₄ (right) (QC-filtered)")
    ax1.set_xlabel("Month")
    ax1.legend([l1, l2], ["CO₂ (solid)", "CH₄ (dashed)"], loc="best")
    save_and_show(os.path.join(OUTPUT_DIR, "monthly_CO2_left_CH4_right.png"))
else:
    print("提示：月均双轴图需要同时存在 CO2 与 CH4 列。")

# ========= 7) 同目录下导出 CSV =========
out_csv = os.path.join(OUTPUT_DIR, "all_years_cleaned_from_parquet.csv")
df_obs.to_csv(out_csv)
print("Also saved a CSV copy to:", out_csv)
