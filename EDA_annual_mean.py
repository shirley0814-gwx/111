# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 14:48:57 2025

@author: 高文萱
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= 路径（相对路径） =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                 # 当前脚本所在目录（EAN-11523777-am）
PARQUET_PATH = os.path.join(BASE_DIR, "all_years_cleaned.parquet")    # 同目录下的 parquet
OUTPUT_DIR = os.path.join(BASE_DIR, "model_outputs")                  # 输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= 1) 读取 & 时间索引 =========
df = pd.read_parquet(PARQUET_PATH)

# 若索引不是时间索引，尝试自动识别
if not isinstance(df.index, pd.DatetimeIndex):
    time_col = None
    for c in df.columns:
        cl = c.lower()
        if ("time" in cl) or ("date" in cl) or ("datetime" in cl):
            time_col = c
            break
    if time_col is None:
        raise ValueError("未找到时间列")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

# ========= 2) 统一列名（并处理重复列名）=========
# 去除完全重复的列（重复名保留第一个）
df = df.loc[:, ~df.columns.duplicated()]

def pick_col(cols, *candidates):
    """按候选名顺序返回第一个存在的列名；找不到返回 None"""
    for pat in candidates:
        if pat in cols:
            return pat
    return None

# 优先使用带单位的原名，没有就用简名
c_ch4 = pick_col(df.columns, "CH4 (ppm)", "CH4")
c_co2 = pick_col(df.columns, "CO2 (ppm)", "CO2")

use_cols = {}
if c_ch4: use_cols[c_ch4] = "CH4"
if c_co2: use_cols[c_co2] = "CO2"
if not use_cols:
    raise ValueError("在数据中未找到 CH4 或 CO2 列")

df2 = df[list(use_cols.keys())].rename(columns=use_cols)

# 若出现多重同名列导致 DataFrame，取第一列
for gas in ["CO2", "CH4"]:
    if gas in df2.columns and isinstance(df2[gas], pd.DataFrame):
        df2[gas] = df2[gas].iloc[:, 0]

# 只保留数值列
df2 = df2[[c for c in ["CO2", "CH4"] if c in df2.columns]]

# ========= 3) 计算 Jan–Sep 年均 =========
daily = df2.resample("D").mean()
subset = daily[daily.index.month <= 9]
annual = subset.groupby(subset.index.year).mean()

# 安全检查：确保是 Series（否则转换）
for gas in ["CO2", "CH4"]:
    if gas in annual.columns and isinstance(annual[gas], pd.DataFrame):
        annual[gas] = annual[gas].iloc[:, 0]

# ========= 工具函数：保存并显示 =========
def save_and_show(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved figure:", path)
    plt.show()
    plt.close()

# ========= 4) 图1：相对 2019（百分比），左轴从 80% =========
annual_rel = annual / annual.iloc[0] * 100

ax = annual_rel[["CO2", "CH4"]].plot(kind="bar", figsize=(9, 4))
ax.set_title("Annual Mean Relative to 2019 (Jan–Sep) — Y-axis from 80%")
ax.set_ylabel("% of 2019 value")
ax.set_ylim(80, max(105, float(annual_rel.max().max()) + 1))
ax.set_xticklabels([str(y) for y in annual_rel.index], rotation=0)

# 在柱顶标注百分比
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)

save_and_show(os.path.join(OUTPUT_DIR, "annual_relative_to_2019.png"))

# ========= 5) 图2：原始年均双轴柱（收紧 y 轴范围放大差异） =========
years = [str(y) for y in annual.index]
x = np.arange(len(years))
width = 0.38

fig, ax1 = plt.subplots(figsize=(9, 4))
b1 = ax1.bar(x - width/2, annual["CO2"].values, width, label="CO₂")
ax1.set_ylabel("CO₂ (ppm)")

# 收紧 CO2 y 轴范围
co2_min, co2_max = float(annual["CO2"].min()), float(annual["CO2"].max())
pad = max((co2_max - co2_min) * 0.15, 0.5)  # 至少给一点余量
ax1.set_ylim(co2_min - pad, co2_max + pad)

ax2 = ax1.twinx()
b2 = ax2.bar(x + width/2, annual["CH4"].values, width, hatch="//", label="CH₄")
ax2.set_ylabel("CH₄ (ppm)")

ch4_min, ch4_max = float(annual["CH4"].min()), float(annual["CH4"].max())
pad2 = max((ch4_max - ch4_min) * 0.15, 0.05)
ax2.set_ylim(ch4_min - pad2, ch4_max + pad2)

ax1.set_xticks(x)
ax1.set_xticklabels(years)
ax1.set_title("Jan–Sep Annual Mean: CO₂ (left) vs CH₄ (right) — Tight Y-limits")

# 柱顶标注具体值
for bar in b1:
    h = bar.get_height()
    ax1.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8)
for bar in b2:
    h = bar.get_height()
    ax2.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8)

ax1.legend([b1, b2], ["CO₂", "CH₄"], loc="best")
save_and_show(os.path.join(OUTPUT_DIR, "annual_dualaxis_tight.png"))
