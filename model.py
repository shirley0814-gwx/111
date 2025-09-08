# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 14:54:42 2025

@author: 高文萱
"""


# ======= CH4 End-to-End Pipeline =======
# Hourly aggregation -> Feature engineering -> XGBoost (fallback Ridge)
# -> Residuals & anomalies -> Diagnostics -> Event aggregation
# -> H2O analysis -> Random Forest comparison

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 路径（相对路径；脚本同目录） ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "all_years_cleaned.parquet")
OUT_DIR = os.path.join(BASE_DIR, "model_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 读入 & 小时聚合 ----------
df = pd.read_parquet(PARQUET_PATH)

if not isinstance(df.index, pd.DatetimeIndex):
    time_col = None
    for c in df.columns:
        if re.search(r'time|date|datetime', c, flags=re.IGNORECASE):
            time_col = c; break
    if time_col is None:
        raise ValueError("未找到时间列，请检查 parquet。")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

# 统一列名
rename = {}
if "CH4 (ppm)" in df.columns: rename["CH4 (ppm)"] = "CH4"
if "CO2 (ppm)" in df.columns: rename["CO2 (ppm)"] = "CO2"
if "CO (ppb)"  in df.columns: rename["CO (ppb)"]  = "CO"
if "H2O (ppm)" in df.columns: rename["H2O (ppm)"] = "H2O"
df = df.rename(columns=rename)
df = df.loc[:, ~df.columns.duplicated()]

# 仅保留可用列再小时聚合
keep_cols = [c for c in ["CH4","CO2","CO","H2O"] if c in df.columns]
hourly = df[keep_cols].resample("H").mean()

# ---------- 特征工程（避免泄漏：先 shift(1) 再 rolling） ----------
feat = hourly.copy()
feat["hour"]  = feat.index.hour
feat["dow"]   = feat.index.dayofweek
feat["month"] = feat.index.month
feat["hour_sin"]  = np.sin(2*np.pi*feat["hour"]/24)
feat["hour_cos"]  = np.cos(2*np.pi*feat["hour"]/24)
feat["month_sin"] = np.sin(2*np.pi*feat["month"]/12)
feat["month_cos"] = np.cos(2*np.pi*feat["month"]/12)

for col in ["CH4","CO2","CO","H2O"]:
    if col in feat.columns:
        past = feat[col].shift(1)                    # 只用过去信息
        feat[f"{col}_lag1"]  = past
        feat[f"{col}_lag2"]  = feat[col].shift(2)
        feat[f"{col}_lag24"] = feat[col].shift(24)
        feat[f"{col}_roll3_mean"]  = past.rolling(3).mean()
        feat[f"{col}_roll24_mean"] = past.rolling(24).mean()
        feat[f"{col}_roll24_std"]  = past.rolling(24).std()

feat = feat.dropna()

# ---------- Train / Test 划分 ----------
train = feat[feat.index.year <= 2022]
test  = feat[feat.index.year == 2023]
X_train, y_train = train.drop(columns=["CH4"]), train["CH4"]
X_test,  y_test  = test.drop(columns=["CH4"]),  test["CH4"]

# ---------- 基线模型：优先 XGBoost，若无则回退 Ridge ----------
model_name = None
try:
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        tree_method="hist", n_jobs=4
    )
    model_name = "XGBoost"
except Exception:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0, random_state=42))
    model_name = "Ridge (fallback; install xgboost to use XGBoost)"

print(f"[Model] Using: {model_name}")
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_test), index=y_test.index)

# ---------- 指标 & 残差 & 异常（μ+3σ） ----------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
resid = (y_test - y_pred).copy()
thr = resid.mean() + 3*resid.std()
anoms = resid[resid > thr]

perf = pd.DataFrame({
    "model":[model_name],
    "n_train_hours":[len(X_train)],
    "n_test_hours":[len(X_test)],
    "RMSE_test":[rmse],
    "MAE_test":[mae],
    "R2_test":[r2],
    "resid_mean":[resid.mean()],
    "resid_std":[resid.std()],
    "anomaly_threshold_mu_plus_3sigma":[thr],
    "anomaly_count":[len(anoms)],
    "anomaly_ratio_percent":[100*len(anoms)/len(resid)]
})
perf_path = os.path.join(OUT_DIR, "regression_performance_summary.csv")
perf.to_csv(perf_path, index=False)
print("[Saved]", perf_path)

# ---------- 残差诊断（可选：需要 scipy / statsmodels） ----------
warnings.filterwarnings("ignore")
try:
    from scipy.stats import skew, kurtosis, jarque_bera
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import durbin_watson
    jb_stat, jb_p = jarque_bera(resid)
    lb = acorr_ljungbox(resid, lags=[24], return_df=True)
    lbq, lbp = float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])
    dw = durbin_watson(resid)
    resid_stats = pd.DataFrame({
        "n":[len(resid)],
        "mean":[resid.mean()],
        "std":[resid.std()],
        "median":[resid.median()],
        "mad":[(resid - resid.median()).abs().median()],
        "skewness":[skew(resid, bias=False)],
        "kurtosis_excess":[kurtosis(resid, fisher=True, bias=False)],
        "jarque_bera_stat":[jb_stat],
        "jarque_bera_pvalue":[jb_p],
        "ljung_box_Q(24)":[lbq],
        "ljung_box_pvalue(24)":[lbp],
        "durbin_watson":[dw],
    })
except Exception as e:
    print("[WARN] 残差统计包缺失（scipy/statsmodels），仅输出基础统计：", e)
    resid_stats = pd.DataFrame({
        "n":[len(resid)],
        "mean":[resid.mean()],
        "std":[resid.std()],
        "median":[resid.median()],
        "mad":[(resid - resid.median()).abs().median()],
    })
resid_path = os.path.join(OUT_DIR, "residual_diagnostics.csv")
resid_stats.to_csv(resid_path, index=False)
print("[Saved]", resid_path)

# ---------- 异常清单 ----------
anom_df = pd.DataFrame({
    "observed_CH4_ppm": y_test,
    "predicted_CH4_ppm": y_pred,
    "residual_ppm": resid
}).loc[resid > thr].sort_values("residual_ppm", ascending=False)
anom_csv = os.path.join(OUT_DIR, "ch4_residual_anomalies_2023.csv")
anom_df.to_csv(anom_csv, index_label="datetime")
print("[Saved]", anom_csv)

# ---------- 三张图（保存并显示） ----------
# (a) 观测 vs 预测（抽样一周）
start, end = 24*100, 24*107
week_idx = y_test.index[start:end]
plt.figure(figsize=(12,4))
plt.plot(y_test.loc[week_idx], label="Observed CH4")
plt.plot(y_pred.loc[week_idx], label=f"{model_name} Prediction")
plt.title(f"Observed vs Predicted CH₄ (sample week, 2023) — {model_name}")
plt.xlabel("Time"); plt.ylabel("CH₄ (ppm)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "obs_vs_pred_week.png"), dpi=200, bbox_inches="tight")
plt.show()

# (b) 残差直方图 + 阈值
plt.figure(figsize=(7,4))
plt.hist(resid, bins=60)
plt.axvline(thr, linestyle="--", label="3σ threshold")
plt.title("Residual Distribution (CH₄, 2023)")
plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residual_hist.png"), dpi=200, bbox_inches="tight")
plt.show()

# (c) 残差时间序列 + 阈值
plt.figure(figsize=(12,4))
plt.plot(resid, label="Residuals")
plt.axhline(thr, linestyle="--", label="3σ threshold")
plt.title("Residuals over time (CH₄, 2023)")
plt.xlabel("Time"); plt.ylabel("Residual"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residual_timeseries.png"), dpi=200, bbox_inches="tight")
plt.show()

# ---------- 异常 → 事件聚合 ----------
def aggregate_events(residual_series, threshold, y_true, h2o_series=None):
    anom = residual_series[residual_series > threshold]
    events = []
    if anom.empty:
        return pd.DataFrame(events)
    current_start = anom.index[0]
    current_end = current_start
    vals = [anom.iloc[0]]
    obs  = [y_true.loc[current_start]]
    for t, v in zip(anom.index[1:], anom.values[1:]):
        if (t - current_end).total_seconds() <= 3600:
            current_end = t; vals.append(v); obs.append(y_true.loc[t])
        else:
            events.append({
                "start_time": current_start,
                "end_time": current_end,
                "duration_hours": (current_end - current_start).total_seconds()/3600 + 1,
                "max_residual": max(vals),
                "peak_CH4_observed": max(obs),
                "mean_H2O": float(h2o_series.loc[current_start:current_end].mean()) if h2o_series is not None else np.nan
            })
            current_start = t; current_end = t; vals = [v]; obs = [y_true.loc[t]]
    events.append({
        "start_time": current_start,
        "end_time": current_end,
        "duration_hours": (current_end - current_start).total_seconds()/3600 + 1,
        "max_residual": max(vals),
        "peak_CH4_observed": max(obs),
        "mean_H2O": float(h2o_series.loc[current_start:current_end].mean()) if h2o_series is not None else np.nan
    })
    return pd.DataFrame(events)

h2o_series = hourly["H2O"] if "H2O" in hourly.columns else None
event_df = aggregate_events(resid, thr, y_test, h2o_series)
event_csv = os.path.join(OUT_DIR, "ch4_anomaly_events_2023.csv")
event_df.to_csv(event_csv, index=False)
print("[Saved]", event_csv)

# 事件持续时长直方图
if not event_df.empty:
    plt.figure(figsize=(8,4))
    bins = range(1, int(event_df["duration_hours"].max())+2)
    plt.hist(event_df["duration_hours"], bins=bins, edgecolor="black")
    plt.title("Distribution of CH₄ Anomaly Event Durations (2023)")
    plt.xlabel("Duration (hours)"); plt.ylabel("Number of events")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "event_duration_hist.png"), dpi=200, bbox_inches="tight")
    plt.show()

# ---------- H2O 分析：异常 vs 正常 ----------
if "H2O" in hourly.columns:
    anom_idx = resid[resid > thr].index
    normal_idx = resid[resid <= thr].index
    h2o_anom = hourly.loc[anom_idx, "H2O"].dropna()
    h2o_norm = hourly.loc[normal_idx, "H2O"].dropna()
    h2o_summary = pd.DataFrame({
        "stat":["mean","median","std","n"],
        "H2O_anomalous":[h2o_anom.mean(), h2o_anom.median(), h2o_anom.std(), len(h2o_anom)],
        "H2O_normal":[h2o_norm.mean(), h2o_norm.median(), h2o_norm.std(), len(h2o_norm)],
    })
    h2o_path = os.path.join(OUT_DIR, "h2o_anom_vs_normal_summary.csv")
    h2o_summary.to_csv(h2o_path, index=False); print("[Saved]", h2o_path)

    # 箱线图
    plt.figure(figsize=(6,4))
    plt.boxplot([h2o_norm, h2o_anom], labels=["Normal","Anomalous"], patch_artist=True)
    plt.ylabel("H₂O (ppm)")
    plt.title("H₂O levels: normal vs anomalous hours (2023)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "h2o_boxplot.png"), dpi=200, bbox_inches="tight")
    plt.show()

    # 残差 vs H2O 散点
    plt.figure(figsize=(7,4))
    plt.scatter(hourly.loc[resid.index, "H2O"], resid, alpha=0.3)
    plt.axhline(thr, linestyle="--", label="3σ threshold")
    plt.xlabel("H₂O (ppm)"); plt.ylabel("Residual CH₄ (ppm)")
    plt.title("CH₄ residuals vs H₂O (2023)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "residual_vs_h2o_scatter.png"), dpi=200, bbox_inches="tight")
    plt.show()

# ---------- 模型对比：Random Forest ----------
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rf = RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = pd.Series(rf.predict(X_test), index=y_test.index)

rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
r2_rf   = r2_score(y_test, y_pred_rf)

compare = pd.DataFrame({
    "Model":[model_name, "Random Forest"],
    "RMSE":[rmse, rmse_rf],
    "MAE":[mae,  mae_rf],
    "R2":[r2,  r2_rf]
})
cmp_path = os.path.join(OUT_DIR, "model_comparison_rf_vs_baseline.csv")
compare.to_csv(cmp_path, index=False); print("[Saved]", cmp_path)

# 一周对比图（观测 vs 基线 vs RF）
plt.figure(figsize=(12,4))
plt.plot(y_test.loc[week_idx], label="Observed CH₄")
plt.plot(y_pred.loc[week_idx], label=f"{model_name} Prediction")
plt.plot(y_pred_rf.loc[week_idx], label="Random Forest Prediction")
plt.title("Observed vs Predicted CH₄ (sample week, 2023)")
plt.xlabel("Time"); plt.ylabel("CH₄ (ppm)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "obs_vs_pred_week_rf_compare.png"), dpi=200, bbox_inches="tight")
plt.show()

print("\n=== DONE ===")
print("Outputs saved to:", OUT_DIR)

# ---------- 可解释性：XGBoost 优先，失败则用置换重要性 ----------
from sklearn.inspection import permutation_importance
try:
    import xgboost as xgb
    is_xgb = isinstance(model, xgb.XGBRegressor)
except Exception:
    is_xgb = False

if is_xgb:
    # 1) XGBoost 内置特征重要性
    importance = model.feature_importances_
    feat_imp = (
        pd.DataFrame({"feature": X_train.columns, "importance": importance})
        .sort_values("importance", ascending=False)
    )
    feat_imp.to_csv(os.path.join(OUT_DIR, "xgb_feature_importance.csv"), index=False)

    plt.figure(figsize=(8,5))
    feat_imp.head(15).plot(kind="barh", x="feature", y="importance", legend=False)
    plt.title("Top 15 Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "xgb_feature_importance.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2) SHAP（可能略慢，抽样加速）
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sample_n = min(1000, len(X_test))
        Xs = X_test.sample(sample_n, random_state=42)
        shap_values = explainer.shap_values(Xs)

        # SHAP summary bar（全局重要性）
        shap.summary_plot(shap_values, Xs, plot_type="bar", show=False)
        plt.title("SHAP summary (bar)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "shap_summary_bar.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # SHAP beeswarm（方向与非线性）
        shap.summary_plot(shap_values, Xs, show=False)
        plt.title("SHAP summary (beeswarm)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "shap_summary_beeswarm.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # 导出全局 SHAP 重要性表
        shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
        shap_imp = (
            pd.DataFrame({"feature": Xs.columns, "mean_abs_SHAP": shap_abs_mean})
            .sort_values("mean_abs_SHAP", ascending=False)
        )
        shap_imp.to_csv(os.path.join(OUT_DIR, "shap_global_importance.csv"), index=False)

        # 3) 依赖图：CH4_lag1（若存在）
        if "CH4_lag1" in Xs.columns:
            shap.dependence_plot("CH4_lag1", shap_values, Xs, interaction_index=None, show=False)
            plt.title("SHAP Dependence Plot for CH₄ lag-1")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "shap_dependence_CH4_lag1.png"), dpi=200, bbox_inches="tight")
            plt.close()

        print("[OK] XGBoost + SHAP 可解释性结果已保存到：", OUT_DIR)

    except Exception as e:
        print("[WARN] SHAP 计算失败，改用置换重要性:", e)
        pi = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        pi_imp = (
            pd.DataFrame({"feature": X_test.columns, "perm_importance": pi.importances_mean})
            .sort_values("perm_importance", ascending=False)
        )
        pi_imp.to_csv(os.path.join(OUT_DIR, "permutation_importance.csv"), index=False)
        plt.figure(figsize=(8,5))
        pi_imp.head(15).plot(kind="barh", x="feature", y="perm_importance", legend=False)
        plt.title("Top 15 Permutation Importances (XGBoost)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "permutation_importance.png"), dpi=200, bbox_inches="tight")
        plt.close()
else:
    # 非 XGBoost（比如 Ridge 回退） → 只能做置换重要性
    print("[INFO] 当前基线非 XGBoost，执行置换重要性。")
    pi = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    pi_imp = (
        pd.DataFrame({"feature": X_test.columns, "perm_importance": pi.importances_mean})
        .sort_values("perm_importance", ascending=False)
    )
    pi_imp.to_csv(os.path.join(OUT_DIR, "permutation_importance.csv"), index=False)
    plt.figure(figsize=(8,5))
    pi_imp.head(15).plot(kind="barh", x="feature", y="perm_importance", legend=False)
    plt.title("Top 15 Permutation Importances (non-XGB baseline)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "permutation_importance.png"), dpi=200, bbox_inches="tight")
    plt.close()

# ---------- （保留原脚本的额外 SHAP 小段；路径同 OUT_DIR） ----------
try:
    import shap
    explainer = shap.TreeExplainer(model)
    X_s = X_test.sample(min(2000, len(X_test)), random_state=42)  # 抽样加速
    shap_values = explainer.shap_values(X_s)

    shap.dependence_plot("CH4_lag1", shap_values, X_s, interaction_index=None, show=False)
    plt.title("SHAP Dependence Plot for CH₄ lag-1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_dependence_CH4_lag1.png"), dpi=200, bbox_inches="tight")
    plt.show()
except Exception as e:
    print("[WARN] 末尾 SHAP 依赖图失败：", e)
