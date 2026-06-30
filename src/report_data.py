"""Coleta métricas e artefatos para os relatórios PDF/DOCX."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, SLA_THRESHOLD_FPS, TRAIN_SIZES
from .data import feature_columns, load_trace, target_column
from .gnettrack import build_modeling_frame, load_gnettrack
from .labels import rename_stat_table
from .metrics import classification_error, nmae
from .splits import sample_training_subset, train_test_split_sla, train_test_split_trace


@dataclass
class ReportData:
    n_obs: int
    n_train: int
    n_test: int
    stats: pd.DataFrame
    q2a: int
    q2b: float
    q2c: float
    features: list[str]
    target: str
    model_m_coef: np.ndarray
    model_m_intercept: float
    nmae_m: float
    task_ii_results: pd.DataFrame
    task_ii_coef: pd.DataFrame
    task_iii_results: pd.DataFrame
    task_iii_coef: pd.DataFrame
    err_best: float
    best_classifier: str
    y_test_mean: float
    bonus_mae: float
    bonus_rmse: float
    bonus_r2: float
    bonus_n: int
    bonus_coef: dict[str, float]


def collect_report_data() -> ReportData:
    df = load_trace()
    features = feature_columns()
    target = target_column()
    cols = features + [target]

    stats = df[cols].agg(
        ["mean", "max", "min", lambda s: s.quantile(0.25), lambda s: s.quantile(0.90), "std"]
    )
    stats.index = ["mean", "max", "min", "p25", "p90", "std"]
    stats_pt = rename_stat_table(stats).round(4)

    q2a = int((df["X..memused"] > 80).sum())
    q2b = float(df.loc[df["sum_intr.s"] > 18000, "tcpsck"].mean())
    q2c = float(df.loc[df["all_..idle"] < 20, "X..memused"].min())

    train_df, test_df, x_train, y_train, x_test, y_test = train_test_split_trace(
        df, features, target
    )
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    nmae_m = nmae(y_test, reg.predict(x_test))

    task_ii_rows = []
    for n in TRAIN_SIZES:
        subset = sample_training_subset(train_df, n, random_state=RANDOM_STATE)
        r = LinearRegression()
        t0 = time.perf_counter()
        r.fit(subset[features].values, subset[target].values)
        ms = (time.perf_counter() - t0) * 1000
        err = nmae(y_test, r.predict(x_test))
        task_ii_rows.append(
            {"n_train": n, "train_time_ms": ms, "nmae": err, "nmae_pct": err * 100, "accurate": err < 0.15}
        )
    task_ii_results = pd.DataFrame(task_ii_rows)

    coef_rows = []
    for n in TRAIN_SIZES:
        subset = sample_training_subset(train_df, n, random_state=RANDOM_STATE)
        r = LinearRegression().fit(subset[features].values, subset[target].values)
        row: dict[str, Any] = {"n_train": n}
        for f, c in zip(features, r.coef_):
            row[f] = c
        row["intercept"] = r.intercept_
        coef_rows.append(row)
    task_ii_coef = pd.DataFrame(coef_rows)

    train_sla, test_sla, _, y_train_sla, x_test_sla, y_test_sla, _, _ = train_test_split_sla(
        df, features, target
    )
    t3_rows = []
    t3_coef_rows = []
    for i, n in enumerate(TRAIN_SIZES):
        subset = sample_training_subset(train_sla, n, random_state=RANDOM_STATE)
        y_sub = (subset[target].values >= SLA_THRESHOLD_FPS).astype(int)
        clf = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)
        t0 = time.perf_counter()
        clf.fit(subset[features].values, y_sub)
        ms = (time.perf_counter() - t0) * 1000
        y_pred = clf.predict(x_test_sla)
        err = classification_error(y_test_sla, y_pred)
        t3_rows.append(
            {
                "classifier": f"C{i + 1}",
                "n_train": n,
                "train_time_ms": ms,
                "err": err,
                "err_pct": err * 100,
                "accurate": err < 0.15,
            }
        )
        row = {"classifier": f"C{i + 1}", "n_train": n, "intercept": clf.intercept_[0]}
        for f, c in zip(features, clf.coef_[0]):
            row[f] = c
        t3_coef_rows.append(row)
    task_iii_results = pd.DataFrame(t3_rows)
    task_iii_coef = pd.DataFrame(t3_coef_rows)
    best_idx = task_iii_results["err"].idxmin()
    best_classifier = task_iii_results.loc[best_idx, "classifier"]
    err_best = float(task_iii_results.loc[best_idx, "err"])

    bonus = build_modeling_frame(load_gnettrack())
    bonus_features = ["Level", "SNR", "DL_bitrate", "UL_bitrate", "Speed"]
    bx_tr, bx_te, by_tr, by_te = train_test_split(
        bonus[bonus_features], bonus["Qual"], test_size=0.3, random_state=RANDOM_STATE
    )
    breg = LinearRegression().fit(bx_tr, by_tr)
    bpred = breg.predict(bx_te)

    return ReportData(
        n_obs=len(df),
        n_train=len(train_df),
        n_test=len(test_df),
        stats=stats_pt,
        q2a=q2a,
        q2b=q2b,
        q2c=q2c,
        features=features,
        target=target,
        model_m_coef=reg.coef_,
        model_m_intercept=float(reg.intercept_),
        nmae_m=float(nmae_m),
        task_ii_results=task_ii_results,
        task_ii_coef=task_ii_coef,
        task_iii_results=task_iii_results,
        task_iii_coef=task_iii_coef,
        err_best=err_best,
        best_classifier=best_classifier,
        y_test_mean=float(y_test.mean()),
        bonus_mae=float(mean_absolute_error(by_te, bpred)),
        bonus_rmse=float(mean_squared_error(by_te, bpred) ** 0.5),
        bonus_r2=float(r2_score(by_te, bpred)),
        bonus_n=len(bonus),
        bonus_coef=dict(zip(bonus_features, breg.coef_)),
    )
