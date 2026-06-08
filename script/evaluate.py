#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if mask.sum() == 0:
        return dict(R2=0, PCC=0, RMSE=0, MAE=0, p1mag=0)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    R2 = r2_score(y_true, y_pred)

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        PCC = 0
    else:
        PCC = np.corrcoef(y_true, y_pred)[0, 1]

    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    MAE = mean_absolute_error(y_true, y_pred)
    p1mag = np.mean(np.abs(y_true - y_pred) < 1.0)

    return dict(
        R2=R2,
        PCC=PCC,
        RMSE=RMSE,
        MAE=MAE,
        p1mag=p1mag,
    )


def simple_evaluate(input_csv, output_csv, dataset_name):
    df = pd.read_csv(input_csv)

    y_true_col = "y_true_log"

    model_config = {
        "dlkcat": "y_pred_log_dlkcat",
        "unikp": "y_pred_log_unikp",
        "catapro": "y_pred_log_catapro",
        "saekp": "y_pred_log_saekp",
    }

    metrics_list = ["R2", "PCC", "RMSE", "MAE", "p1mag"]

    rows = []

    for metric_name in metrics_list:
        row = {
            "datasets": dataset_name,
            "metrics": metric_name,
        }

        for model_name, pred_col in model_config.items():
            if pred_col not in df.columns:
                row[model_name] = np.nan
                continue

            result = compute_metrics(
                df[y_true_col].values,
                df[pred_col].values,
            )

            row[model_name] = result[metric_name]

        rows.append(row)

    df_out = pd.DataFrame(rows)

    df_out = df_out[
        [
            "datasets",
            "metrics",
            "dlkcat",
            "unikp",
            "catapro",
            "saekp",
        ]
    ]

    for col in ["dlkcat", "unikp", "catapro", "saekp"]:
        df_out[col] = df_out[col].astype(float).round(3)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    print(df_out)
    print(f"\n✅ saved: {output_csv}")


if __name__ == "__main__":

    simple_evaluate(
        input_csv="/path/to/tn.csv",
        output_csv="/path/to/tn_metrics_summary.csv",
        dataset_name="tn",
    )
    simple_evaluate(
        input_csv="/path/to/km.csv",
        output_csv="/path/to/km_metrics_summary.csv",
        dataset_name="km",
    )
    simple_evaluate(
        input_csv="/path/to/ki.csv",
        output_csv="/path/to/ki_metrics_summary.csv",
        dataset_name="ki",
    )