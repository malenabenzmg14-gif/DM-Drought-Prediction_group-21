import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SAMPLE_SUB_PATH = "data/sample_submission.csv"
OUTPUT_PATH = "outputs/submission_final.csv"

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range"
]

SCORE_LAGS = [1, 2, 3, 4, 8, 12]
ROLL_WINDOWS = [2, 4, 8, 13]


def extract_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["date"].astype(str).str.split("-", expand=True)
    df["year"] = parts[0].astype(np.int32)
    df["month"] = parts[1].astype(np.int16)
    df["day"] = parts[2].astype(np.int16)

    month_offsets = {
        1: 0, 2: 31, 3: 59, 4: 90, 5: 120, 6: 151,
        7: 181, 8: 212, 9: 243, 10: 273, 11: 304, 12: 334
    }
    df["dayofyear"] = df["month"].map(month_offsets).astype(np.int16) + df["day"].astype(np.int16)
    return df


def add_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_date_parts(df)
    df = df.sort_values(["region_id", "year", "month", "day"]).reset_index(drop=True)
    df["day_number"] = df.groupby("region_id").cumcount().astype(np.int32)
    df["week_idx"] = (df["day_number"] // 7).astype(np.int32)
    return df


def build_weekly_weather(df: pd.DataFrame, has_score: bool) -> pd.DataFrame:
    agg = {
        "month": "last",
        "dayofyear": "last",
    }

    for col in WEATHER_COLS:
        agg[col] = ["mean", "std", "min", "max"]

    weekly = df.groupby(["region_id", "week_idx"]).agg(agg)
    weekly.columns = [
        f"{a}_{b}" if b else a
        for a, b in weekly.columns.to_flat_index()
    ]
    weekly = weekly.reset_index()

    weekly = weekly.rename(
        columns={
            "month_last": "month",
            "dayofyear_last": "dayofyear"
        }
    )

    if "prec_sum" not in weekly.columns:
        weekly["prec_sum"] = df.groupby(["region_id", "week_idx"])["prec"].sum().values

    if has_score:
        score_weekly = (
            df.dropna(subset=["score"])
            .groupby(["region_id", "week_idx"])["score"]
            .mean()
            .reset_index()
        )
        weekly = weekly.merge(score_weekly, on=["region_id", "week_idx"], how="left")
    else:
        weekly["score"] = np.nan

    weekly["month_sin"] = np.sin(2 * np.pi * weekly["month"] / 12)
    weekly["month_cos"] = np.cos(2 * np.pi * weekly["month"] / 12)
    weekly["dayofyear_sin"] = np.sin(2 * np.pi * weekly["dayofyear"] / 365)
    weekly["dayofyear_cos"] = np.cos(2 * np.pi * weekly["dayofyear"] / 365)

    return weekly


def add_weekly_weather_rollups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["region_id", "week_idx"]).copy()

    base_cols = [
        "prec_sum",
        "prec_mean",
        "humidity_mean",
        "tmp_mean",
        "tmp_max_mean",
        "tmp_min_mean",
        "wind_mean",
        "surf_pre_mean",
    ]

    base_cols = [c for c in base_cols if c in df.columns]

    for col in base_cols:
        for window in ROLL_WINDOWS:
            df[f"{col}_roll{window}_mean"] = (
                df.groupby("region_id")[col]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
            )
            df[f"{col}_roll{window}_std"] = (
                df.groupby("region_id")[col]
                .transform(lambda s: s.rolling(window, min_periods=2).std())
                .fillna(0)
            )

    if "prec_sum_roll4_mean" in df.columns and "prec_sum_roll13_mean" in df.columns:
        df["prec_4w_vs_13w"] = df["prec_sum_roll4_mean"] - df["prec_sum_roll13_mean"]

    if "humidity_mean_roll4_mean" in df.columns and "humidity_mean_roll13_mean" in df.columns:
        df["humidity_4w_vs_13w"] = df["humidity_mean_roll4_mean"] - df["humidity_mean_roll13_mean"]

    if "tmp_mean_roll4_mean" in df.columns and "tmp_mean_roll13_mean" in df.columns:
        df["tmp_4w_vs_13w"] = df["tmp_mean_roll4_mean"] - df["tmp_mean_roll13_mean"]

    return df


def add_score_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["region_id", "week_idx"]).copy()

    for lag in SCORE_LAGS:
        df[f"score_lag_{lag}"] = df.groupby("region_id")["score"].shift(lag)

    for window in [4, 8, 12]:
        df[f"score_roll{window}_mean"] = (
            df.groupby("region_id")["score"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"score_roll{window}_std"] = (
            df.groupby("region_id")["score"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
            .fillna(0)
        )

    return df


def build_region_features(train_weekly: pd.DataFrame):
    train_scored = train_weekly.dropna(subset=["score"]).copy()

    region_mean = (
        train_scored.groupby("region_id")["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "region_score_mean"})
    )

    region_month = (
        train_scored.groupby(["region_id", "month"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "future_region_month_score_mean"})
    )

    return region_mean, region_month


def get_feature_columns(df: pd.DataFrame):
    exclude = {
        "score",
        "target",
        "region_id",
        "date",
    }

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def predict_test_week_scores(train_weekly, test_weekly):
    print("Predicting missing scores for the 13 observed test weeks sequentially...")

    train_model_data = train_weekly.dropna(subset=["score"]).copy()
    feature_cols = get_feature_columns(train_model_data)

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=0.4,
        random_state=42,
        verbose=-1,
    )

    X = train_model_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train_model_data["score"]
    model.fit(X, y)

    test_original = test_weekly.copy()
    test_shifted = test_weekly.copy()

    last_train_week = train_weekly.groupby("region_id")["week_idx"].max().reset_index()
    last_train_week = last_train_week.rename(columns={"week_idx": "last_train_week"})

    test_shifted = test_shifted.merge(last_train_week, on="region_id", how="left")
    test_shifted["orig_week_idx"] = test_shifted["week_idx"]
    test_shifted["week_idx"] = test_shifted["last_train_week"] + 1 + test_shifted["week_idx"]
    test_shifted = test_shifted.drop(columns=["last_train_week"])

    combined = pd.concat([train_weekly, test_shifted], ignore_index=True, sort=False)

    test_abs_weeks = sorted(test_shifted["week_idx"].unique())

    for week in test_abs_weeks:
        combined = add_score_lags(combined)
        rows = combined["week_idx"].eq(week) & combined["score"].isna()

        X_test = combined.loc[rows, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, 5)

        combined.loc[rows, "score"] = preds

    predicted = combined[combined["orig_week_idx"].notna()].copy()
    predicted["week_idx"] = predicted["orig_week_idx"].astype(np.int32)
    predicted = predicted.drop(columns=["orig_week_idx"], errors="ignore")

    test_original = test_original.drop(columns=["score"], errors="ignore")
    predicted_scores = predicted[["region_id", "week_idx", "score"]]

    out = test_original.merge(predicted_scores, on=["region_id", "week_idx"], how="left")

    print("Predicted test-week score summary:")
    print(out["score"].describe())

    return out


def build_horizon_train_table(train_weekly, region_mean, region_month, train_window=450):
    train_weekly = train_weekly.sort_values(["region_id", "week_idx"]).copy()

    # Keep the most recent history per region. This was the best public-LB window.
    train_recent = (
        train_weekly.groupby("region_id", group_keys=False)
        .tail(train_window)
        .copy()
    )

    frames = []

    for horizon in range(1, 6):
        print(f"  creating horizon {horizon} training rows...")

        temp = train_recent.copy()
        temp["horizon"] = horizon
        temp["target"] = temp.groupby("region_id")["score"].shift(-horizon)
        temp["future_month"] = temp.groupby("region_id")["month"].shift(-horizon)
        temp["future_dayofyear"] = temp.groupby("region_id")["dayofyear"].shift(-horizon)

        temp = temp.dropna(subset=["target", "score"])

        temp["future_month_sin"] = np.sin(2 * np.pi * temp["future_month"] / 12)
        temp["future_month_cos"] = np.cos(2 * np.pi * temp["future_month"] / 12)
        temp["future_dayofyear_sin"] = np.sin(2 * np.pi * temp["future_dayofyear"] / 365)
        temp["future_dayofyear_cos"] = np.cos(2 * np.pi * temp["future_dayofyear"] / 365)

        temp = temp.merge(region_mean, on="region_id", how="left")
        temp = temp.merge(
            region_month,
            left_on=["region_id", "future_month"],
            right_on=["region_id", "month"],
            how="left",
            suffixes=("", "_rm")
        )
        temp = temp.drop(columns=["month_rm"], errors="ignore")

        frames.append(temp)

    out = pd.concat(frames, ignore_index=True, sort=False)

    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = out[col].astype(np.float32)

    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = out[col].astype(np.int32)

    return out


def build_test_horizon_table(train_weekly, test_weekly, region_mean, region_month):
    last_train_week = train_weekly.groupby("region_id")["week_idx"].max().reset_index()
    last_train_week = last_train_week.rename(columns={"week_idx": "last_train_week"})

    test_shifted = test_weekly.copy()
    test_shifted = test_shifted.merge(last_train_week, on="region_id", how="left")
    test_shifted["week_idx"] = test_shifted["last_train_week"] + 1 + test_shifted["week_idx"]
    test_shifted = test_shifted.drop(columns=["last_train_week"])

    train_plus_test = pd.concat([train_weekly, test_shifted], ignore_index=True, sort=False)
    train_plus_test = add_score_lags(train_plus_test)

    latest = (
        train_plus_test.sort_values(["region_id", "week_idx"])
        .groupby("region_id", as_index=False)
        .tail(1)
        .copy()
    )

    frames = []

    for horizon in range(1, 6):
        temp = latest.copy()
        temp["horizon"] = horizon

        temp["future_dayofyear"] = ((temp["dayofyear"] + 7 * horizon - 1) % 365) + 1
        temp["future_month"] = temp["month"]

        temp["future_month_sin"] = np.sin(2 * np.pi * temp["future_month"] / 12)
        temp["future_month_cos"] = np.cos(2 * np.pi * temp["future_month"] / 12)
        temp["future_dayofyear_sin"] = np.sin(2 * np.pi * temp["future_dayofyear"] / 365)
        temp["future_dayofyear_cos"] = np.cos(2 * np.pi * temp["future_dayofyear"] / 365)

        temp = temp.merge(region_mean, on="region_id", how="left")
        temp = temp.merge(
            region_month,
            left_on=["region_id", "future_month"],
            right_on=["region_id", "month"],
            how="left",
            suffixes=("", "_rm")
        )
        temp = temp.drop(columns=["month_rm"], errors="ignore")

        frames.append(temp)

    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def train_predict(train_h, test_h, feature_cols):
    preds = []

    for horizon in range(1, 6):
        print(f"\nTraining horizon {horizon}...")

        tr = train_h[train_h["horizon"] == horizon].copy()
        te = test_h[test_h["horizon"] == horizon].copy()

        cutoff = tr["week_idx"].quantile(0.8)
        train_part = tr[tr["week_idx"] <= cutoff]
        valid_part = tr[tr["week_idx"] > cutoff]

        X_train = train_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_part["target"]

        X_valid = valid_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_valid = valid_part["target"]

        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=0.4,
            random_state=42,
            verbose=-1,
        )

        model.fit(X_train, y_train)

        valid_pred = np.clip(model.predict(X_valid), 0, 5)
        mae = mean_absolute_error(y_valid, valid_pred)
        print(f"Horizon {horizon} validation MAE: {mae:.5f}")

        X_full = tr[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_full = tr["target"]

        final_model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=0.4,
            random_state=42,
            verbose=-1,
        )

        final_model.fit(X_full, y_full)

        X_test = te[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        pred = np.clip(final_model.predict(X_test), 0, 5)

        temp = te[["region_id", "horizon", "future_region_month_score_mean", "region_score_mean"]].copy()
        temp["model_pred"] = pred
        preds.append(temp)

    return pd.concat(preds, ignore_index=True)


def build_submission(preds, sample_sub):
    preds = preds.copy()

    seasonal = preds["future_region_month_score_mean"]
    seasonal = seasonal.fillna(preds["region_score_mean"])
    seasonal = seasonal.fillna(0.85)

    preds["prediction"] = 0.80 * preds["model_pred"] + 0.20 * seasonal
    preds["prediction"] = preds["prediction"].clip(0, 5)

    wide = preds.pivot(index="region_id", columns="horizon", values="prediction").reset_index()
    wide.columns = ["region_id"] + [f"pred_week{i}" for i in range(1, 6)]

    submission = sample_sub[["region_id"]].merge(wide, on="region_id", how="left")

    for col in [f"pred_week{i}" for i in range(1, 6)]:
        submission[col] = submission[col].fillna(0.85).clip(0, 5)

    return submission


def main():
    os.makedirs("outputs", exist_ok=True)

    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    train = add_basic_columns(train)
    test = add_basic_columns(test)

    print("Building weekly tables...")
    train_weekly = build_weekly_weather(train, has_score=True)
    train_weekly = add_weekly_weather_rollups(train_weekly)
    train_weekly = add_score_lags(train_weekly)

    test_weekly = build_weekly_weather(test, has_score=False)
    test_weekly = add_weekly_weather_rollups(test_weekly)

    region_mean, region_month = build_region_features(train_weekly)

    test_weekly = predict_test_week_scores(train_weekly, test_weekly)

    print("Building horizon train/test tables...")
    train_h = build_horizon_train_table(
        train_weekly=train_weekly,
        region_mean=region_mean,
        region_month=region_month,
        train_window=450,
    )

    test_h = build_test_horizon_table(
        train_weekly=train_weekly,
        test_weekly=test_weekly,
        region_mean=region_mean,
        region_month=region_month,
    )

    # Align train and test feature columns.
    for col in train_h.columns:
        if col not in test_h.columns:
            test_h[col] = np.nan

    for col in test_h.columns:
        if col not in train_h.columns:
            train_h[col] = np.nan

    feature_cols = get_feature_columns(train_h)

    print(f"train_h shape: {train_h.shape}")
    print(f"test_h shape: {test_h.shape}")
    print(f"number of features: {len(feature_cols)}")

    preds = train_predict(train_h, test_h, feature_cols)

    submission = build_submission(preds, sample_sub)
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Submission shape: {submission.shape}")
    print(f"Null values: {submission.isna().sum().sum()}")
    print(submission.head())

    print("\nPrediction summary:")
    print(submission[[f"pred_week{i}" for i in range(1, 6)]].describe())


if __name__ == "__main__":
    main()