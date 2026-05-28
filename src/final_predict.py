import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error


TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SAMPLE_SUB_PATH = "data/sample_submission.csv"
OUTPUT_PATH = "submission_final_1.csv"


WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range"
]


def extract_date_parts(date_series):
    s = date_series.astype(str)
    parts = s.str.extract(r"(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})")

    year = pd.to_numeric(parts["year"], errors="coerce").fillna(0).astype(int)
    month = pd.to_numeric(parts["month"], errors="coerce").fillna(1).astype(int)
    day = pd.to_numeric(parts["day"], errors="coerce").fillna(1).astype(int)

    pseudo_date = pd.to_datetime(
        "2001-" + month.astype(str).str.zfill(2) + "-" + day.astype(str).str.zfill(2),
        errors="coerce"
    )

    dayofyear = pseudo_date.dt.dayofyear.fillna(1).astype(int)

    return year, month, day, dayofyear


def add_basic_columns(df):
    df = df.copy()

    df["region_id"] = df["region_id"].astype(str)
    df["region_num"] = (
        df["region_id"]
        .str.extract(r"(\d+)")
        .astype(float)
        .fillna(-1)
        .astype(int)
    )

    year, month, day, dayofyear = extract_date_parts(df["date"])

    df["year_fake"] = year
    df["month"] = month
    df["day"] = day
    df["dayofyear"] = dayofyear

    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)
    df["day_idx"] = df.groupby("region_id").cumcount()
    df["week_idx"] = df["day_idx"] // 7

    return df


def build_weekly_weather(df):
    agg_dict = {}

    for col in WEATHER_COLS:
        agg_dict[col] = ["mean", "std", "min", "max"]

    weekly = df.groupby(["region_id", "week_idx"]).agg(agg_dict)
    weekly.columns = [f"{col}_{stat}" for col, stat in weekly.columns]
    weekly = weekly.reset_index()

    week_end = (
        df.sort_values(["region_id", "day_idx"])
        .groupby(["region_id", "week_idx"])
        .tail(1)[["region_id", "week_idx", "region_num", "month", "dayofyear", "year_fake"]]
    )

    weekly = weekly.merge(week_end, on=["region_id", "week_idx"], how="left")

    if "score" in df.columns:
        score_weekly = df[df["score"].notna()][["region_id", "week_idx", "score"]].copy()
        weekly = weekly.merge(score_weekly, on=["region_id", "week_idx"], how="left")

    if "prec" in df.columns:
        prec_week = (
            df.groupby(["region_id", "week_idx"])["prec"]
            .agg(prec_sum="sum")
            .reset_index()
        )
        weekly = weekly.merge(prec_week, on=["region_id", "week_idx"], how="left")

    return weekly.sort_values(["region_id", "week_idx"]).reset_index(drop=True)


def add_score_lags(weekly):
    weekly = weekly.copy()
    weekly = weekly.sort_values(["region_id", "week_idx"]).reset_index(drop=True)

    for lag in [1, 2, 3, 4, 8, 12]:
        weekly[f"score_lag_{lag}"] = weekly.groupby("region_id")["score"].shift(lag)

    for window in [4, 8, 12]:
        shifted = weekly.groupby("region_id")["score"].shift(1)

        weekly[f"score_roll_mean_{window}"] = (
            shifted.groupby(weekly["region_id"])
            .rolling(window, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )

        weekly[f"score_roll_std_{window}"] = (
            shifted.groupby(weekly["region_id"])
            .rolling(window, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

    return weekly

def add_weekly_weather_rollups(weekly):
    """
    Adds rolling weather features over the last 2, 4, 8 and 13 weeks.
    This lets the model use the full 91-day input window.
    """
    weekly = weekly.copy()
    weekly = weekly.sort_values(["region_id", "week_idx"]).reset_index(drop=True)

    base_cols = [
        "prec_sum",
        "prec_mean",
        "humidity_mean",
        "tmp_mean",
        "tmp_max_mean",
        "tmp_min_mean",
        "wind_mean",
        "surf_pre_mean"
    ]

    base_cols = [c for c in base_cols if c in weekly.columns]

    for col in base_cols:
        for window in [2, 4, 8, 13]:
            weekly[f"{col}_roll_mean_{window}"] = (
                weekly.groupby("region_id")[col]
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            weekly[f"{col}_roll_std_{window}"] = (
                weekly.groupby("region_id")[col]
                .rolling(window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )

    if "prec_sum" in weekly.columns:
        for window in [4, 8, 13]:
            weekly[f"prec_sum_roll_sum_{window}"] = (
                weekly.groupby("region_id")["prec_sum"]
                .rolling(window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )

    if "prec_sum_roll_sum_4" in weekly.columns and "prec_sum_roll_sum_13" in weekly.columns:
        weekly["prec_4w_vs_13w"] = weekly["prec_sum_roll_sum_4"] - weekly["prec_sum_roll_sum_13"] / 13 * 4

    if "humidity_mean_roll_mean_4" in weekly.columns and "humidity_mean_roll_mean_13" in weekly.columns:
        weekly["humidity_4w_vs_13w"] = weekly["humidity_mean_roll_mean_4"] - weekly["humidity_mean_roll_mean_13"]

    if "tmp_mean_roll_mean_4" in weekly.columns and "tmp_mean_roll_mean_13" in weekly.columns:
        weekly["tmp_4w_vs_13w"] = weekly["tmp_mean_roll_mean_4"] - weekly["tmp_mean_roll_mean_13"]

    return weekly

def add_region_seasonal_features(train_weekly):
    train_weekly = train_weekly.copy()

    region_mean = (
        train_weekly.groupby("region_id")["score"]
        .mean()
        .rename("region_score_mean")
        .reset_index()
    )

    region_month_mean = (
        train_weekly.groupby(["region_id", "month"])["score"]
        .mean()
        .rename("region_month_score_mean")
        .reset_index()
    )

    return region_mean, region_month_mean


def build_horizon_train_table(train_weekly, region_mean, region_month_mean):
    """
    Creates:
    features at week t -> target score at week t+1 ... t+5

    Memory-safe version:
    uses only the last 450 weeks per region.
    """
    rows = []

    base = train_weekly.copy()

    # Keep only recent history to avoid RAM explosion.
    base = (
        base.sort_values(["region_id", "week_idx"])
        .groupby("region_id")
        .tail(450)
        .reset_index(drop=True)
    )

    base = base.merge(region_mean, on="region_id", how="left")

    for h in range(1, 6):
        print(f"  creating horizon {h} training rows...")

        part = base.copy()
        part["horizon"] = h

        part["target"] = part.groupby("region_id")["score"].shift(-h)
        part["future_month"] = part.groupby("region_id")["month"].shift(-h)
        part["future_dayofyear"] = part.groupby("region_id")["dayofyear"].shift(-h)

        part = part[part["target"].notna()]

        part = part.merge(
            region_month_mean.rename(columns={
                "month": "future_month",
                "region_month_score_mean": "future_region_month_score_mean"
            }),
            on=["region_id", "future_month"],
            how="left"
        )

        rows.append(part)

    result = pd.concat(rows, ignore_index=True)

    result["future_month_sin"] = np.sin(2 * np.pi * result["future_month"] / 12)
    result["future_month_cos"] = np.cos(2 * np.pi * result["future_month"] / 12)
    result["future_doy_sin"] = np.sin(2 * np.pi * result["future_dayofyear"] / 366)
    result["future_doy_cos"] = np.cos(2 * np.pi * result["future_dayofyear"] / 366)

    # Reduce memory usage.
    float_cols = result.select_dtypes(include=["float64"]).columns
    result[float_cols] = result[float_cols].astype("float32")

    int_cols = result.select_dtypes(include=["int64"]).columns
    result[int_cols] = result[int_cols].astype("int32")

    return result

def predict_test_week_scores(train_weekly, test_weekly):
    """
    Stage 1 improved:
    Predict the missing score for each of the 13 observed test weeks sequentially.
    This uses score lags from train history and then feeds predicted test-week scores
    forward into the next test weeks.
    """
    print("Predicting missing scores for the 13 observed test weeks sequentially...")

    train_hist = train_weekly.copy()
    test_rows = test_weekly.copy()

    train_hist = train_hist.sort_values(["region_id", "week_idx"]).reset_index(drop=True)
    test_rows = test_rows.sort_values(["region_id", "week_idx"]).reset_index(drop=True)

    # Train Stage-1 model on historical weekly rows.
    train_rows = train_hist[train_hist["score"].notna()].copy()

    exclude = {"region_id", "date", "score", "target"}
    feature_cols = []

    for col in train_rows.columns:
        if col in exclude:
            continue
        if col not in test_rows.columns:
            continue
        if train_rows[col].dtype == "object":
            continue
        feature_cols.append(col)

    X_train = train_rows[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
    y_train = train_rows["score"].astype(float)

    model = LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=0.4,
        random_state=123,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    predicted_parts = []

    for region, g_test in test_rows.groupby("region_id"):
        g_test = g_test.sort_values("week_idx").copy()

        hist = train_hist[train_hist["region_id"] == region].copy()
        hist = hist.sort_values("week_idx").copy()

        region_predictions = []

        for _, row in g_test.iterrows():
            row_df = pd.DataFrame([row]).copy()

            # Add current score-lag features from hist, including earlier predicted test weeks.
            hist_scores = hist[hist["score"].notna()].sort_values("week_idx")

            for lag in [1, 2, 3, 4, 8, 12]:
                if len(hist_scores) >= lag:
                    row_df[f"score_lag_{lag}"] = float(hist_scores.iloc[-lag]["score"])
                else:
                    row_df[f"score_lag_{lag}"] = np.nan

            for window in [4, 8, 12]:
                last_scores = hist_scores["score"].tail(window)
                row_df[f"score_roll_mean_{window}"] = float(last_scores.mean()) if len(last_scores) > 0 else np.nan
                row_df[f"score_roll_std_{window}"] = float(last_scores.std()) if len(last_scores) > 1 else np.nan

            # Ensure all features exist.
            for col in feature_cols:
                if col not in row_df.columns:
                    row_df[col] = np.nan

            X_one = row_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)

            pred = float(np.clip(model.predict(X_one)[0], 0, 5))

            row_out = row.copy()
            row_out["score"] = pred
            region_predictions.append(row_out)

            # Feed predicted score into history for next test week.
            hist_add = row.copy()
            hist_add["score"] = pred
            hist = pd.concat([hist, pd.DataFrame([hist_add])], ignore_index=True, sort=False)

        predicted_parts.append(pd.DataFrame(region_predictions))

    test_pred = pd.concat(predicted_parts, ignore_index=True)

    print("Predicted test-week score summary:")
    print(test_pred["score"].describe())

    return test_pred

def build_test_horizon_table(test_weekly, train_weekly, region_mean, region_month_mean):
    last_test_week = (
        test_weekly.sort_values(["region_id", "week_idx"])
        .groupby("region_id")
        .tail(1)
        .copy()
    )

    latest_train = train_weekly.sort_values(["region_id", "week_idx"]).copy()

    # Add score lags from latest training history.
    for lag in [1, 2, 3, 4, 8, 12]:
        lag_df = (
            latest_train[["region_id", "week_idx", "score"]]
            .dropna(subset=["score"])
            .sort_values(["region_id", "week_idx"])
            .groupby("region_id")
            .tail(lag)
            .groupby("region_id")
            .head(1)
            [["region_id", "score"]]
            .rename(columns={"score": f"score_lag_{lag}"})
        )

        last_test_week = last_test_week.merge(lag_df, on="region_id", how="left")

    # Rolling score stats from latest training history.
    for window in [4, 8, 12]:
        tmp = (
            latest_train[["region_id", "week_idx", "score"]]
            .dropna(subset=["score"])
            .sort_values(["region_id", "week_idx"])
            .groupby("region_id")
            .tail(window)
            .groupby("region_id")["score"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={
                "mean": f"score_roll_mean_{window}",
                "std": f"score_roll_std_{window}"
            })
        )

        last_test_week = last_test_week.merge(tmp, on="region_id", how="left")

    last_test_week = last_test_week.merge(region_mean, on="region_id", how="left")

    rows = []

    for _, row in last_test_week.iterrows():
        for h in range(1, 6):
            r = row.copy()
            r["horizon"] = h

            future_dayofyear = int(((int(row["dayofyear"]) - 1 + 7 * h) % 365) + 1)
            pseudo_future = pd.Timestamp("2001-01-01") + pd.Timedelta(days=future_dayofyear - 1)

            r["future_dayofyear"] = future_dayofyear
            r["future_month"] = int(pseudo_future.month)

            rows.append(r)

    test_h = pd.DataFrame(rows)

    test_h = test_h.merge(
        region_month_mean.rename(columns={
            "month": "future_month",
            "region_month_score_mean": "future_region_month_score_mean"
        }),
        on=["region_id", "future_month"],
        how="left"
    )

    test_h["future_month_sin"] = np.sin(2 * np.pi * test_h["future_month"] / 12)
    test_h["future_month_cos"] = np.cos(2 * np.pi * test_h["future_month"] / 12)
    test_h["future_doy_sin"] = np.sin(2 * np.pi * test_h["future_dayofyear"] / 366)
    test_h["future_doy_cos"] = np.cos(2 * np.pi * test_h["future_dayofyear"] / 366)

    return test_h

def get_feature_columns(df):
    exclude = {
        "region_id",
        "date",
        "score",
        "target"
    }

    feature_cols = []

    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == "object":
            continue
        feature_cols.append(col)

    return feature_cols


def train_predict(train_h, test_h, feature_cols):
    test_h = test_h.copy()
    test_h["model_pred"] = np.nan

    validation_scores = []

    for h in range(1, 6):
        print(f"\nTraining horizon {h}...")

        tr = train_h[train_h["horizon"] == h].copy()
        te_idx = test_h[test_h["horizon"] == h].index

        # Use last 20% of weeks as validation.
        cutoff = tr["week_idx"].quantile(0.80)

        train_part = tr[tr["week_idx"] < cutoff].copy()
        valid_part = tr[tr["week_idx"] >= cutoff].copy()

        X_train = train_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
        y_train = train_part["target"].astype(float)

        X_valid = valid_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
        y_valid = valid_part["target"].astype(float)

        X_full = tr[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
        y_full = tr["target"].astype(float)

        X_test = test_h.loc[te_idx, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)

        model = LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=0.4,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        model.fit(X_train, y_train)

        valid_pred = np.clip(model.predict(X_valid), 0, 5)
        mae = mean_absolute_error(y_valid, valid_pred)
        validation_scores.append(mae)

        print(f"Horizon {h} validation MAE: {mae:.5f}")

        final_model = LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=0.4,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        final_model.fit(X_full, y_full)

        test_pred = np.clip(final_model.predict(X_test), 0, 5)
        test_h.loc[te_idx, "model_pred"] = test_pred

    print("\nValidation MAE by horizon:", [round(x, 5) for x in validation_scores])
    print("Mean validation MAE:", round(float(np.mean(validation_scores)), 5))

    return test_h


def build_submission(test_h, sample):
    test_h = test_h.copy()

    seasonal = test_h["future_region_month_score_mean"].copy()
    seasonal = seasonal.fillna(test_h["region_score_mean"])
    seasonal = seasonal.fillna(0.85)

    test_h["pred"] = 0.80 * test_h["model_pred"] + 0.20 * seasonal
    test_h["pred"] = test_h["pred"].clip(0, 5)

    rows = []

    for region in sample["region_id"].astype(str):
        g = test_h[test_h["region_id"].astype(str) == region].sort_values("horizon")

        row = {"region_id": region}

        for h in range(1, 6):
            gh = g[g["horizon"] == h]
            if len(gh) == 0:
                row[f"pred_week{h}"] = 0.85
            else:
                row[f"pred_week{h}"] = float(gh.iloc[0]["pred"])

        rows.append(row)

    submission = pd.DataFrame(rows)

    submission = sample[["region_id"]].astype({"region_id": str}).merge(
        submission,
        on="region_id",
        how="left"
    )

    for col in ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]:
        submission[col] = submission[col].fillna(0.85).clip(0, 5)

    return submission[sample.columns]


def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_SUB_PATH)

    train = add_basic_columns(train)
    test = add_basic_columns(test)

    print("Building weekly tables...")

    train_weekly = build_weekly_weather(train)
    train_weekly = add_weekly_weather_rollups(train_weekly)
    train_weekly = add_score_lags(train_weekly)

    test_weekly = build_weekly_weather(test)
    test_weekly = add_weekly_weather_rollups(test_weekly)

    region_mean, region_month_mean = add_region_seasonal_features(train_weekly)

    test_weekly = predict_test_week_scores(train_weekly, test_weekly)

# Important:
# test_weekly has week_idx 0..12, but chronologically it comes AFTER train_weekly.
# Therefore we shift test week_idx behind the last training week per region.
    train_last_week = (
        train_weekly.groupby("region_id")["week_idx"]
        .max()
        .rename("train_last_week_idx")
        .reset_index()
    )

    test_weekly = test_weekly.merge(train_last_week, on="region_id", how="left")
    test_weekly["week_idx"] = test_weekly["week_idx"] + test_weekly["train_last_week_idx"] + 1
    test_weekly = test_weekly.drop(columns=["train_last_week_idx"])

    print("Building horizon train/test tables...")
    train_h = build_horizon_train_table(train_weekly, region_mean, region_month_mean)

# Use train history + predicted test-week scores for recent score lags.
    train_plus_test_weekly = pd.concat(
        [train_weekly, test_weekly],
        ignore_index=True,
        sort=False
    )

    test_h = build_test_horizon_table(
        test_weekly,
        train_plus_test_weekly,
        region_mean,
        region_month_mean
    )

    for col in train_h.columns:
        if col not in test_h.columns and col != "target":
            test_h[col] = np.nan

    for col in test_h.columns:
        if col not in train_h.columns:
            train_h[col] = np.nan

    feature_cols = get_feature_columns(train_h)

    print("train_h shape:", train_h.shape)
    print("test_h shape:", test_h.shape)
    print("number of features:", len(feature_cols))

    test_h = train_predict(train_h, test_h, feature_cols)

    submission = build_submission(test_h, sample)
    submission.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)
    print("Submission shape:", submission.shape)
    print("Null values:", submission.isnull().sum().sum())
    print(submission.head())
    print("\nPrediction summary:")
    print(submission.drop(columns=["region_id"]).describe())

if __name__ == "__main__":
    main()