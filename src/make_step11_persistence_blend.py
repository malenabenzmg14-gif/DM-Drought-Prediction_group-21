import pandas as pd
import numpy as np


STEP7_PATH = "submission_final_step7_rollups.csv"
TEST_WEEK_SCORE_SOURCE = "submission_final_step7_rollups.csv"  # only used for structure check
OUTPUT_PATH = "submission_step11_persistence_blend.csv"


def main():
    # Load current best submission
    step7 = pd.read_csv(STEP7_PATH)

    # Persistence proxy:
    # Use pred_week1 as approximation of current/near-future drought state.
    # Blend later weeks toward pred_week1, but only lightly.
    out = step7.copy()

    # Keep week 1 unchanged.
    out["pred_week1"] = step7["pred_week1"]

    # Light persistence blend.
    out["pred_week2"] = 0.90 * step7["pred_week2"] + 0.10 * step7["pred_week1"]
    out["pred_week3"] = 0.85 * step7["pred_week3"] + 0.15 * step7["pred_week1"]
    out["pred_week4"] = 0.80 * step7["pred_week4"] + 0.20 * step7["pred_week1"]
    out["pred_week5"] = 0.75 * step7["pred_week5"] + 0.25 * step7["pred_week1"]

    for col in ["pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]:
        out[col] = out[col].clip(0, 5)

    out.to_csv(OUTPUT_PATH, index=False)

    print("Original Step 7:")
    print(step7.drop(columns=["region_id"]).describe())

    print("\nStep 11 persistence blend:")
    print(out.drop(columns=["region_id"]).describe())

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()