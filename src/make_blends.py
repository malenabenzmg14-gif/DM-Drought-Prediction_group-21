import pandas as pd

SUB_390 = "outputs/submission_step12_train390.csv"
SUB_450 = "outputs/submission_step12_train450.csv"

COLS = [f"pred_week{i}" for i in range(1, 6)]

a = pd.read_csv(SUB_390)
b = pd.read_csv(SUB_450)

# 50/50 blend
out = a.copy()
out[COLS] = 0.50 * a[COLS] + 0.50 * b[COLS]
out.to_csv("outputs/submission_blend_390_450_50_50.csv", index=False)

# Slightly favor the better 450-week model
out = a.copy()
out[COLS] = 0.30 * a[COLS] + 0.70 * b[COLS]
out.to_csv("outputs/submission_blend_390_450_30_70.csv", index=False)

# Strongly favor the better 450-week model
out = a.copy()
out[COLS] = 0.20 * a[COLS] + 0.80 * b[COLS]
out.to_csv("outputs/submission_blend_390_450_20_80.csv", index=False)

print("Created blend submissions.")
print("390 summary:")
print(a[COLS].describe())
print("\n450 summary:")
print(b[COLS].describe())