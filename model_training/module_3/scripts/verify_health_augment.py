import pandas as pd
df = pd.read_csv("processed/features/health_features_top.csv")  # adjust path

ok = df[df.qc_pass == True]
aug = ok[ok.is_augmented == True]
orig = ok[ok.is_augmented == False]

print("original rows:", len(orig), " augmented rows:", len(aug))
print("hole_count == -1 among augmented (should be 0):", (aug.hole_count == -1).sum())
print("hole_count == -1 among original (should be 0):", (orig.hole_count == -1).sum())
print("level counts:\n", ok.level.value_counts())