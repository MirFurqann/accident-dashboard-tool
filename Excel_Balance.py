import pandas as pd
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("Road_Accident_Data.csv")

# Map severities
df["Severity"] = df["Accident_Severity"].str.lower().str.strip().map({
    "slight": "Low",
    "serious": "Medium",
    "fatal": "High"
})

# Keep only valid severities
df = df[df["Severity"].isin(["Low", "Medium", "High"])]

# Define total target dataset size and class proportions
total_target_size = 300000  # adjust if needed
proportions = {"Low": 0.31, "Medium": 0.36, "High": 0.33}
target_counts = {cls: int(total_target_size * prop) for cls, prop in proportions.items()}

# Define the extra boost factor for Road Types 0 and 3
extra_boost = 1.05  # +5%

balanced_severity_dfs = []

print("\n=== STARTING BALANCING ===")
for severity, target_n in target_counts.items():
    df_sev = df[df["Severity"] == severity]
    road_types = df_sev["Road_Type"].unique()
    
    # Calculate total adjusted weights
    weights = []
    for rtype in road_types:
        if rtype in [0, 3]:
            weights.append(extra_boost)
        else:
            weights.append(1)
    total_weight = sum(weights)
    
    # Calculate how many samples per Road Type
    target_per_rtype = {rtype: int((w / total_weight) * target_n)
                        for rtype, w in zip(road_types, weights)}
    
    resampled_roadtype_groups = []
    total_collected = 0
    
    print(f"\n→ {severity} class: original total = {len(df_sev)}, per road type targets = {target_per_rtype}")
    
    for rtype in road_types:
        group = df_sev[df_sev["Road_Type"] == rtype]
        sample_n = target_per_rtype[rtype]
        
        if group.empty or sample_n == 0:
            continue
        
        replace = len(group) < sample_n
        resampled = resample(
            group,
            replace=replace,
            n_samples=sample_n,
            random_state=42
        )
        
        resampled_roadtype_groups.append(resampled)
        total_collected += len(resampled)
        
        print(f"  Road Type {rtype}: original={len(group)}, sampled={len(resampled)}")
    
    print(f"✔ Total collected for {severity}: {total_collected}")
    balanced_severity_dfs.append(pd.concat(resampled_roadtype_groups))

# Combine and shuffle final dataset
balanced_df = pd.concat(balanced_severity_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

# Check final counts
print("\n=== FINAL COUNTS ===")
print(balanced_df["Severity"].value_counts())
print(balanced_df.groupby(["Severity", "Road_Type"]).size())

# Save to CSV
balanced_df.to_csv("balanced_realistic_accident_dataset.csv", index=False)
print("\n✅ Balanced dataset saved as 'balanced_realistic_accident_dataset.csv'")
