# ============================================================
# AI-OPTIMIZED TRIM & FUSE
# MODULE 4: MCU FUSE MAP GENERATOR
# ============================================================

import pandas as pd
import joblib

# STEP 1: Load dataset
print("\nLoading MCU dataset...")

df = pd.read_csv("data/mcu_trim_fuse.csv")

# STEP 2: Select MCU-A records
df = df[df["derivative"] == "MCU-A"].copy()

# STEP 3: Load trained AI models
print("Loading trained AI models...")

bg_model = joblib.load("mcu_bandgap_model.joblib")
hsi_model = joblib.load("mcu_hsi_model.joblib")
ldo_model = joblib.load("mcu_ldo_model.joblib")

# STEP 4: Select common input features
common_features = [
    "die_x",
    "die_y",
    "x_mm",
    "y_mm",
    "radius_mm",
    "pcm_vth_n_mV",
    "pcm_vth_p_mV",
    "pcm_rs_ohm_sq",
    "pcm_tox_A",
    "pcm_mobility_rel",
    "package",
    "temp_grade"
]

# STEP 5: Prepare input data for each model
bg_features = common_features + ["bg_untrim_V"]
hsi_features = common_features + ["hsi_untrim_MHz"]
ldo_features = common_features + ["ldo_untrim_V"]

# STEP 6: Predict trim codes
print("\nPredicting trim codes...")

df["bg_predicted_code"] = (
    bg_model.predict(df[bg_features]).round().astype(int)
)

df["hsi_predicted_code"] = (
    hsi_model.predict(df[hsi_features]).round().astype(int)
)

df["ldo_predicted_code"] = (
    ldo_model.predict(df[ldo_features]).round().astype(int)
)

# STEP 7: Display predictions
print("\n===== MCU TRIM PREDICTIONS =====")

print(df[[
    "die_id",
    "bg_predicted_code",
    "hsi_predicted_code",
    "ldo_predicted_code"
]].head(10).to_string(index=False))

# STEP 8: Generate fuse map table
fuse_map = df[[
    "die_id",
    "bg_predicted_code",
    "hsi_predicted_code",
    "ldo_predicted_code"
]].copy()

# STEP 9: Save predicted trim and fuse map data
fuse_map.to_csv(
    "mcu_generated_fuse_map.csv",
    index=False
)

print("\nFuse map prediction file saved successfully!")

# STEP 10: Compare predicted codes with existing codes
print("\n===== PREDICTION COMPARISON =====")

print(
    "Bandgap within +/-1:",
    (
        (df["bg_predicted_code"] -
         df["bg_trim_code_opt"]).abs() <= 1
    ).mean() * 100,
    "%"
)

print(
    "HSI within +/-1:",
    (
        (df["hsi_predicted_code"] -
         df["hsi_trim_code_opt"]).abs() <= 1
    ).mean() * 100,
    "%"
)

print(
    "LDO within +/-1:",
    (
        (df["ldo_predicted_code"] -
         df["ldo_trim_code_opt"]).abs() <= 1
    ).mean() * 100,
    "%"
)

print("\nMODULE 4 COMPLETED!")