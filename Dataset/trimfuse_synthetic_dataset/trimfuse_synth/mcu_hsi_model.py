import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: LOAD DATASET
print("Loading MCU dataset...")

df = pd.read_csv("data/mcu_trim_fuse.csv")

# STEP 2: SELECT MCU-A
df = df[df["derivative"] == "MCU-A"].copy()

print("MCU-A dataset:", df.shape)

# STEP 3: SELECT FEATURES AND TARGET
features = [
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
    "temp_grade",
    "hsi_untrim_MHz"
]

target = "hsi_trim_code_opt"

X = df[features]
y = df[target]
groups = df["wafer_id"]

# STEP 4: PREPROCESSING
categorical_features = ["package", "temp_grade"]

numeric_features = [
    col for col in features
    if col not in categorical_features
]

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "num",
        "passthrough",
        numeric_features
    )
])

# STEP 5: TRAIN-TEST SPLIT BY WAFER
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_wafers = set(groups.iloc[train_idx])
test_wafers = set(groups.iloc[test_idx])

print("\nTraining wafers:", len(train_wafers))
print("Testing wafers:", len(test_wafers))
print("Common wafers:", train_wafers.intersection(test_wafers))
print("No wafer overlap:", train_wafers.isdisjoint(test_wafers))

# STEP 6: CREATE MODEL
model = Pipeline([
    ("preprocessing", preprocessor),
    (
        "regressor",
        GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    )
])

# STEP 7: TRAIN MODEL
print("\nTraining HSI model...")

model.fit(X_train, y_train)

print("Training completed!")

# STEP 8: PREDICT TEST DATA
predictions = model.predict(X_test)
rounded_predictions = np.rint(predictions).astype(int)

# STEP 9: EVALUATE MODEL
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

exact_accuracy = np.mean(
    rounded_predictions == y_test.to_numpy()
) * 100

within_one = np.mean(
    np.abs(rounded_predictions - y_test.to_numpy()) <= 1
) * 100

print("\n======================================")
print("         MCU HSI MODEL RESULTS")
print("======================================")

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))
print("Exact accuracy:", round(exact_accuracy, 2), "%")
print("Within +/-1 code:", round(within_one, 2), "%")

print("\nActual vs Predicted:")

for actual, predicted in zip(
    y_test.head(10),
    predictions[:10]
):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")

# STEP 10: SAVE MODEL
joblib.dump(model, "mcu_hsi_model.joblib")

# STEP 11: SAVE PREDICTIONS WITH WAFER IDs
results = pd.DataFrame({
    "die_id": df.iloc[test_idx]["die_id"].values,
    "wafer_id": groups.iloc[test_idx].values,
    "Actual_Trim_Code": y_test.values,
    "Predicted_Trim_Code": rounded_predictions
})

results.to_csv(
    "mcu_hsi_predictions.csv",
    index=False
)

print("\nHSI model saved successfully!")
print("HSI predictions saved with wafer IDs!")

print("\nMODULE 2 COMPLETED!")