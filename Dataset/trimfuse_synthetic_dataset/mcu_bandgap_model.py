import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load dataset
df = pd.read_csv("data/mcu_trim_fuse.csv")

# 2. Select MCU-A records
df = df[df["derivative"] == "MCU-A"].copy()

print("MCU-A dataset:", df.shape)

# 3. Select input features
features = [
    "die_x", "die_y", "x_mm", "y_mm", "radius_mm",
    "pcm_vth_n_mV", "pcm_vth_p_mV",
    "pcm_rs_ohm_sq", "pcm_tox_A", "pcm_mobility_rel",
    "package", "temp_grade", "bg_untrim_V"
]

target = "bg_trim_code_opt"

X = df[features]
y = df[target]
groups = df["wafer_id"]

# 4. Preprocessing
numeric_features = [
    col for col in features
    if col not in ["package", "temp_grade"]
]

categorical_features = ["package", "temp_grade"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

# 5. Split by wafer
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training wafers:", groups.iloc[train_idx].nunique())
print("Testing wafers:", groups.iloc[test_idx].nunique())

# 6. Create AI model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

# 7. Train model
print("\nTraining AI model...")
model.fit(X_train, y_train)

# 8. Predict
predictions = model.predict(X_test)

# 9. Evaluate
mae = mean_absolute_error(y_test, predictions)

exact = (predictions.round() == y_test).mean() * 100

within_one = (
    (predictions.round() - y_test).abs() <= 1
).mean() * 100

print("\n===== MCU BANDGAP RESULTS =====")
print(f"MAE: {mae:.3f} codes")
print(f"Exact accuracy: {exact:.2f}%")
print(f"Within +/- 1 code: {within_one:.2f}%")

print("\nActual vs Predicted:")
for actual, predicted in zip(y_test.iloc[:10], predictions[:10]):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")

print("\nModel training completed!")