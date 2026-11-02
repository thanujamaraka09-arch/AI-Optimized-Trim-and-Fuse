# ============================================================
# AI-OPTIMIZED TRIM & FUSE
# MODULE 5: MCU FUSE MAP VALIDATION
# ============================================================

import pandas as pd

# STEP 1: Load generated fuse map
print("\nLoading generated fuse map...")

fuse = pd.read_csv("mcu_generated_fuse_map.csv")

# STEP 2: Load original dataset
df = pd.read_csv("data/mcu_trim_fuse.csv")

# STEP 3: Load fuse schema
schema = pd.read_csv("data/mcu_fuse_schema.csv")

print("Fuse map records:", len(fuse))
print("Schema fields:", len(schema))

# STEP 4: Join die information
source = df[[
    "die_id",
    "derivative",
    "ts_cal1_30C_counts"
]]

fuse = fuse.merge(
    source,
    on="die_id",
    how="left",
    validate="one_to_one"
)

# STEP 5: Define predicted trim fields
trim_columns = {
    "BG_TRIM": "bg_predicted_code",
    "HSI_TRIM": "hsi_predicted_code",
    "LDO_TRIM": "ldo_predicted_code"
}

# STEP 6: Convert values safely
def parse_value(value):
    value = str(value).strip()
    return int(value, 0)

# STEP 7: Validate and generate fuse maps
fuse_words = []
validation_status = []
error_messages = []

for _, row in fuse.iterrows():

    errors = []
    fuse_word = 0

    derivative = row["derivative"]

    # Check source information
    if pd.isna(derivative):
        errors.append("Missing derivative")

    if pd.isna(row["ts_cal1_30C_counts"]):
        errors.append("Missing temperature calibration")

    # Check all trim codes
    for field, column in trim_columns.items():

        value = row[column]

        schema_row = schema[
            schema["field"] == field
        ].iloc[0]

        nbits = int(schema_row["nbits"])
        maximum = (1 << nbits) - 1

        if pd.isna(value) or float(value) != int(value):
            errors.append(field + " is not an integer")
            continue

        value = int(value)

        if value < 0 or value > maximum:
            errors.append(field + " is out of range")

    # Generate fields using schema
    if not errors:

        for _, field_row in schema.iterrows():

            field = field_row["field"]
            lsb = int(field_row["lsb"])
            nbits = int(field_row["nbits"])

            if field == "BG_TRIM":
                value = int(row["bg_predicted_code"])

            elif field == "HSI_TRIM":
                value = int(row["hsi_predicted_code"])

            elif field == "LDO_TRIM":
                value = int(row["ldo_predicted_code"])

            elif field == "TS_CAL1":
                value = int(row["ts_cal1_30C_counts"])

            elif field == "RESERVED":
                value = 0

            else:
                value = parse_value(
                    field_row[derivative]
                )

            maximum = (1 << nbits) - 1

            if value < 0 or value > maximum:
                errors.append(field + " exceeds bit capacity")
                continue

            fuse_word |= value << lsb

    # Save validation result
    if errors:
        validation_status.append("FAIL")
        error_messages.append("; ".join(errors))
        fuse_words.append("NOT GENERATED")

    else:
        validation_status.append("PASS")
        error_messages.append("None")
        fuse_words.append(
            "0x" + format(fuse_word, "012X")
        )

# STEP 8: Add results
fuse["fuse_map_hex_generated"] = fuse_words
fuse["validation_status"] = validation_status
fuse["validation_errors"] = error_messages

# STEP 9: Save validation report
fuse.to_csv(
    "mcu_fuse_validation_report.csv",
    index=False
)

# STEP 10: Display results
total = len(fuse)
passed = validation_status.count("PASS")
failed = validation_status.count("FAIL")

print("\n===== FUSE MAP VALIDATION RESULTS =====")

print("Total fuse maps:", total)
print("Valid fuse maps:", passed)
print("Invalid fuse maps:", failed)

if total > 0:
    print(
        "Validation success rate:",
        round(passed / total * 100, 2),
        "%"
    )

print("\nSample validation results:")

print(
    fuse[[
        "die_id",
        "fuse_map_hex_generated",
        "validation_status",
        "validation_errors"
    ]].head(10).to_string(index=False)
)

print("\nValidation report saved successfully!")
print("\nMODULE 5 COMPLETED!")