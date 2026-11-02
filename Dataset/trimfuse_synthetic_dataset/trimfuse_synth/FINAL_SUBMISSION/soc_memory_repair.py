import pandas as pd

# Load memory repair dataset
df = pd.read_csv("data/memory_repair_bitmaps.csv")

print("Memory repair dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Analyze failed memory cells
def analyze_repair(row):

    fail_cells = str(row["fail_cells"])

    failed_rows = set()
    failed_cols = set()

    if fail_cells and fail_cells != "nan":

        for cell in fail_cells.split(";"):

            if ":" in cell:
                r, c = cell.split(":")
                failed_rows.add(int(r))
                failed_cols.add(int(c))

    required_rows = len(failed_rows)
    required_cols = len(failed_cols)

    available_rows = int(row["spare_rows"])
    available_cols = int(row["spare_cols"])

    if required_rows <= available_rows and required_cols <= available_cols:
        status = "REPAIRABLE"
    else:
        status = "NOT REPAIRABLE"

    return pd.Series({
        "failed_row_count": required_rows,
        "failed_col_count": required_cols,
        "available_spare_rows": available_rows,
        "available_spare_cols": available_cols,
        "estimated_repair_status": status
    })


# Analyze all memory arrays
analysis = df.apply(analyze_repair, axis=1)

# Combine results
report = pd.concat([df, analysis], axis=1)

# Compare with optimal repair results
report["matches_optimal_result"] = (
    report["estimated_repair_status"] ==
    report["optimal_repairable"].map({
        True: "REPAIRABLE",
        False: "NOT REPAIRABLE"
    })
)

# Display results
print("\n========== MEMORY REPAIR RESULTS ==========")

print("\nOptimal repairability:")
print(df["optimal_repairable"].value_counts())

print("\nEstimated repairability:")
print(report["estimated_repair_status"].value_counts())

accuracy = report["matches_optimal_result"].mean() * 100

print("\nAgreement with optimal result:",
      round(accuracy, 2), "%")

# Save report
report.to_csv("soc_memory_repair_report.csv", index=False)

print("\nMEMORY REPAIR ANALYSIS COMPLETED!")
print("Report saved: soc_memory_repair_report.csv")