from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parents[1]
data = pd.read_csv(project_folder / "data/engine_data.csv")
expected_columns = ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure",
                    "lub oil temp", "Coolant temp", "Engine Condition"]
if set(data.columns) != set(expected_columns):
    raise ValueError("The dataset does not have the expected columns.")
if not data["Engine Condition"].isin([0, 1]).all():
    raise ValueError("Engine Condition must contain 0 or 1.")
print("Dataset registered successfully.")
print("Rows and columns:", data.shape)
print("Missing values:\n", data.isnull().sum())
print("Condition counts:\n", data["Engine Condition"].value_counts())
