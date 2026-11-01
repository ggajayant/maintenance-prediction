from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

project_folder = Path(__file__).resolve().parents[1]
data = pd.read_csv(project_folder / "data/engine_data.csv")
data = data.drop_duplicates()
# All six sensor columns are useful, so no column is removed.
train, test = train_test_split(data, test_size=0.20, random_state=42,
                               stratify=data["Engine Condition"])
split_folder = project_folder / "artifacts/splits"
split_folder.mkdir(parents=True, exist_ok=True)
train.to_csv(split_folder / "train.csv", index=False)
test.to_csv(split_folder / "test.csv", index=False)
print("Training rows:", len(train))
print("Testing rows:", len(test))
