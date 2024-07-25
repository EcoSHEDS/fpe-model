import numpy as np
import pandas as pd
from pathlib import Path

path = Path("~/azurefiles/projects/streamflow/jeff_data/stations").expanduser()
stations = [x for x in path.iterdir() if x.is_dir()]
random_seeds = [1632, 2927, 3274, 8436]  # List of random seeds

# Define the training sizes for nested subsets
train_sizes = [
    100,
    200,
    300,
    400,
    500,
    750,
    1000,
    1250,
    1500,
    2000,
    2500,
    3000,
    4000,
]

for random_seed in random_seeds:  # Iterate over each random seed
    for station in stations:
        print(f"Processing station: {station} with seed: {random_seed}")
        save_dir = station / f"input_{random_seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get the full dataset of pairs
        station_annotation_dir = station / "input"
        station_pairs_f = station_annotation_dir / "pairs.csv"
        station_pairs = pd.read_csv(station_pairs_f)

        # Split the dataset into training and validation sets
        station_train_pairs = station_pairs[station_pairs["split"] == "train"]
        station_val_pairs = station_pairs[station_pairs["split"] == "val"]

        # Save the full dataset to a CSV file with number of training pairs for convenience
        station_pairs.to_csv(
            save_dir / f"pairs-train_{len(station_train_pairs)}.csv", index=False
        )

        # Create a shuffled copy of the training set
        station_train_pairs_shuffled = station_train_pairs.sample(
            frac=1, random_state=random_seed
        )

        # Get the training sizes that are less than the number of records in the training set
        station_train_sizes = [x for x in train_sizes if x < len(station_train_pairs)]
        print(
            f"Station {station} has {len(station_train_sizes)} train sizes with seed: {random_seed}"
        )

        # Split the training set into nested subsets
        for size in station_train_sizes:
            station_train_subset = station_train_pairs_shuffled.iloc[:size]
            # Combine the training subset with the validation set
            station_train_val_subset = pd.concat(
                [station_train_subset, station_val_pairs]
            )

            # Save the subset to a CSV file
            station_train_val_subset.to_csv(
                save_dir / f"pairs-train_{size}.csv", index=False
            )
