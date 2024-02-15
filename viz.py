import json
from collections import Counter
from pathlib import Path

def count_unique_labels(directory):
    # Create a Counter object to store the counts of each label
    label_counter = Counter()

    # Iterate over the JSON files in the directory
    for file_path in Path(directory).glob("*.json"):
        with open(file_path) as file:
            data = json.load(file)
            labels = data["label"]
            label_counter.update(labels)

    # Get the unique labels and their occurrences
    unique_labels = label_counter.keys()
    label_occurrences = label_counter.values()

    return unique_labels, label_occurrences

# Usage example
directory = "/home/orfeu/Documents/cours/3A/MLOps/mlops-tps-2024/datasets/plastic_in_river/annotations/"
unique_labels, label_occurrences = count_unique_labels(directory)

# Print the unique labels and their occurrences
for label, occurrence in zip(unique_labels, label_occurrences):
    print(f"Label: {label}, Occurrences: {occurrence}")