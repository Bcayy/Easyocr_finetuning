import csv
import os

def load_dataset(image_folder, labels_csv):
    """
    Returns:
        samples: List of tuples (image_path, ground_truth_text)
    """
    samples = []
    with open(labels_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(image_folder, row['filename'])
            label = row['words']
            samples.append((img_path, label))
    return samples