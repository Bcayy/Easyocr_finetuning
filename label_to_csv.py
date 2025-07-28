import pandas as pd

data = []
with open("/Users/burakcay/Desktop/ocr_fine_tuning/EasyOCR/trainer/all_data/en_test/test.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            data.append({'filename': parts[0], 'words': parts[1]})

df = pd.DataFrame(data)
df.to_csv("//Users/burakcay/Desktop/ocr_fine_tuning/EasyOCR/trainer/all_data/en_test/labels.csv", index=False)