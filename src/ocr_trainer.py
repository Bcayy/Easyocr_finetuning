import easyocr
from dataset_loader import TxtDatasetLoader

def run_easyocr_on_dataset(txt_path, images_dir, num_samples=5209):
    loader = TxtDatasetLoader(txt_path, images_dir)
    data = loader.load_and_preprocess()

    reader = easyocr.Reader(['tr'], gpu=False)

    results = []
    for idx, item in enumerate(data[:num_samples]):
        img = item['image']
        true_label = item['label']

        # EasyOCR prediction (tek satır)
        prediction = reader.readtext(img, detail=0, paragraph=False)
        # prediction list döner, en uzununu al:
        pred_text = max(prediction, key=len) if prediction else ""
        results.append({'ground_truth': true_label, 'prediction': pred_text})

        if idx % 10 == 0:
            print(f"{idx+1}/{num_samples} tamamlandı.")

    return results

if __name__ == "__main__":
    images_dir = "../data/raw/images"
    txt_path = "../data/raw/labels/train.txt"

    results = run_easyocr_on_dataset(txt_path, images_dir, num_samples=5209)
    print("İlk 5 örnek prediction:")
    for r in results[:5]:
        print("Gerçek:", r['ground_truth'], "| Prediction:", r['prediction'])