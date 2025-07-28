import os
import cv2

class TxtDatasetLoader:
    def __init__(self, txt_path, images_dir):
        self.txt_path = txt_path
        self.images_dir = images_dir

    def load_data(self):
        data = []
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    continue
                img_name, label = parts
                img_path = os.path.join(self.images_dir, img_name)
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    data.append({'image': image, 'label': label, 'img_path': img_path})
        return data

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
        return denoised

    def load_and_preprocess(self):
        data = []
        for item in self.load_data():
            processed_img = self.preprocess_image(item['image'])
            data.append({'image': processed_img, 'label': item['label'], 'img_path': item['img_path']})
        return data

# BU KISMI EN ALTA YAZ
if __name__ == "__main__":
    images_dir = "../data/raw/images"
    txt_path = "../data/raw/labels/train.txt"

    loader = TxtDatasetLoader(txt_path, images_dir)
    data = loader.load_and_preprocess()
    print(f"Yüklenen örnek sayısı: {len(data)}")
    print("Bir örnek label:", data[0]['label'])