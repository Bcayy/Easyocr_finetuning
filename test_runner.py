import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import Model
from dataset import OCRDataset, AlignCollate
from utils import CTCLabelConverter

# === 1. Opt objesi (eğitimle aynı parametrelerle!)
class Opt:
    experiment_name = 'ocr_fine_tuning'
    Transformation = 'None'
    FeatureExtraction = 'VGG'
    SequenceModeling = 'BiLSTM'
    Prediction = 'CTC'
    num_fiducial = 20
    imgH = 64
    imgW = 600
    input_channel = 1
    output_channel = 256
    hidden_size = 256
    batch_max_length = 34
    character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzçğıöşüÇĞİÖŞÜ'
    num_class = len(character)
    rgb = False
    sensitive = True
    PAD = True
    contrast_adjust = 0.0
    data_filtering_off = False
    workers = 2  # testte çok fazla worker gerekmez

opt = Opt()

# === 2. Karakter Converter
converter = CTCLabelConverter(opt.character)

# === 3. Model yükle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(opt).to(device)
model.load_state_dict(torch.load(
    '/Users/burakcay/Desktop/ocr_fine_tuning/EasyOCR/trainer/saved_models/ocr_fine_tuning/tr_recognet.pth',
    map_location=device
))
model.eval()

# === 4. Test dataset/dataloader hazırla
test_root = '/Users/burakcay/Desktop/ocr_fine_tuning/EasyOCR/trainer/all_data/en_test'
test_dataset = OCRDataset(test_root, opt)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,  # büyük batch ile daha hızlı olur
    shuffle=False,
    num_workers=opt.workers,
    collate_fn=AlignCollate(opt.imgH, opt.imgW, opt.PAD, opt.contrast_adjust)
)

# === 5. Test loop
all_preds = []
all_gts = []

with torch.no_grad():
    for image_tensors, labels in test_loader:
        image_tensors = image_tensors.to(device)
        batch_size = image_tensors.size(0)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        preds = model(image_tensors, text_for_pred, is_train=False)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, preds_size)

        all_preds.extend(preds_str)
        all_gts.extend(labels)

# === 6. Sonuçları yazdır (ilk 10 örnek)
print('\nİlk 10 örnek:')
for i in range(10):
    print(f'GT: {all_gts[i]:20} | Pred: {all_preds[i]}')

# === 7. Accuracy ve norm_ED hesapla (daha önce yazdığımız fonksiyonları kullan)
from evaluation_metrics import calculate_accuracy, calculate_norm_ed
acc = calculate_accuracy(all_preds, all_gts)
ned = calculate_norm_ed(all_preds, all_gts)
print(f'\nAccuracy: {acc:.2f}%, Norm_ED: {ned:.4f}')

# === 8. İstersen sonuçları CSV'ye yazabilirsin
import csv
with open('finetuned_test_results.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['gt', 'pred'])
    for gt, pred in zip(all_gts, all_preds):
        writer.writerow([gt, pred])