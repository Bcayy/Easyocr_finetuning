import Levenshtein
from ocr_trainer import run_easyocr_on_dataset
import numpy as np

def wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    d = np.zeros((len(ref_words)+1)*(len(hyp_words)+1), dtype=np.uint8)
    d = d.reshape((len(ref_words)+1, len(hyp_words)+1))
    for i in range(len(ref_words)+1):
        for j in range(len(hyp_words)+1):
            if i == 0:
                d[i][j] = j
            elif j == 0:
                d[i][j] = i
    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i-1][j]+1,      # deletion
                          d[i][j-1]+1,      # insertion
                          d[i-1][j-1]+cost) # substitution
    return d[len(ref_words)][len(hyp_words)] / float(len(ref_words)) if len(ref_words) else 0.0

def cer(ref, hyp):
    return Levenshtein.distance(ref, hyp) / len(ref) if len(ref) else 0.0

def jaccard_similarity(a, b):
    set_a, set_b = set(a), set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 1.0

def levenshtein_ratio(ref, hyp):
    return Levenshtein.ratio(ref, hyp)

def evaluate_all(results):
    wer_list, cer_list, jac_list, lev_list = [], [], [], []
    for item in results:
        gt, pred = item['ground_truth'], item['prediction']
        wer_list.append(wer(gt, pred))
        cer_list.append(cer(gt, pred))
        jac_list.append(jaccard_similarity(gt, pred))
        lev_list.append(levenshtein_ratio(gt, pred))
    print(f"Ortalama WER: {np.mean(wer_list):.3f}")
    print(f"Ortalama CER: {np.mean(cer_list):.3f}")
    print(f"Ortalama Jaccard: {np.mean(jac_list):.3f}")
    print(f"Ortalama Levenshtein Oranı: {np.mean(lev_list):.3f}")

# Kullanım örneği:
if __name__ == "__main__":
    images_dir = "../data/raw/images"
    txt_path = "../data/raw/labels/train.txt"
    results = run_easyocr_on_dataset(txt_path, images_dir, num_samples=5209)
    evaluate_all(results)