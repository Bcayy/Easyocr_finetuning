# evaluation_metrics.py
import numpy as np
import editdistance

def calculate_accuracy(predictions, ground_truths):
    correct = sum([pred == gt for pred, gt in zip(predictions, ground_truths)])
    return correct / len(predictions) * 100

def calculate_norm_ed(predictions, ground_truths):
    """
    Normalized edit distance (Karakter bazlı benzerlik skoru: 1-perfect, 0-worst)
    """
    total_norm_ed = 0
    for pred, gt in zip(predictions, ground_truths):
        if len(gt) == 0:
            norm_ed = 1 if len(pred) == 0 else 0
        else:
            norm_ed = 1 - (editdistance.eval(pred, gt) / max(len(gt), len(pred)))
        total_norm_ed += norm_ed
    return total_norm_ed / len(predictions)