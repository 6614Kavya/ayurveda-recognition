import cv2
import numpy as np

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_light_denoise(image, mask, edge_trim_px=2):
    result = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)
    k = edge_trim_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tight_mask = cv2.erode(mask, kernel)
    result[tight_mask == 0] = [255, 255, 255]
    return result