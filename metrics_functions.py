"""
Funcții pentru calculul metricilor de calitate a imaginii
"""
import numpy as np


def calculate_mse(img1, img2):
    """
    Mean Squared Error
    """
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def calculate_psnr(img1, img2):
    """
    Peak Signal-to-Noise Ratio
    """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    """
    Structural Similarity Index (SSIM)
    Implementare simplificată
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constante pentru stabilitate
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Media
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Varianța
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    
    # Covarianța
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    # SSIM formula
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim
