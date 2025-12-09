"""
Funcții pentru aplicarea filtrelor pe imagini zgomotoase
Portate din MATLAB în Python/NumPy/SciPy
"""
import numpy as np
from scipy.ndimage import median_filter as scipy_median
from scipy.ndimage import convolve, generic_filter
from scipy.signal import medfilt2d


def mean_filter(img, kernel_size=3):
    """
    Filtru de mediere (Mean Filter)
    """
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return convolve(img, kernel, mode='reflect').astype(np.uint8)


def median_filter(img, kernel_size=3):
    """
    Filtru median
    """
    return scipy_median(img, size=kernel_size).astype(np.uint8)


def adaptive_median_filter(img, Smax=7):
    """
    Filtru median adaptiv
    Implementare bazată pe algoritmul din Gonzalez & Woods
    """
    if Smax % 2 == 0 or Smax <= 1:
        raise ValueError("Smax trebuie să fie impar și > 1")
    
    M, N = img.shape
    img_out = np.zeros_like(img)
    already_processed = np.zeros((M, N), dtype=bool)
    
    # Padding pentru a evita problemele la margini
    pad_size = Smax // 2
    img_padded = np.pad(img, pad_size, mode='reflect')
    
    for k in range(3, Smax + 1, 2):
        half_k = k // 2
        for i in range(M):
            for j in range(N):
                if already_processed[i, j]:
                    continue
                
                # Extrage regiunea
                region = img_padded[i:i+k, j:j+k]
                zmin = np.min(region)
                zmax = np.max(region)
                zmed = np.median(region)
                zxy = img[i, j]
                
                # Level A
                if zmed > zmin and zmed < zmax:
                    # Level B
                    if zxy > zmin and zxy < zmax:
                        img_out[i, j] = zxy
                    else:
                        img_out[i, j] = zmed
                    already_processed[i, j] = True
        
        if np.all(already_processed):
            break
    
    # Pixelii neprocessați primesc mediana cu Smax
    img_out[~already_processed] = median_filter(img, Smax)[~already_processed]
    
    return img_out.astype(np.uint8)


def geometric_mean_filter(img, kernel_size=3):
    """
    Filtru medie geometrică
    """
    pad_size = kernel_size // 2
    img_padded = np.pad(img.astype(np.float64), pad_size, mode='reflect')
    M, N = img.shape
    img_out = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            region = img_padded[i:i+kernel_size, j:j+kernel_size]
            # Evită log(0) adăugând o valoare mică
            region = np.maximum(region, 1e-10)
            img_out[i, j] = np.exp(np.mean(np.log(region)))
    
    return np.clip(img_out, 0, 255).astype(np.uint8)


def harmonic_mean_filter(img, kernel_size=3):
    """
    Filtru medie armonică
    """
    pad_size = kernel_size // 2
    img_padded = np.pad(img.astype(np.float64), pad_size, mode='reflect')
    M, N = img.shape
    img_out = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            region = img_padded[i:i+kernel_size, j:j+kernel_size]
            # Evită împărțirea la 0
            region = np.maximum(region, 1e-10)
            img_out[i, j] = kernel_size ** 2 / np.sum(1.0 / region)
    
    return np.clip(img_out, 0, 255).astype(np.uint8)


def contra_harmonic_mean_filter(img, kernel_size=3, Q=1.5):
    """
    Filtru medie contra-armonică
    Q > 0: reduce zgomot sare (pepper)
    Q < 0: reduce zgomot piper (salt)
    """
    pad_size = kernel_size // 2
    img_padded = np.pad(img.astype(np.float64), pad_size, mode='reflect')
    M, N = img.shape
    img_out = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            region = img_padded[i:i+kernel_size, j:j+kernel_size]
            region = np.maximum(region, 1e-10)
            numerator = np.sum(region ** (Q + 1))
            denominator = np.sum(region ** Q)
            img_out[i, j] = numerator / (denominator + 1e-10)
    
    return np.clip(img_out, 0, 255).astype(np.uint8)


def midpoint_filter(img, kernel_size=3):
    """
    Filtru midpoint (medie între min și max)
    """
    pad_size = kernel_size // 2
    img_padded = np.pad(img, pad_size, mode='reflect')
    M, N = img.shape
    img_out = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            region = img_padded[i:i+kernel_size, j:j+kernel_size]
            img_out[i, j] = (np.min(region) + np.max(region)) / 2
    
    return img_out.astype(np.uint8)


def wiener_filter_img(img, PSF=None, NSR=0.1):
    """
    Filtru Wiener pentru restaurare
    PSF: Point Spread Function (kernel blur)
    NSR: Noise-to-Signal Ratio
    """
    if PSF is None:
        # PSF implicit - motion blur
        PSF = motion_kernel(7, 90)
    
    img_float = img.astype(np.float64) / 255.0
    
    # FFT
    G = np.fft.fft2(img_float)
    H = np.fft.fft2(PSF, s=img.shape)
    
    # Filtru Wiener
    H_conj = np.conj(H)
    Hw = H_conj / (np.abs(H) ** 2 + NSR)
    F_hat = Hw * G
    
    # IFFT
    img_restored = np.real(np.fft.ifft2(F_hat))
    img_restored = np.clip(img_restored, 0, 1)
    
    return (img_restored * 255).astype(np.uint8)


def inverse_filter_img(img, PSF):
    """
    Filtru invers pentru restaurare
    """
    img_float = img.astype(np.float64) / 255.0
    
    G = np.fft.fft2(img_float)
    H = np.fft.fft2(PSF, s=img.shape)
    
    # Evită împărțirea la 0
    H_safe = np.where(np.abs(H) < 0.01, 0.01, H)
    F_hat = G / H_safe
    
    img_restored = np.real(np.fft.ifft2(F_hat))
    img_restored = np.clip(img_restored, 0, 1)
    
    return (img_restored * 255).astype(np.uint8)


def notch_filter_img(img, coords, rad=7, n_order=4):
    """
    Filtru Notch Butterworth pentru eliminarea zgomotului periodic
    coords: listă de coordonate [(x1, y1), (x2, y2), ...]
    rad: raza notch-ului
    n_order: ordinul filtrului Butterworth
    """
    M, N = img.shape
    img_float = img.astype(np.float64) / 255.0
    
    # FFT
    G = np.fft.fftshift(np.fft.fft2(img_float))
    
    # Centrul FFT
    cx_centru = N // 2
    cy_centru = M // 2
    
    # Creează grid de coordonate
    V, U = np.meshgrid(np.arange(N), np.arange(M))
    
    # Inițializează masca
    H = np.ones((M, N))
    
    # Iterează prin coordonatele detectate
    for x_k, y_k in coords:
        # Distanța față de vârful (k)
        D_k = np.sqrt((U - y_k) ** 2 + (V - x_k) ** 2)
        
        # Coordonatele simetrice
        x_sym = 2 * cx_centru - x_k
        y_sym = 2 * cy_centru - y_k
        
        # Distanța față de vârful simetric
        D_neg_k = np.sqrt((U - y_sym) ** 2 + (V - x_sym) ** 2)
        
        # Filtru Butterworth Notch
        H_k = 1 / (1 + (rad / (D_k + 1e-10)) ** (2 * n_order))
        H_neg_k = 1 / (1 + (rad / (D_neg_k + 1e-10)) ** (2 * n_order))
        
        H = H * H_k * H_neg_k
    
    # Aplicare filtru
    Ffilt = G * H
    img_restored = np.real(np.fft.ifft2(np.fft.ifftshift(Ffilt)))
    img_restored = np.clip(img_restored, 0, 1)
    
    return (img_restored * 255).astype(np.uint8)


def motion_kernel(length, angle):
    """
    Creează un kernel de motion blur
    """
    if length == 0:
        return np.array([[1]])
    
    angle_rad = np.deg2rad(angle)
    half = length // 2
    kernel = np.zeros((length, length))
    
    for i in range(length):
        x = int(round((i - half) * np.cos(angle_rad) + half))
        y = int(round((i - half) * np.sin(angle_rad) + half))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    
    kernel = kernel / (np.sum(kernel) + 1e-10)
    return kernel


def gaussian_kernel(size, sigma):
    """
    Creează un kernel Gaussian
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)
