"""
Detectie automată a tipului de zgomot
Portat din MATLAB în Python (detecteaza_zgomot_literatura.m)
Bazat pe criterii din Gonzalez & Woods
"""
import numpy as np
from scipy.stats import skew, kurtosis


def detecteaza_zgomot_literatura(img_orig, noisy_img):
    """
    Detectează tipul de zgomot aplicat pe imagine
    
    Args:
        img_orig: imaginea originală (numpy array, uint8)
        noisy_img: imaginea cu zgomot (numpy array, uint8)
    
    Returns:
        tip_zgomot: string cu tipul detectat
    """
    # Calculează diferența
    diff = noisy_img.astype(np.float64) - img_orig.astype(np.float64)
    
    # Statistici
    s = skew(diff.flatten())
    k = kurtosis(diff.flatten(), fisher=False)  # Pearson's kurtosis
    mean_val = np.mean(diff)
    
    # Normalizare la [0,1]
    noisy_norm = noisy_img.astype(np.float64) / 255.0
    
    # Procentaj sare și piper
    pct_sare = np.mean(noisy_norm == 0)
    pct_piper = np.mean(noisy_norm == 1)
    
    print(f"Skewness: {s:.3f}, Kurtosis: {k:.3f}, Mean: {mean_val:.3f}")
    print(f"Pct_sare: {pct_sare:.3f}, Pct_piper: {pct_piper:.3f}")
    
    # --- SARE & PIPER: peakuri extremități ---
    SARE_PRAG = 0.01
    PIPER_PRAG = 0.01
    if (pct_sare + pct_piper > 0.28) and (pct_sare > SARE_PRAG and pct_piper > PIPER_PRAG):
        print('Zgomot sare și piper detectat!')
        return 'sarepiper'
    
    # --- GAUSSIAN: simetric, kurtosis ≈ 3, skewness ≈ 0 ---
    if abs(s) < 0.15 and abs(k - 3) < 0.25:
        print('Zgomot gaussian detectat!')
        return 'gaussian'
    
    # --- RAYLEIGH: histogramă skewed ---
    if s > 0.3 and s < 1.5 and k > 1.8 and k < 4:
        print('Zgomot rayleigh detectat!')
        return 'rayleigh'
    
    # --- ERLANG: skewness mare, kurtosis între 3.5 și 6 ---
    if s > 1.0 and k > 4.5 and k <= 6 and mean_val > 42:
        print('Zgomot erlang detectat!')
        return 'erlang'
    
    # --- EXPONENTIAL: skewness și kurtosis mari ---
    if s > 1.2 and s < 1.8 and k > 4.8 and k < 6.8:
        print('Zgomot exponential detectat!')
        return 'exponential'
    
    # --- LINIAR INVARIANT: kurtosis foarte mare ---
    if k > 10:
        print('Zgomot liniar invariant detectat!')
        return 'liniar invariant'
    
    # --- PERIODIC: concentrare mare energie în FFT ---
    F = np.abs(np.fft.fftshift(np.fft.fft2(noisy_norm)))
    M, N = F.shape
    F_temp = F.copy()
    
    zona_exclusa = 10
    cy = M // 2
    cx = N // 2
    
    y_start = max(0, cy - zona_exclusa)
    y_end = min(M, cy + zona_exclusa)
    x_start = max(0, cx - zona_exclusa)
    x_end = min(N, cx + zona_exclusa)
    
    F_temp[y_start:y_end, x_start:x_end] = 0
    
    # Energie totală
    energie_totala = np.sum(F_temp**2) + 1e-10
    
    # Top 20 vârfuri
    numar_varfuri_de_test = 20
    top_valori = np.partition(F_temp.flatten(), -numar_varfuri_de_test)[-numar_varfuri_de_test:]
    energie_varfuri = np.sum(top_valori**2)
    
    concentrare_energie = energie_varfuri / energie_totala
    print(f"[DEBUG Periodic] Concentrare Energie: {concentrare_energie * 100:.2f}%")
    
    if concentrare_energie > 0.25:
        print('Zgomot periodic detectat!')
        return 'periodic'
    
    # --- UNIFORM: entropie mare ---
    hist, _ = np.histogram(diff.flatten(), bins=256, density=True)
    h_nonzero = hist[hist > 0]
    
    if len(h_nonzero) > 0:
        entropie = -np.sum(h_nonzero * np.log2(h_nonzero))
        print(f"[DEBUG Uniform] Entropie: {entropie:.2f}")
        
        if entropie > 6.5:
            print('Zgomot uniform detectat!')
            return 'uniform'
    
    print('Nu s-a putut detecta zgomotul cu precizie.')
    return 'neclasificat'


def detect_periodic_coords(noisy_img, freq_x=30, freq_y=42):
    """
    Detectează coordonatele vârfurilor pentru zgomotul periodic
    
    Returns:
        coords: listă de tuple [(x1, y1), (x2, y2), ...]
    """
    noisy_norm = noisy_img.astype(np.float64) / 255.0
    F = np.fft.fftshift(np.fft.fft2(noisy_norm))
    M, N = F.shape
    
    cx = N // 2
    cy = M // 2
    
    # Coordonate bazate pe frecvențele cunoscute
    coords = [
        (cx + freq_x, cy + freq_y),  # Colț Sus-Dreapta
        (cx + freq_x, cy - freq_y),  # Colț Jos-Dreapta
        (cx - freq_x, cy + freq_y),  # Colț Sus-Stânga
        (cx - freq_x, cy - freq_y),  # Colț Jos-Stânga
        (cx + freq_x, cy),            # Mijloc Dreapta
        (cx - freq_x, cy),            # Mijloc Stânga
        (cx, cy + freq_y),            # Mijloc Sus
        (cx, cy - freq_y)             # Mijloc Jos
    ]
    
    return coords
