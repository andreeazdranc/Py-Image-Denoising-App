"""
Detectia automată a filtrului optimal
Portat din MATLAB în Python (detecteaza_filtru_optimal.m)
"""
import numpy as np
from filter_functions import (
    mean_filter, median_filter, adaptive_median_filter,
    geometric_mean_filter, harmonic_mean_filter,
    wiener_filter_img, inverse_filter_img, notch_filter_img,
    motion_kernel, gaussian_kernel
)
from metrics_functions import calculate_mse, calculate_psnr, calculate_ssim
from detect_noise_type import detect_periodic_coords


def detecteaza_filtru_optimal(img_orig, noisy_img, tip_zgomot, 
                               coords=None, rad=7, 
                               psf_tip='motion', psf_param1=7, psf_param2=90):
    """
    Detectează și aplică filtrul optimal pentru tipul de zgomot detectat
    
    Args:
        img_orig: imaginea originală
        noisy_img: imaginea cu zgomot
        tip_zgomot: tipul de zgomot detectat
        coords: coordonatele pentru filtrul notch (periodic)
        rad: raza pentru filtrul notch
        psf_tip: tipul PSF pentru liniar invariant
        psf_param1, psf_param2: parametri PSF
    
    Returns:
        best_img: imaginea restaurată
        best_filter: numele filtrului optimal
        rezultate: dicționar cu metricile pentru fiecare filtru
    """
    print(f"Detectare filtru optimal pentru zgomot: {tip_zgomot}")
    
    # Definire filtre candidate în funcție de tip zgomot
    if tip_zgomot == 'gaussian':
        filtre = {
            'mean_filter': lambda img: mean_filter(img, 3),
            'wiener_filter': lambda img: wiener_filter_img(img, None, 0.1)
        }
    
    elif tip_zgomot in ['rayleigh', 'erlang', 'exponential', 'uniform']:
        filtre = {
            'mean_filter': lambda img: mean_filter(img, 3),
            'geometric_mean_filter': lambda img: geometric_mean_filter(img, 3),
            'harmonic_mean_filter': lambda img: harmonic_mean_filter(img, 3)
        }
    
    elif tip_zgomot == 'sarepiper':
        filtre = {
            'median_filter': lambda img: median_filter(img, 3),
            'adaptive_median_filter': lambda img: adaptive_median_filter(img, 7)
        }
    
    elif tip_zgomot == 'periodic':
        if coords is None:
            coords = detect_periodic_coords(noisy_img)
        filtre = {
            'notch_filter': lambda img: notch_filter_img(img, coords, rad, 4)
        }
    
    elif tip_zgomot == 'liniar invariant':
        PSF = motion_kernel(psf_param1, psf_param2) if psf_tip == 'motion' \
              else gaussian_kernel(psf_param1, psf_param2)
        
        filtre = {
            'inverse_filter': lambda img: inverse_filter_img(img, PSF),
            'wiener_filter': lambda img: wiener_filter_img(img, PSF, 0.1)
        }
    
    else:
        print(f"Zgomot necunoscut: {tip_zgomot}")
        return noisy_img, 'necunoscut', {}
    
    # Calculează metricile pentru imaginea zgomotoasă
    psnr_noisy = calculate_psnr(noisy_img, img_orig)
    ssim_noisy = calculate_ssim(noisy_img, img_orig)
    mse_noisy = calculate_mse(noisy_img, img_orig)
    
    print(f"\nImagine zgomotoasă - PSNR: {psnr_noisy:.2f}, SSIM: {ssim_noisy:.3f}, MSE: {mse_noisy:.4f}")
    
    # Testează fiecare filtru
    rezultate = {}
    
    for nume_filtru, filtru_func in filtre.items():
        try:
            print(f"\nAplicare filtru: {nume_filtru}...")
            img_filtrata = filtru_func(noisy_img)
            
            psnr = calculate_psnr(img_filtrata, img_orig)
            ssim = calculate_ssim(img_filtrata, img_orig)
            mse = calculate_mse(img_filtrata, img_orig)
            
            rezultate[nume_filtru] = {
                'psnr': psnr,
                'ssim': ssim,
                'mse': mse,
                'img': img_filtrata
            }
            
            print(f"{nume_filtru} - PSNR: {psnr:.2f}, SSIM: {ssim:.3f}, MSE: {mse:.4f}")
            
        except Exception as e:
            print(f"Eroare la aplicarea filtrului {nume_filtru}: {e}")
            rezultate[nume_filtru] = {
                'psnr': 0,
                'ssim': 0,
                'mse': float('inf'),
                'img': noisy_img
            }
    
    # Găsește filtrul cu cele mai bune rezultate
    # Scor combinat: maximizează PSNR + 40*SSIM, minimizează MSE
    best_filter = None
    best_score = -float('inf')
    best_img = noisy_img
    
    for nume_filtru, metrici in rezultate.items():
        scor = metrici['psnr'] + 40 * metrici['ssim'] - 0.001 * metrici['mse']
        if scor > best_score:
            best_score = scor
            best_filter = nume_filtru
            best_img = metrici['img']
    
    if best_filter:
        print(f"\n{'='*60}")
        print(f"Filtrul optimal: {best_filter}")
        print(f"PSNR: {rezultate[best_filter]['psnr']:.2f}")
        print(f"SSIM: {rezultate[best_filter]['ssim']:.3f}")
        print(f"MSE: {rezultate[best_filter]['mse']:.4f}")
        print(f"{'='*60}\n")
    
    return best_img, best_filter, rezultate
