"""
Funcții pentru aplicarea zgomotelor pe imagini
Portate din MATLAB în Python/NumPy
"""
import numpy as np
from scipy.ndimage import convolve


def zgomot_gaussian(img, mu=0, sigma=0.1):
    """
    Aplică zgomot Gaussian
    img: numpy array (0-255, uint8)
    mu: media zgomotului
    sigma: deviația standard
    """
    img_float = img.astype(np.float64) / 255.0
    noise = sigma * np.random.randn(*img.shape) + mu
    img_noisy = img_float + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    return (img_noisy * 255).astype(np.uint8)


def zgomot_sare_piper(img, densitate=0.05):
    """
    Aplică zgomot sare și piper
    img: numpy array (0-255, uint8)
    densitate: proporția de pixeli afectați
    """
    img_noisy = img.copy()
    # Sare (alb)
    num_salt = int(densitate * img.size * 0.5)
    coords_salt = [np.random.randint(0, i, num_salt) for i in img.shape]
    img_noisy[tuple(coords_salt)] = 255
    
    # Piper (negru)
    num_pepper = int(densitate * img.size * 0.5)
    coords_pepper = [np.random.randint(0, i, num_pepper) for i in img.shape]
    img_noisy[tuple(coords_pepper)] = 0
    
    return img_noisy


def zgomot_periodic(img, amplitudine=0.19, freq_x=30, freq_y=42, faza=0):
    """
    Aplică zgomot periodic (sinusoidal)
    """
    img_float = img.astype(np.float64) / 255.0
    rows, cols = img.shape
    
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    noise = amplitudine * (
        np.sin(2 * np.pi * freq_x * X / cols + faza) +
        np.sin(2 * np.pi * freq_y * Y / rows + faza)
    )
    
    img_noisy = img_float + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    return (img_noisy * 255).astype(np.uint8)


def zgomot_rayleigh(img, a=0.4):
    """
    Aplică zgomot Rayleigh
    a: parametrul de scală
    """
    img_float = img.astype(np.float64) / 255.0
    U = np.random.rand(*img.shape)
    noise = a * np.sqrt(-2 * np.log(U))
    img_noisy = img_float + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    return (img_noisy * 255).astype(np.uint8)


def zgomot_uniform(img, a=-40/255, b=40/255):
    """
    Aplică zgomot uniform pe intervalul [a, b]
    """
    img_float = img.astype(np.float64) / 255.0
    noise = (b - a) * np.random.rand(*img.shape) + a
    img_noisy = img_float + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    return (img_noisy * 255).astype(np.uint8)


def zgomot_exponential(img, mu=0.2):
    """
    Aplică zgomot exponențial
    mu: parametrul lambda (rata)
    """
    img_float = img.astype(np.float64) / 255.0
    noise = np.random.exponential(mu, img.shape)
    img_noisy = img_float + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    return (img_noisy * 255).astype(np.uint8)


def zgomot_erlang(img, k=2, lambda_param=10):
    """
    Aplică zgomot Erlang (Gamma cu k întreg)
    k: parametrul de formă
    lambda_param: parametrul de rată
    """
    img_float = img.astype(np.float64) / 255.0
    noise = np.random.gamma(k, 1/lambda_param, img.shape)
    img_noisy = img_float + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    return (img_noisy * 255).astype(np.uint8)


def zgomot_liniar_invariant(img, psf_tip='motion', psf_param1=7, psf_param2=90):
    """
    Aplică degradare liniară invariantă la poziție (blur)
    psf_tip: 'motion' sau 'gaussian'
    psf_param1, psf_param2: parametri PSF
        - 'motion': lungime, unghi (grade)
        - 'gaussian': mărime kernel, sigma
    """
    if psf_tip == 'motion':
        PSF = motion_kernel(psf_param1, psf_param2)
    elif psf_tip == 'gaussian':
        PSF = gaussian_kernel(psf_param1, psf_param2)
    else:
        raise ValueError("Tip PSF necunoscut!")
    
    img_noisy = convolve(img, PSF, mode='constant')
    return np.clip(img_noisy, 0, 255).astype(np.uint8)


def motion_kernel(length, angle):
    """
    Creează un kernel de motion blur
    """
    if length == 0:
        return np.array([[1]])
    
    # Convertește unghiul în radiani
    angle_rad = np.deg2rad(angle)
    
    # Creează kernel-ul
    half = length // 2
    kernel = np.zeros((length, length))
    
    # Linia de mișcare
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
