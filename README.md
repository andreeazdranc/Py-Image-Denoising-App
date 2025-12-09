# Py-Image-Denoising-App
# 🖼️ Smart Image Restoration & Noise Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-yellow)

A full-stack web application designed to automatically detect noise patterns in digital images and apply optimal restoration filters. The project combines **Digital Signal Processing (DSP)** theories with a modern web interface.

## 🌟 Key Features

* **Automated Noise Detection:** Analyzes statistical moments (Skewness, Kurtosis, Entropy) and frequency domain (FFT) to classify noise types.
* **Adaptive Filtering:** Automatically selects the best restoration method (e.g., Wiener, Median, Gaussian, Notch Filter) based on the detected degradation.
* **Image Inpainting:** Capable of removing scratches, scribbles, and artifacts from deteriorated photos (as seen in the demo).
* **Performance Metrics:** Calculates PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and MSE to evaluate restoration quality.
* **User-Friendly Interface:** Simple Drag & Drop functionality for processing images.

## 📊 Before & After Demo

| Deteriorated Input | Restored Output |
|:---:|:---:|
| <img src="img_deteriorata1.png" width="300" alt="Scratched Input"> | <img src="img_restaurata1.png" width="300" alt="Restored Output"> |
| *Image with heavy artifacts* | *Restored using inpainting* |

| <img src="img_deteriorata2.jpg" width="300" alt="Noisy Input"> | <img src="img_restaurata2.jpg" width="300" alt="Restored Output"> |
| *Image with scribbles/noise* | *Restored Result* |

## 🛠️ Tech Stack

### Backend
* **Python 3.x**: Core logic.
* **Flask**: REST API for handling image uploads and processing requests.
* **OpenCV & NumPy**: Matrix operations, filtering, and image transformations.
* **SciPy**: Advanced signal processing calculations.

### Frontend
* **HTML5 / CSS3**: Responsive styling.
* **JavaScript (Vanilla)**: DOM manipulation and API communication.

## ⚙️ Algorithms Implemented

The application implements logic derived from classical Digital Image Processing literature (e.g., Gonzalez & Woods):

1.  **Noise Classification**:
    * **Gaussian**: Detected via symmetry of the histogram (Kurtosis ≈ 3).
    * **Salt & Pepper**: Detected via high density of min/max pixel values.
    * **Periodic**: Detected using Fourier Transform (FFT) spectral spikes.
    * **Uniform/Rayleigh/Erlang**: Identified via specific statistical distributions.

2.  **Restoration Filters**:
    * Arithmetic & Geometric Mean Filters.
    * Adaptive Median Filter (for Salt & Pepper).
    * Wiener Filter (for Gaussian/Linear noise).
    * Notch Filter (for Periodic noise removal in frequency domain).

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Pip (Python Package Manager)

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/username/Smart-Image-Restoration.git](https://github.com/username/Smart-Image-Restoration.git)
    cd Smart-Image-Restoration
    ```

2.  **Install Backend Dependencies**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
    *If you encounter issues, install manually:* `pip install Flask Flask-CORS numpy scipy opencv-python Pillow`.

3.  **Run the Server**
    ```bash
    # Windows
    python app.py
    ```
    The backend will start at `http://localhost:5000`.

4.  **Launch Frontend**
    * Simply open `frontend/index.html` in your browser.
    * Or serve it using Python:
        ```bash
        cd frontend
        python -m http.server 8000
        ```

## 📂 Project Structure
