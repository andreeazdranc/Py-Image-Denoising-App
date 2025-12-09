"""
Backend Flask pentru aplicația de restaurare imagini
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

from noise_functions import (
    zgomot_gaussian, zgomot_sare_piper, zgomot_periodic,
    zgomot_rayleigh, zgomot_uniform, zgomot_exponential,
    zgomot_erlang, zgomot_liniar_invariant
)
from detect_noise_type import detecteaza_zgomot_literatura
from detect_optimal_filter import detecteaza_filtru_optimal

app = Flask(__name__)
CORS(app)

# Configurare
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Creare foldere dacă nu există
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img_array):
    """Convertește numpy array în base64 string"""
    img_pil = Image.fromarray(img_array)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.route('/api/health', methods=['GET'])
def health_check():
    """Verificare că serverul funcționează"""
    return jsonify({'status': 'ok', 'message': 'Backend funcționează!'})


@app.route('/api/process', methods=['POST'])
def process_image():
    """
    Procesează imaginea: aplică zgomot, detectează zgomot, aplică filtru optimal
    """
    try:
        # Verifică dacă a fost trimisă o imagine
        if 'image' not in request.files:
            return jsonify({'error': 'Nicio imagine trimisă'}), 400
        
        file = request.files['image']
        noise_type = request.form.get('noise_type', 'gaussian')
        
        if file.filename == '':
            return jsonify({'error': 'Niciun fișier selectat'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Tip fișier invalid'}), 400
        
        # Salvează imaginea originală
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Citește imaginea în grayscale
        img_orig = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if img_orig is None:
            return jsonify({'error': 'Eroare la citirea imaginii'}), 400
        
        # PASUL 1: Aplică zgomotul
        print(f"\n{'='*60}")
        print(f"PASUL 1: Aplicare zgomot {noise_type}")
        print(f"{'='*60}")
        
        if noise_type == 'gaussian':
            noisy_img = zgomot_gaussian(img_orig, 0, 0.1)
        elif noise_type == 'sarepiper':
            noisy_img = zgomot_sare_piper(img_orig, 0.45)
        elif noise_type == 'periodic':
            noisy_img = zgomot_periodic(img_orig, 0.19, 30, 42, 0)
        elif noise_type == 'rayleigh':
            noisy_img = zgomot_rayleigh(img_orig, 0.4)
        elif noise_type == 'uniform':
            noisy_img = zgomot_uniform(img_orig, -40/255, 40/255)
        elif noise_type == 'exponential':
            noisy_img = zgomot_exponential(img_orig, 0.2)
        elif noise_type == 'erlang':
            noisy_img = zgomot_erlang(img_orig, 2, 10)
        elif noise_type == 'liniar_invariant':
            noisy_img = zgomot_liniar_invariant(img_orig, 'motion', 7, 90)
        else:
            return jsonify({'error': 'Tip zgomot invalid'}), 400
        
        # PASUL 2: Detectează tipul de zgomot
        print(f"\n{'='*60}")
        print(f"PASUL 2: Detectie automată zgomot")
        print(f"{'='*60}")
        
        tip_detectat = detecteaza_zgomot_literatura(img_orig, noisy_img)
        
        print(f"\nZgomot aplicat: {noise_type}")
        print(f"Zgomot detectat: {tip_detectat}")
        
        # PASUL 3: Aplică filtrul optimal
        print(f"\n{'='*60}")
        print(f"PASUL 3: Aplicare filtru optimal")
        print(f"{'='*60}")
        
        img_restored, best_filter, rezultate = detecteaza_filtru_optimal(
            img_orig, noisy_img, tip_detectat
        )
        
        # Convertește imaginile la base64 pentru frontend
        img_orig_b64 = image_to_base64(img_orig)
        noisy_img_b64 = image_to_base64(noisy_img)
        img_restored_b64 = image_to_base64(img_restored)
        
        # Pregătește metrici pentru răspuns
        metrici_finale = {}
        if best_filter and best_filter in rezultate:
            metrici_finale = {
                'psnr': float(rezultate[best_filter]['psnr']),
                'ssim': float(rezultate[best_filter]['ssim']),
                'mse': float(rezultate[best_filter]['mse'])
            }
        
        # Pregătește toate metricile
        toate_metricile = {}
        for filtru, metrici in rezultate.items():
            toate_metricile[filtru] = {
                'psnr': float(metrici['psnr']),
                'ssim': float(metrici['ssim']),
                'mse': float(metrici['mse'])
            }
        
        return jsonify({
            'success': True,
            'images': {
                'original': img_orig_b64,
                'noisy': noisy_img_b64,
                'restored': img_restored_b64
            },
            'noise_applied': noise_type,
            'noise_detected': tip_detectat,
            'filter_used': best_filter if best_filter else 'necunoscut',
            'metrics': metrici_finale,
            'all_metrics': toate_metricile
        })
        
    except Exception as e:
        print(f"Eroare în procesare: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/noise-types', methods=['GET'])
def get_noise_types():
    """Returnează lista de tipuri de zgomot disponibile"""
    return jsonify({
        'noise_types': [
            {'value': 'gaussian', 'label': 'Gaussian'},
            {'value': 'sarepiper', 'label': 'Sare și Piper'},
            {'value': 'periodic', 'label': 'Periodic'},
            {'value': 'rayleigh', 'label': 'Rayleigh'},
            {'value': 'uniform', 'label': 'Uniform'},
            {'value': 'exponential', 'label': 'Exponențial'},
            {'value': 'erlang', 'label': 'Erlang'},
            {'value': 'liniar_invariant', 'label': 'Liniar Invariant'}
        ]
    })


if __name__ == '__main__':
    print("Pornire server Flask pe http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
