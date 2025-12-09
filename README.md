# Backend - Aplicație Restaurare Imagini

## Instalare dependințe

```bash
pip install -r requirements.txt
```

## Pornire server

```bash
python app.py
```

Serverul va rula pe `http://localhost:5000`

## Endpoints API

### GET /api/health
Verificare status server

### GET /api/noise-types
Returnează lista de zgomote disponibile

### POST /api/process
Procesează imaginea:
- Primește: `image` (file), `noise_type` (string)
- Returnează: imagini (original, zgomot, restaurată) + metrici
