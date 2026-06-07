# StockLens — Python Microservice

FastAPI + yfinance + pandas-ta + scikit-learn  
Datos financieros 100% gratuitos. Sin APIs de pago.

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/analyze/technical/{ticker}?period=1y` | OHLCV + indicadores + señales |
| GET | `/analyze/fundamental/{ticker}` | Métricas + Value Score |
| POST | `/predict/random-forest/{ticker}` | Predicción ML a N días |
| POST | `/predict/gradient-boosting/{ticker}` | Predicción ML alternativa |

Docs interactiva (Swagger): `https://tu-servicio.railway.app/docs`

---

## Dev local

```bash
# 1. Entorno virtual
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 2. Dependencias
pip install -r requirements.txt

# 3. Variables de entorno
cp .env.example .env
# (editar .env si quieres Supabase)

# 4. Arrancar
uvicorn main:app --reload --port 8000

# 5. Test rápido (en otra terminal)
python scripts/test_local.py
```

Swagger UI: http://localhost:8000/docs

---

## Deploy en Railway (RECOMENDADO — más fácil)

### Paso 1 — Crear cuenta Railway
- Ve a **https://railway.app**
- Clic en **"Start a New Project"**
- Regístrate con tu cuenta de **GitHub** (mismo que usas para Vercel)
- Plan gratuito: **5$ de crédito/mes** ≈ ~500 horas = suficiente para este servicio

### Paso 2 — Subir código a GitHub

```bash
# Dentro de la carpeta stocklens-python:
git init
git add .
git commit -m "feat: StockLens Python Microservice v1"
gh repo create stocklens-python --public --push --source=.
# O manualmente en github.com/new → subir carpeta
```

### Paso 3 — Crear el servicio en Railway

1. En Railway → **"New Project"** → **"Deploy from GitHub repo"**
2. Selecciona el repo `stocklens-python`
3. Railway detecta Python automáticamente (Nixpacks)
4. Clic en **"Deploy"** — espera ~3 minutos el primer build

### Paso 4 — Variables de entorno en Railway

En tu proyecto Railway → pestaña **"Variables"** → añade:

```
CORS_ORIGINS=https://tu-app.vercel.app,http://localhost:3000
```

(Supabase es opcional, déjalo vacío si no lo usas desde Python)

### Paso 5 — Obtener la URL pública

1. Railway → tu servicio → pestaña **"Settings"**
2. En **"Networking"** → clic **"Generate Domain"**
3. Te da algo como: `stocklens-python-production.up.railway.app`

### Paso 6 — Conectar con el frontend Next.js

En **Vercel** → tu proyecto → **Settings → Environment Variables**:

```
PYTHON_SERVICE_URL = https://stocklens-python-production.up.railway.app
```

Redeploy el frontend → ¡listo!

---

## Deploy en Render (alternativa gratuita)

### Paso 1 — Crear cuenta
- Ve a **https://render.com**
- **"Sign Up"** con GitHub
- Plan Free: servicios duermen tras 15min de inactividad (cold start ~30s)
  - Solución: el frontend maneja el fallback automáticamente

### Paso 2 — Nuevo Web Service

1. Dashboard → **"New +"** → **"Web Service"**
2. Conecta el repo `stocklens-python`
3. Configura:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120`

### Paso 3 — Variables de entorno en Render

En **Environment** → añade:
```
CORS_ORIGINS = https://tu-app.vercel.app,http://localhost:3000
```

### Paso 4 — URL y conexión

- Render te da: `https://stocklens-python.onrender.com`
- Añade en Vercel: `PYTHON_SERVICE_URL=https://stocklens-python.onrender.com`

---

## Resumen de cuentas necesarias

| Servicio | Para qué | Coste | URL registro |
|---------|---------|-------|-------------|
| **GitHub** | Código fuente | Gratis | github.com |
| **Vercel** | Frontend Next.js | Gratis (Hobby) | vercel.com |
| **Supabase** | Base de datos | Gratis (500MB) | supabase.com |
| **Railway** | Este microservicio Python | 5$/mes crédito gratis | railway.app |
| **Render** | Alternativa Railway | Gratis (con cold start) | render.com |

**Coste total: 0€**

---

## Orden de deploy recomendado

```
1. ✅ GitHub     → Sube AMBOS repos (Next.js + Python)
2. ✅ Supabase   → Ejecuta supabase_migration.sql
3. ✅ Railway    → Deploy del repo Python
                   Copia la URL pública
4. ✅ Vercel     → Deploy del repo Next.js
                   Añade las env vars (Supabase + Railway URL)
5. ✅ Test       → Abre tu dominio Vercel y busca AAPL
```

---

## Arquitectura de datos

```
Petición usuario (AAPL)
         │
    Next.js API Route
         │
    ┌────┴────────┐
    │ Supabase?   │ ← caché hit (< 1h) → respuesta inmediata
    └────┬────────┘
         │ miss
    ┌────┴──────────────┐
    │ Python Service    │ ← yfinance + pandas-ta
    │ (Railway)         │
    └────┬──────────────┘
         │ fallo / no configurado
    ┌────┴──────────────┐
    │ Yahoo Finance     │ ← API no oficial directa
    │ (desde Vercel)    │
    └────┬──────────────┘
         │ fallo
    ┌────┴──────────────┐
    │ Demo data         │ ← siempre disponible
    └───────────────────┘
```
