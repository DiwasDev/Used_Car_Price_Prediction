# 🚗 Used Car Price Prediction — Production ML Pipeline

> An end-to-end machine learning system — not a notebook — that ingests, validates, trains, evaluates, and serves used car price predictions, built the way a real business would need it to run.



[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](#)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#) <!-- TODO: change if you pick a different license -->

**[Live Demo](#) <!-- TODO: add link if deployed --> · [Architecture](#architecture) · [Results](#model-performance) · [Quick Start](#quick-start)**

---

## Why This Exists

Most "used car price prediction" projects are a notebook with a Random Forest and an R² score, run once and never touched again. They answer "can this predict a price?" — not "could a business actually trust this in production?"

This project is built to answer the second question. This project implements software engineering principles to seamlessly integrate and deploy the changes to production.

The result is a system with proper data validation, swappable preprocessing strategies, automated model evaluation gating, and a deployable API — the infrastructure layer that separates a take-home assignment from something a business could actually run.

---

## Demo

## Demo

Demo URL: [http://car-price.ddns.net](http://car-price.ddns.net)

[![Watch the Demo video](https://img.youtube.com/vi/PLpQoS2NfBU/maxresdefault.jpg)](https://youtu.be/PLpQoS2NfBU)
---

## Architecture

```
┌─────────────┐     ┌───────────┐     ┌────────────────┐     ┌─────────────────┐
│  MongoDB /  │ --> │  Schema   │ --> │  Preprocessing  │ --> │  Model Training  │
│  CSV Source │     │ Validation│     │  (Strategy chain)│     │  (Linear Model)  │
└─────────────┘     └───────────┘     └────────────────┘     └────────┬─────────┘
                                                                       │
                                                                       ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│   FastAPI   │ <-- │  Azure Blob  │ <-- │ Model Pusher    │ <-- │  Champion vs  │
│   Web UI    │     │   Registry   │     │ (if approved)   │     │  Challenger   │
└─────────────┘     └──────────────┘     └────────────────┘     └──────────────┘
```

<!-- TODO: this ASCII version is a placeholder — swap in a real diagram image (draw.io, Excalidraw, or even a hand-drawn photo) for the actual top-1% effect. ASCII reads fine in a pinch, but a clean visual converts browsers into readers much better. -->

**Pipeline stages:**
1. **Ingestion** — pulls from MongoDB or CSV via a Template Method pattern (swap the source without touching the pipeline logic)
2. **Validation** — checks incoming data against `config/schema.yaml` before anything downstream runs
3. **Transformation** — a chain of Strategy-pattern components: cleaning → missing value imputation → outlier treatment → feature engineering → encoding → scaling
4. **Training** — fits the model, verifies it clears a minimum R² threshold before proceeding
5. **Evaluation** — champion-vs-challenger comparison against the model currently live in Azure
6. **Deployment** — only models that beat production get pushed; metadata + artifacts go to Azure Blob Storage
7. **Serving** — FastAPI app pulls the latest approved model and preprocessor for live predictions

---

## Key Engineering Decisions

This is the section that actually differentiates this project — lead with it.

| Pattern / Decision | Where | Why it matters |
|---|---|---|
| **Strategy Pattern** | `data_transformers/` (cleaning, missing values, outliers, encoding, scaling) | Swap any preprocessing step without touching the rest of the pipeline. New encoding scheme? One new class, zero refactors. |
| **Template Method** | `data_ingestion.py` | Source can change (CSV today, MongoDB tomorrow) without changing the ingestion contract. |
| **Factory Method** | `model_builder.py` | New model algorithms plug in without rewriting the training step. |
| **Champion-Challenger Evaluation** | `model_evaluation.py` | A new model only replaces production if it actually performs better — protects against silent accuracy regressions. |
| **Schema-Driven Validation** | `data_validation.py` + `schema.yaml` | Bad data gets caught before it reaches the model, not after a bad prediction ships. |


---

## Model Performance

| Metric | Value |
|---|---|
| R² | `0.747` |
| RMSE (Log Scale)| `0.435` |
| MAE (Log Scale)| `0.293` |
| Training set size | `3200` |
| Test set size | `800` |

**What this means in practice:** The predicted prices of cars are roughly off by 29% from the actual market value.
---

## Tech Stack

- **Modeling:** scikit-learn
- **Data Storage:** MongoDB (ingestion), Azure Blob Storage (model registry)
- **Serving:** FastAPI + Tailwind-styled web UI
- **Containerization:** Docker
- **Config:** PyYAML-driven schema and pipeline settings

---

## Project Structure

```
.
├── app.py                      # FastAPI web app
├── run_training.py             # Orchestrates the full training pipeline
├── run_deployment.py           # Pushes approved model + preprocessor to Azure
├── config/
│   ├── schema.yaml             # Column definitions, types, scaling targets
│   └── model.yaml
├── steps/                      # Pipeline orchestration steps
├── src/
│   ├── components/             # Core pipeline logic (ingestion, transformation, training, evaluation, pusher)
│   │   └── data_transformers/  # Strategy-pattern preprocessing components
│   ├── entity/                 # Config & artifact dataclasses, Azure estimator, model wrapper
│   ├── configuration/           # MongoDB / Azure connection setup
│   └── pipline/                # Training, deployment, and prediction pipeline sequencing
├── templates/                  # FastAPI web UI
├── static/                     # CSS
└── notebooks.ipynb             # EDA and visualization
```

---

## Quick Start

```bash
# clone and enter the repo
git clone https://github.com/DiwasDev/Used_Car_Price_Prediction.git && cd USED_CAR_PRICE_PREDICTION

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# set required environment variables
export MONGODB_URL="<your-mongo-connection-string>"
export AZURE_STORAGE_CONNECTION_STRING="<your-azure-connection-string>"

# run the training pipeline
python run_training.py

# launch the web UI
uvicorn app:app --reload
```

Visit `http://localhost:5000` to use the prediction interface. 

**Or run with Docker:**
```bash
docker build -t car-price-predictor .
docker run -p 5000:5000 car-price-predictor
```

---


## What I'd Build Next

1. Add gradient boosted models and compare via the existing champion-challenger flow.
2. Add drift monitoring on incoming prediction requests.

---

## About This Project

Hey, I'm Divas. Currently I'm upskilling in AI. I'm currently buiding production grade AI projects which can provide value to the business.

Linkedin: [Linkedin Profile](https://www.linkedin.com/in/diwas-pathak/)
---

## License

MIT LICENSE
