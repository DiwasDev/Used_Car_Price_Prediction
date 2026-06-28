# Used Car & Logistics ML Pipeline

An end-to-end orchestrated machine learning pipeline and web application designed for predicting used car prices (`price_usd`). The project features step-by-step preprocessing, robust data validation, modular modeling strategies, and cloud integration with MongoDB and Azure Blob Storage.

---

## 🚀 Features

- **End-to-End Orchestrated Pipeline**: Run training pipelines that ingest, split, validate, transform, and evaluate models.
- **FastAPI Web Application**: A beautiful and responsive web interface to predict vehicle prices in real time.
- **Robust Preprocessing**: Custom transformers for target encoding, handling outliers, scaling features, and missing value imputation.
- **Hybrid Storage Setup**: Ingests data from MongoDB and logs metrics/saves models in Azure Blob Storage.
- **Fully Dockerized**: Easily package and deploy the entire solution using Docker.

---

## 🛠️ Tech Stack

- **Backend / Web Framework**: Python, FastAPI, Uvicorn, Jinja2, Pydantic
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Category Encoders, Joblib, Dill
- **Databases & Cloud**: MongoDB (PyMongo), Azure Blob Storage
- **Infrastructure**: Docker, GitHub Actions Runner

---

## 📂 Project Structure

```text
├── src/
│   ├── components/            # Pipeline components (Ingestion, Validation, Transformation, etc.)
│   │   ├── data_transformers/ # Custom transformers (scaler, encoder, outliers, etc.)
│   │   └── modeler/           # Model building & evaluation strategies
│   ├── configuration/         # Database and Cloud Storage connections
│   ├── constants/             # Centralized pipeline constants and configurations
│   ├── entity/                # Data artifacts and configuration entities
│   ├── pipline/               # Orchestrated training and prediction pipelines
│   └── utils/                 # General helper utilities
├── steps/                     # Executable steps representing stages in the pipeline
├── templates/                 # HTML templates for the FastAPI UI
├── app.py                     # Entry point for the FastAPI web server
├── test.py                    # Entry point to trigger the orchestrator training pipeline
├── requirements.txt           # Cleaned and minimized project dependencies
├── Dockerfile                 # Configuration to containerize the application
└── setup.py                   # Setup script to install the project in editable mode
```

---

## ⚙️ Installation & Local Setup

### 1. Clone and Navigate to the Repository
```bash
git clone <repository_url>
cd logistics_project
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Installs only the necessary libraries and the `src` directory in editable mode:
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Make sure to configure the required environment variables:
```bash
export MONGODB_URL="your_mongodb_connection_string"
export AZURE_STORAGE_CONNECTION_STRING="your_azure_storage_connection_string"
```

---

## 🖥️ Running the Application

### Train the ML Pipeline
To ingest data from MongoDB, split it, perform validation, run the transformations, build the model, and evaluate:
```bash
python test.py
```

### Start the FastAPI Web Server
To launch the interactive web interface on `http://localhost:5000`:
```bash
python app.py
```

---

## 🐳 Docker Deployment

### Build the Docker Image
```bash
docker build -t logistics-ml-pipeline .
```

### Run the Container
```bash
docker run -p 5000:5000 \
  -e MONGODB_URL="your_mongodb_connection_string" \
  -e AZURE_STORAGE_CONNECTION_STRING="your_azure_storage_connection_string" \
  logistics-ml-pipeline
```
