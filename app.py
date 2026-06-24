import os
import json
from typing import List

from fastapi import FastAPI, Request, HTTPException, status, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from uvicorn import run as app_run

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehiclePredictor, sync_latest_artifact
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# DEPENDENCIES & SCHEMAS (Refactored Layer)
# ==========================================

def get_brand_list() -> List[str]:
    """
    Dependency helper to extract all valid brands from the latest inference_meta.json.
    Falls back to a default list upon failure.
    """
    try:
        sync_latest_artifact()
        meta_path = os.path.join("artifact", "latest_artifact", "data_transformation", "transformation_object", "inference_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return sorted([b for b in meta.get("target_encodings", {}).get("brand", {}).keys() if b != "__global_fallback__"])
    except Exception as e:
        print(f"Error loading brands: {e}")
    return ["BMW", "Ford", "Toyota", "Lexus", "Audi", "Subaru", "Chevrolet"]


class VehicleForm:
    """
    Standard Python class dependency for parsing HTML Form data.
    FastAPI will automatically inject the form payload into these parameters.
    """
    def __init__(
        self,
        brand: str = Form(...), 
        model_year: int = Form(...),
        mileage_num: float = Form(...),
        engine_hp: float = Form(...),
        engine_displacement: float = Form(...),
        fuel_type: str = Form(...),
        transmission_type: str = Form(...),
        ext_col: str = Form(...),
        int_col: str = Form(...),
        accident: int = Form(0),
        clean_title: int = Form(1),
        is_sport: int = Form(0),
        is_premium: int = Form(0),
        is_4WD_AWD: int = Form(0)
    ):
        self.brand = brand
        self.model_year = model_year
        self.mileage_num = mileage_num
        self.engine_hp = engine_hp
        self.engine_displacement = engine_displacement
        self.fuel_type = fuel_type
        self.transmission_type = transmission_type
        self.ext_col = ext_col
        self.int_col = int_col
        self.accident = accident
        self.clean_title = clean_title
        self.is_sport = is_sport
        self.is_premium = is_premium
        self.is_4WD_AWD = is_4WD_AWD


# ==========================================
# ENDPOINTS / ROUTERS
# ==========================================

@app.get("/", tags=["authentication"])
async def index(request: Request, brands: List[str] = Depends(get_brand_list)):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"context": "Rendering", "brands": brands}
    )


@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    Properly utilizes HTTPException to send back valid server errors.
    # """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error Occurred During Training! {str(e)}"
        )


@app.post("/")
async def predictRouteClient(
    request: Request, 
    form: VehicleForm = Depends(), 
    brands: List[str] = Depends(get_brand_list)
):
    """
    Endpoint to receive form data automatically via Pydantic model injection,
    process calculations, and return visual results.
    """
    try:
        # Map structured fields into the raw text format expected by the preprocessing pipeline
        accident_str = "At least 1 accident or damage reported" if form.accident == 1 else "None reported"
        clean_title_str = "Yes" if form.clean_title == 1 else "No"
        
        # Build model keyword string for flag extraction
        model_keywords = []
        if form.is_sport == 1:
            model_keywords.append("Sport")
        if form.is_premium == 1:
            model_keywords.append("Premium")
        if form.is_4WD_AWD == 1:
            model_keywords.append("4WD")
        model_name = " ".join(model_keywords) if model_keywords else "Base Model"
        
        milage_str = f"{form.mileage_num} mi."
        engine_str = f"{form.engine_hp} HP {form.engine_displacement}L"
        
        vehicle_data = VehicleData(
            brand=form.brand,
            model=model_name,
            model_year=form.model_year,
            milage=milage_str,
            fuel_type=form.fuel_type,
            engine=engine_str,
            transmission=form.transmission_type,
            ext_col=form.ext_col,
            int_col=form.int_col,
            accident=accident_str,
            clean_title=clean_title_str
        )

        # Convert form data into a DataFrame for the model
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        # Initialize the prediction pipeline and fetch output
        model_predictor = VehiclePredictor()
        predicted_price = model_predictor.predict(dataframe=vehicle_df)

        # Format predicted price as USD currency
        status_msg = f"${predicted_price:,.2f}"

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"context": status_msg, "brands": brands},
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"context": f"Error: {e}", "brands": brands},
        )


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)