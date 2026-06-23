from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehiclePredictor
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the vehicle-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.brand: Optional[str] = None
        self.model: Optional[str] = None
        self.model_year: Optional[int] = None
        self.milage: Optional[str] = None
        self.fuel_type: Optional[str] = None
        self.engine: Optional[str] = None
        self.transmission: Optional[str] = None
        self.ext_col: Optional[str] = None
        self.int_col: Optional[str] = None
        self.accident: Optional[str] = None
        self.clean_title: Optional[str] = None

    async def get_vehicle_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.brand = form.get("brand")
        self.model = form.get("model")
        self.model_year = form.get("model_year")
        self.milage = form.get("milage")
        self.fuel_type = form.get("fuel_type")
        self.engine = form.get("engine")
        self.transmission = form.get("transmission")
        self.ext_col = form.get("ext_col")
        self.int_col = form.get("int_col")
        self.accident = form.get("accident")
        self.clean_title = form.get("clean_title")

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
        request=request, name="index.html", context={"context": "Rendering"}
    )

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_vehicle_data()
        
        vehicle_data = VehicleData(
            brand=form.brand,
            model=form.model,
            model_year=form.model_year,
            milage=form.milage,
            fuel_type=form.fuel_type,
            engine=form.engine,
            transmission=form.transmission,
            ext_col=form.ext_col,
            int_col=form.int_col,
            accident=form.accident,
            clean_title=form.clean_title
        )

        # Convert form data into a DataFrame for the model
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = VehiclePredictor()

        # Make a prediction and retrieve the result
        predicted_price = model_predictor.predict(dataframe=vehicle_df)

        # Format predicted price as USD currency
        status = f"${predicted_price:,.2f}"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"context": status},
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"context": f"Error: {e}"},
        )

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
