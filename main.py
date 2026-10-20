from fastapi import FastAPI, Query
import asyncio
from datetime import datetime

from services.sentinel_client import fetch_sentinel2_waterlogging_scenes
from services.nasa_power_client import fetch_nasa_meteorological_matrix
from services.dghs_client import fetch_dghs_dengue_baseline
from services.spatiotemporal_pipeline import execute_spatiotemporal_pipeline
from services.st_gnn_engine import execute_phase4_stgnn_training_and_forecast

app = FastAPI(
    title="EpiGrid: Epidemiological Preemption Engine",
    description="Phase 4: Spatio-Temporal Graph Neural Network (PyTorch ST-GNN) Outbreak Forecasting Engine",
    version="4.0.0"
)

@app.get("/")
async def root():
    return {
        "project": "EpiGrid",
        "phase": "Phase 4: Predictive Neural Network Training (PyTorch ST-GNN) Active",
        "timestamp_utc": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    await asyncio.sleep(0.05)
    return {"server": "Uvicorn ASGI", "healthy": True, "phase": 4}

@app.get("/api/v1/ingest/sentinel2")
async def ingest_sentinel2(
    lat: float = Query(23.8103, description="Target Latitude (Default: Dhaka)"),
    lon: float = Query(90.4125, description="Target Longitude (Default: Dhaka)"),
    days_back: int = Query(30, description="Lookback window in days")
):
    return await fetch_sentinel2_waterlogging_scenes(lat, lon, days_back)

@app.get("/api/v1/ingest/nasa-power")
async def ingest_nasa_power(
    lat: float = Query(23.8103, description="Target Latitude"),
    lon: float = Query(90.4125, description="Target Longitude"),
    days_back: int = Query(14, description="Lookback window in days")
):
    return await fetch_nasa_meteorological_matrix(lat, lon, days_back)

@app.get("/api/v1/ingest/dghs-dengue")
async def ingest_dghs():
    return await fetch_dghs_dengue_baseline()

@app.get("/api/v1/ingest/multimodal-sync")
async def ingest_all_streams(
    lat: float = Query(23.8103, description="Target Latitude"),
    lon: float = Query(90.4125, description="Target Longitude")
):
    """Concurrently executes all three Phase 2 data acquisition streams."""
    sentinel_data, nasa_data, dghs_data = await asyncio.gather(
        fetch_sentinel2_waterlogging_scenes(lat, lon, days_back=30),
        fetch_nasa_meteorological_matrix(lat, lon, days_back=14),
        fetch_dghs_dengue_baseline()
    )
    return {
        "synchronization_timestamp_utc": datetime.utcnow().isoformat(),
        "target_coordinates": {"latitude": lat, "longitude": lon},
        "streams": {
            "copernicus_sentinel2": sentinel_data,
            "nasa_power_meteorology": nasa_data,
            "dghs_epidemiological_baseline": dghs_data
        }
    }

@app.get("/api/v1/pipeline/spatiotemporal-matrix")
async def run_spatiotemporal_pipeline(
    lat: float = Query(23.8103, description="Target Urban Latitude (Default: Dhaka)"),
    lon: float = Query(90.4125, description="Target Urban Longitude (Default: Dhaka)"),
    days_back: int = Query(14, description="Meteorological incubation window in days")
):
    """Phase 3 Pipeline: Isolates urban micro-waterlogging nodes and computes Aedes incubation kinetics."""
    return await execute_spatiotemporal_pipeline(lat, lon, days_back)

@app.get("/api/v1/ai/train-and-predict-stgnn")
async def run_stgnn_training_and_prediction(
    lat: float = Query(23.8103, description="Target Urban Latitude (Default: Dhaka)"),
    lon: float = Query(90.4125, description="Target Urban Longitude (Default: Dhaka)"),
    epochs: int = Query(120, ge=20, le=500, description="PyTorch backpropagation training epochs")
):
    """
    Phase 4 Pipeline: Constructs a Spatio-Temporal Graph Neural Network (ST-GNN) in PyTorch,
    trains on cross-referenced Sentinel-2, NASA POWER, and DGHS baseline tensors, and outputs
    a 14-day predictive outbreak probability matrix across urban municipal wards.
    """
    return await execute_phase4_stgnn_training_and_forecast(lat=lat, lon=lon, epochs=epochs)
