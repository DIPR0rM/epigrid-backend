import os
import asyncio
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from services.sentinel_client import fetch_sentinel2_waterlogging_scenes
from services.nasa_power_client import fetch_nasa_meteorological_matrix
from services.dghs_client import fetch_dghs_dengue_baseline
from services.spatiotemporal_pipeline import execute_spatiotemporal_pipeline
from services.st_gnn_engine import execute_phase4_stgnn_training_and_forecast
from services.alert_dispatcher import execute_phase5_command_and_alerting, read_alert_ledger

app = FastAPI(
    title="EpiGrid: Epidemiological Preemption Engine",
    description=(
        "Complete 5-Phase Architecture: Sentinel-2 MNDWI Segmentation, NASA POWER Brière "
        "Thermodynamics, PyTorch ST-GNN 14-Day Outbreak Forecasting, Web Geospatial Heat Map "
        "Dashboard, and Automated Municipal Preemption Alerting."
    ),
    version="5.0.0"
)


@app.get("/")
async def root():
    return {
        "project": "EpiGrid: Epidemiological Preemption Engine",
        "phase": "Phase 5: Output Interface and Automated Alerting (All 5 Phases Active)",
        "interactive_geospatial_dashboard_url": "/dashboard",
        "swagger_api_documentation_url": "/docs",
        "timestamp_utc": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    await asyncio.sleep(0.05)
    return {"server": "Uvicorn ASGI", "healthy": True, "phase": 5}


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_geospatial_dashboard():
    """
    Phase 5 Output Interface: Serves the interactive Web-Based Geographic Heat Map
    and Municipal Preemption Command Center Dashboard.
    """
    template_path = os.path.join("templates", "dashboard.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


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
    """Phase 2 Pipeline: Concurrently executes all three multi-modal data acquisition streams."""
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
    """Phase 4 Pipeline: Trains the PyTorch ST-GNN and outputs the 14-day ward probability matrix."""
    return await execute_phase4_stgnn_training_and_forecast(lat=lat, lon=lon, epochs=epochs)


@app.get("/api/v1/alerts/evaluate-and-dispatch")
async def evaluate_and_dispatch_municipal_alerts(
    lat: float = Query(23.8103, description="Target Urban Latitude (Default: Dhaka)"),
    lon: float = Query(90.4125, description="Target Urban Longitude (Default: Dhaka)"),
    epochs: int = Query(120, ge=20, le=500, description="PyTorch ST-GNN training epochs"),
    alert_threshold: float = Query(0.55, ge=0.10, le=0.95, description="Outbreak probability threshold for automated municipal dispatch")
):
    """
    Phase 5 Pipeline: Executes the full ST-GNN forecast, evaluates ward trajectories against
    preemptive thresholds, computes larvicidal/fogging resource requirements, and dispatches
    cryptographically signed alerts to DNCC/DSCC municipal authorities.
    """
    return await execute_phase5_command_and_alerting(
        lat=lat,
        lon=lon,
        epochs=epochs,
        alert_threshold=alert_threshold
    )


@app.get("/api/v1/alerts/ledger")
async def get_municipal_alert_ledger():
    """Returns the persistent SHA-256 audit ledger of all triggered municipal preemption alerts."""
    records = read_alert_ledger()
    return {
        "total_logged_dispatches": len(records),
        "ledger_entries": records
    }
