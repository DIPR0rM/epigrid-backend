import os
import asyncio
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from services.sentinel_client import fetch_sentinel2_waterlogging_scenes
from services.nasa_power_client import fetch_nasa_meteorological_matrix
from services.dghs_client import fetch_dghs_dengue_baseline
from services.spatiotemporal_pipeline import execute_spatiotemporal_pipeline
from services.st_gnn_engine import (
    execute_phase4_stgnn_training_and_forecast,
    get_full_bangladesh_hierarchy
)
from services.alert_dispatcher import (
    execute_phase5_command_and_alerting,
    read_alert_ledger
)

app = FastAPI(
    title="EpiGrid: Epidemiological Preemption Engine (All 64 Districts of Bangladesh)",
    description=(
        "Complete 5-Phase Calibrated Architecture across all 8 Divisions, 64 Districts, "
        "and 768 Municipal Wards of Bangladesh with 4-Tier Risk Classification."
    ),
    version="5.3.0"
)


@app.get("/")
async def root():
    return {
        "project": "EpiGrid: Epidemiological Preemption Engine",
        "phase": "Phase 5: All-Bangladesh 4-Tier Calibrated Geospatial Command Center Active",
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
    template_path = os.path.join("templates", "dashboard.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/api/v1/locations/bangladesh")
async def get_bangladesh_locations():
    """Returns all 8 Divisions, 64 Districts, and 768 Municipal Wards of Bangladesh."""
    return get_full_bangladesh_hierarchy()


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
    lat: float = Query(23.8103, description="Target Urban Latitude"),
    lon: float = Query(90.4125, description="Target Urban Longitude"),
    days_back: int = Query(14, description="Meteorological incubation window in days")
):
    return await execute_spatiotemporal_pipeline(lat, lon, days_back)


@app.get("/api/v1/ai/train-and-predict-stgnn")
async def run_stgnn_training_and_prediction(
    lat: float = Query(23.8103),
    lon: float = Query(90.4125),
    epochs: int = Query(120, ge=20, le=1000),
    horizon_days: int = Query(
        14, ge=1, le=14, description="Selected Temporal Forecast Horizon (Day +1 to Day +14)"
    ),
    district: str | None = Query(
        None, description="Optional Bangladesh District Name (e.g., Dhaka, Rajshahi, Sherpur, Chattogram)"
    ),
    ward_filter: str | None = Query(None, description="Optional Ward ID filter or ALL")
):
    return await execute_phase4_stgnn_training_and_forecast(
        lat=lat,
        lon=lon,
        epochs=epochs,
        district=district,
        ward_filter=ward_filter,
        horizon_days=horizon_days
    )


@app.get("/api/v1/alerts/evaluate-and-dispatch")
async def evaluate_and_dispatch_municipal_alerts(
    lat: float = Query(23.8103),
    lon: float = Query(90.4125),
    epochs: int = Query(120, ge=20, le=1000),
    horizon_days: int = Query(
        14, ge=1, le=14, description="Selected Temporal Forecast Horizon (Day +1 to Day +14)"
    ),
    alert_threshold: float = Query(0.55, ge=0.10, le=0.95),
    district: str | None = Query(None, description="Any of the 64 Districts of Bangladesh"),
    ward_filter: str | None = Query(None, description="Specific Ward ID or ALL")
):
    return await execute_phase5_command_and_alerting(
        lat=lat,
        lon=lon,
        epochs=epochs,
        alert_threshold=alert_threshold,
        district=district,
        ward_filter=ward_filter,
        horizon_days=horizon_days
    )


@app.get("/api/v1/alerts/ledger")
async def get_municipal_alert_ledger():
    records = read_alert_ledger()
    return {
        "total_logged_dispatches": len(records),
        "ledger_entries": records
    }
