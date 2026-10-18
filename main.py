from fastapi import FastAPI, Query
import asyncio
from datetime import datetime

from services.sentinel_client import fetch_sentinel2_waterlogging_scenes
from services.nasa_power_client import fetch_nasa_meteorological_matrix
from services.dghs_client import fetch_dghs_dengue_baseline
from services.spatiotemporal_pipeline import execute_spatiotemporal_pipeline

app = FastAPI(
    title="EpiGrid: Epidemiological Preemption Engine",
    description="Phase 3: Spatio-Temporal Pipeline (Sentinel-2 MNDWI Rooftop/Alley Isolation & Aedes Thermodynamic Synchronization)",
    version="3.0.0"
)

@app.get("/")
async def root():
    return {
        "project": "EpiGrid",
        "phase": "Phase 3: Spatio-Temporal Pipeline Active",
        "timestamp_utc": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    await asyncio.sleep(0.05)
    return {"server": "Uvicorn ASGI", "healthy": True, "phase": 3}

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
    """
    Phase 3 Pipeline: Isolates urban rooftop/alley water-retaining coordinates via Sentinel-2
    MNDWI/NDWI morphological segmentation and synchronizes them with NASA POWER thermal data
    to compute Aedes aegypti biological incubation parameters (EIP, MDR, Gonotrophic Cycle).
    """
    return await execute_spatiotemporal_pipeline(lat, lon, days_back)
