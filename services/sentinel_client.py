import os
import httpx
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

async def get_cdse_oauth_token() -> bool:
    """Authenticates with Copernicus Identity Server using OAuth2 Password Grant."""
    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")
    if not username or not password or "example.com" in username:
        return False

    payload = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password"
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(CDSE_TOKEN_URL, data=payload)
        return response.status_code == 200

async def fetch_sentinel2_waterlogging_scenes(lat: float, lon: float, days_back: int = 30) -> dict:
    """
    Queries Copernicus STAC v1 API for Sentinel-2 L2A multispectral scenes
    and extracts spectral band links (B03 Green, B08 NIR, B11 SWIR) for NDWI/MNDWI calculation.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    datetime_range = f"{start_date.strftime('%Y-%m-%dT00:00:00Z')}/{end_date.strftime('%Y-%m-%dT23:59:59Z')}"

    # Define a ~2km bounding box around the target urban coordinate
    delta = 0.02
    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

    stac_payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": 5
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(CDSE_STAC_URL, json=stac_payload)
        response.raise_for_status()
        features = response.json().get("features", [])

    oauth_ready = await get_cdse_oauth_token()
    parsed_scenes = []

    for item in features:
        props = item.get("properties", {})
        assets = item.get("assets", {})
        parsed_scenes.append({
            "scene_id": item.get("id"),
            "acquisition_timestamp": props.get("datetime"),
            "cloud_cover_percentage": props.get("eo:cloud_cover"),
            "waterlogging_spectral_bands": {
                "B03_Green_10m": assets.get("B03_10m", {}).get("href"),
                "B08_NIR_10m": assets.get("B08_10m", {}).get("href"),
                "B11_SWIR_20m": assets.get("B11_20m", {}).get("href"),
                "SCL_Scene_Classification_20m": assets.get("SCL_20m", {}).get("href")
            }
        })

    return {
        "source": "Copernicus Data Space Ecosystem (STAC v1 API)",
        "collection": "sentinel-2-l2a",
        "oauth2_authenticated": oauth_ready,
        "target_bbox": bbox,
        "scenes_retrieved": len(parsed_scenes),
        "latest_scenes": parsed_scenes
    }
