import httpx
from datetime import datetime, timedelta

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

async def fetch_nasa_meteorological_matrix(lat: float, lon: float, days_back: int = 14) -> dict:
    """
    Ingests continuous daily thermal, humidity, and precipitation data from NASA POWER API
    to evaluate Aedes mosquito biological incubation thresholds.
    """
    # NASA POWER near-real-time satellite assimilation has a 2-day processing lag
    end_date = datetime.utcnow() - timedelta(days=2)
    start_date = end_date - timedelta(days=days_back)

    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.get(NASA_POWER_URL, params=params)
        response.raise_for_status()
        raw_data = response.json()

    parameter_block = raw_data.get("properties", {}).get("parameter", {})

    # Filter out NASA's -999.0 fill value for any unassimilated days
    cleaned_series = {}
    for param_key, daily_dict in parameter_block.items():
        cleaned_series[param_key] = {
            date_str: val for date_str, val in daily_dict.items() if val != -999.0
        }

    return {
        "source": "NASA POWER Agroclimatology Daily API",
        "coordinates": {"latitude": lat, "longitude": lon},
        "window": {"start": start_date.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d")},
        "metrics_legend": {
            "T2M": "Temperature at 2 Meters (C)",
            "RH2M": "Relative Humidity at 2 Meters (%)",
            "PRECTOTCORR": "Corrected Precipitation (mm/day)"
        },
        "time_series": cleaned_series
    }
