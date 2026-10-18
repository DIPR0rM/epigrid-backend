from fastapi import FastAPI
import asyncio

app = FastAPI(
    title="EpiGrid: Epidemiological Preemption Engine",
    description="Phase 1 Cloud Backend for Dengue Outbreak Forecasting",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "project": "EpiGrid",
        "phase": "Phase 1: Cloud Infrastructure Initialized",
        "status": "Online and handling async requests"
    }

@app.get("/health")
async def health_check():
    # Simulating an asynchronous non-blocking operation
    await asyncio.sleep(0.1)
    return {"server": "Uvicorn ASGI", "healthy": True}
