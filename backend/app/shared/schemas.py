from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    plant_name: str
    confidence: float
    module: str
    sinhala_name: Optional[str] = None
    uses: Optional[str] = None
    diseases_treated: Optional[list[str]] = None

class HealthResponse(BaseModel):
    status: str
    database: str
    version: str = "1.0.0"