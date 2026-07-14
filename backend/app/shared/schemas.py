from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    plant_name       : str
    confidence       : float
    module           : str
    sinhala_name     : str = ""
    uses             : str = ""
    diseases_treated : List[str] = []

class HealthResponse(BaseModel):
    status: str
    database: str
    version: str = "1.0.0"