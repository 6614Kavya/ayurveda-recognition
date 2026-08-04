from pydantic import BaseModel
from typing import List, Dict

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

class HealthAssessmentResponse(BaseModel):
    species              : str
    decision             : str            # "healthy" | "unhealthy"  (Stage 1)
    decision_confidence  : float          # 0-1
    health_value         : float          # 0-100, 100 = healthiest
    severity_score_raw   : float          # 0-100, internal severity direction (100 = most severe)
    breakdown            : Dict[str, float]  # per-subscore % contribution to deviation
