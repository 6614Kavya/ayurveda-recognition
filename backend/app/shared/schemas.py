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

class SymptomOut(BaseModel):
    name             : str    # e.g. "Yellowing" -- never a raw `worst_*` column name
    description      : str    # one-line plain-language explanation
    group            : str    # e.g. "Discolouration", for UI grouping
    percentage       : float  # % share of this leaf's own total deviation; nonzero entries sum to ~100

class HealthAssessmentResponse(BaseModel):
    species              : str
    decision             : str            # "healthy" | "unhealthy"  (Stage 1)
    decision_confidence  : float          # 0-1
    health_value         : float          # 0-100, 100 = healthiest
    severity_score_raw   : float          # 0-100, internal severity direction (100 = most severe)
    symptoms             : List[SymptomOut]  # notable findings, worst-first; [] if none