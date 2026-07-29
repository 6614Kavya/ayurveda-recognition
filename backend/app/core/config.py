from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mongodb_uri: str
    db_name: str = "ayurveda_db"
    module3_model_path: str = "models/species_id/vedavision_species_model.pkl"
    module3_health_index_model_path: str = "models/health/vedavision_health_index_model.pkl"
    module3_stage1_model_path: str = "models/health/vedavision_stage1_svm_model.pkl"
    class Config:
        env_file = ".env"

settings = Settings()
