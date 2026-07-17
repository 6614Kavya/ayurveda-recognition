from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mongodb_uri: str
    db_name: str = "ayurveda_db"
    module3_model_path: str = "models/species_id/vedavision_species_model.pkl"
    class Config:
        env_file = ".env"

settings = Settings()