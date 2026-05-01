from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mongodb_uri: str
    db_name: str = "ayurveda_db"

    class Config:
        env_file = ".env"

settings = Settings()