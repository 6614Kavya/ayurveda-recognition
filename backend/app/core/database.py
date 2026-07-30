import re
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

client: AsyncIOMotorClient = None
db = None

async def connect_db():
    global client, db
    client =  AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.db_name]
    print(f"Connected to MongoDB: {settings.db_name}")

async def close_db():
    global client
    if client:
        client.close()
        print("MongoDB connection closed")

def get_db():
    return db



async def get_species_metadata(plant_name: str) -> dict:
    database = get_db()
    if database is None:
        print("DEBUG: Database connection is None!")
        return {"sinhala_name": "", "uses": "", "diseases_treated": []}

    clean_name = plant_name.strip()

    # Query matches either 'label' or 'plant_name' case-insensitively
    query = {
        "$or": [
            {"label": re.compile(f"^{re.escape(clean_name)}$", re.IGNORECASE)},
            {"plant_name": re.compile(f"^{re.escape(clean_name)}$", re.IGNORECASE)}
        ]
    }

    # IMPORTANT: Ensure 'species' matches your EXACT collection name in Compass/Atlas
    collection = database["single_leaves"]
    
    doc = await collection.find_one(query)
    print(f"DEBUG: Searched for '{clean_name}', found doc: {doc}")

    if doc:
        return {
            "sinhala_name": doc.get("sinhala_name", ""),
            "uses": doc.get("uses", ""),
            "diseases_treated": doc.get("diseases_treated", []),
        }

    return {"sinhala_name": "", "uses": "", "diseases_treated": []}