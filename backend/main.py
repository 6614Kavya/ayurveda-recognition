from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.database import connect_db, close_db, get_db
from app.shared.schemas import HealthResponse
from app.module1_flowers.router import router as m1_router
from app.module2_single_leaves.router import router as m2_router
from app.module3_compound_leaves.router import router as m3_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()      # runs on startup
    yield
    await close_db()        # runs on shutdown

app = FastAPI(
    title="Ayurveda Plant Recognition API",
    description="Three-module system for identifying Sri Lankan Ayurveda plants via leaf and flower images",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:57517"],  # Flutter port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all three module routers
app.include_router(m1_router)
app.include_router(m2_router)
app.include_router(m3_router)

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    db = get_db()
    try:
        await db.command("ping")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    
    return HealthResponse(status="ok", database=db_status)