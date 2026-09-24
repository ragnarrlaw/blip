from fastapi import FastAPI
from db.db import db_manager
from router import vle, sync, assignments, grading
import sys
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Create database tables defined in model/model.py
db_manager.create_tables()

app = FastAPI(
    title="Calibration-Aware Automated Grading API",
    version="1.2",
    description="Backend for Moodle RPA integration and calibrated RAG grading evaluation.",
)

# Register endpoints
app.include_router(vle.router)
app.include_router(sync.router)
app.include_router(assignments.router)
app.include_router(grading.router)


@app.get("/health")
def health_check():
    return {
        "status": "operational",
        "database_url": db_manager.config.blip_database_url,
    }

