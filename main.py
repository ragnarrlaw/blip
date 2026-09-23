from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from db.db import db_manager
import model.model as model# Must be imported so Base knows about your schema
from router import sync, vle

# Initialize database tables
db_manager.create_tables()

app = FastAPI(
    title="Calibration-Aware Automated Grading API",
    version="1.1",
    description="Backend for Moodle integration and RAG-based LLM evaluation."
)

app.include_router(vle.router)
app.include_router(sync.router)

@app.get("/health")
def health_check(db: Session = Depends(db_manager.get_session)):
    """Verifies that the config-injected database is reachable."""
    # A simple query to ensure the connection and tables are active
    course_count = db.query(model.Course).count()
    return {
        "status": "operational",
        "courses_tracked": course_count,
        "database_url": db_manager.config.blip_database_url
    }