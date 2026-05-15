from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import SessionLocal
import models

router = APIRouter(prefix="/teams", tags=["Teams"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_teams(db: Session = Depends(get_db)):
    teams = db.query(models.Team).all()

    return [
        {
            "id": t.team_id,
            "name": t.team_name,
            "lat": t.latitude,
            "lng": t.longitude
        }
        for t in teams
    ]