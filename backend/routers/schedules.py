from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from routers.csp_router import get_db
from database import SessionLocal
import models, schemas
from respositories.schedule_repo import ScheduleRepository

router = APIRouter(prefix="/schedules", tags=["Management"])

@router.get("/", response_model=list[schemas.AlgorithmRun])
def list_all_runs(db: Session = Depends(get_db)):
    """Used for the comparison table on the frontend."""
    return db.query(models.AlgorithmRun).all()

@router.get("/{run_id}", response_model=schemas.ScheduleResponse)
def get_run_details(run_id: int, db: Session = Depends(get_db)):
    """Used to view the full 22-round schedule for a specific algorithm."""
    run = db.query(models.AlgorithmRun).filter(models.AlgorithmRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return ScheduleRepository.format_to_response(db, run)

@router.delete("/{run_id}")
def delete_run(run_id: int, db: Session = Depends(get_db)):
    """Deletes the result from the database."""
    run = db.query(models.AlgorithmRun).filter(models.AlgorithmRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Cascade delete matches manually if not handled by DB constraints
    db.query(models.ScheduledMatch).filter(models.ScheduledMatch.run_id == run_id).delete()
    db.delete(run)
    db.commit()
    return {"status": "deleted"}