
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import SessionLocal

from respositories.schedule_repo import ScheduleRepository

import models
import schemas

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter(
    prefix="/schedules",
    tags=["Management"]
)


@router.get(
    "/",
    response_model=list[schemas.SolverRun]
)
def list_all_runs(
    db: Session = Depends(get_db)
):

    return db.query(
        models.SolverRun
    ).all()


@router.get(
    "/{run_id}",
    response_model=schemas.FullSolverResult
)
def get_run_details(
    run_id: int,
    db: Session = Depends(get_db)
):

    run = db.query(
        models.SolverRun
    ).filter(
        models.SolverRun.run_id == run_id
    ).first()

    if not run:

        raise HTTPException(
            status_code=404,
            detail="Run not found"
        )

    return ScheduleRepository.get_full_result(
        db,
        run_id
    )


@router.delete("/{run_id}")
def delete_run(
    run_id: int,
    db: Session = Depends(get_db)
):

    run = db.query(
        models.SolverRun
    ).filter(
        models.SolverRun.run_id == run_id
    ).first()

    if not run:

        raise HTTPException(
            status_code=404,
            detail="Run not found"
        )

    db.delete(run)

    db.commit()

    return {
        "status": "deleted"
    }
