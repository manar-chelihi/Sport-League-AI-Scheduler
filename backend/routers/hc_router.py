
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal

from services.hc_service import HillClimbingScheduler
from respositories.schedule_repo import ScheduleRepository

import models
import schemas
import pandas as pd


router = APIRouter(
    prefix="/generate/hc",
    tags=["Algorithms"]
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post(
    "/",
    response_model=schemas.FullSolverResult
)
def run_hc(
    db: Session = Depends(get_db)
):

    # =====================================================
    # LOAD DATABASE TABLES
    # =====================================================

    teams = pd.read_sql(
        db.query(models.Team).statement,
        db.bind
    )

    dist = pd.read_sql(
        db.query(models.Distance).statement,
        db.bind
    )

    derbies = pd.read_sql(
        db.query(models.Derby).statement,
        db.bind
    )

    # =====================================================
    # RUN ALGORITHM
    # =====================================================

    scheduler = HillClimbingScheduler(
        teams,
        dist,
        derbies
    )

    result = scheduler.solve()

    # =====================================================
    # SAVE RESULT
    # =====================================================

    db_run = ScheduleRepository.save_run(

        db=db,

        solver_type="HillClimbing",

        schedule=result["schedule"],

        metrics=result["metrics"],

        search_stats=result.get(
            "search_statistics"
        ),

        cost_history=result.get(
            "cost_history"
        ),

        blackout_rounds=result.get(
            "blackout_rounds"
        ),

        hard_constraints_ok=result[
            "constraint_report"
        ]["valid_after_optimization"],

        execution_time_sec=result[
            "metrics"
        ].get(
            "execution_time_seconds"
        ),
    )

    # =====================================================
    # RETURN FULL RESULT
    # =====================================================

   
    return ScheduleRepository.get_full_result(
        db,
        db_run.run_id,
        constraint_report=result.get("constraint_report"),
    )
