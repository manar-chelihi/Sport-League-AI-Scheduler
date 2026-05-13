from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from services.csp_service import CSPScheduler
from respositories.schedule_repo import ScheduleRepository
import models, schemas, time
import pandas as pd

router = APIRouter(prefix="/generate/csp", tags=["Algorithms"])

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@router.post("/", response_model=schemas.ScheduleResponse)
def run_csp(db: Session = Depends(get_db)):
    # 1. Load data from DB into DataFrames for the Service
    teams = pd.read_sql(db.query(models.Team).statement, db.bind).rename(columns={
        'id': 'team_id', 
        'name': 'team_name'
    })
    dist = pd.read_sql(db.query(models.Distance).statement, db.bind)
    derbies = pd.read_sql(db.query(models.Derby).statement, db.bind)

    # 2. Run Algorithm and measure time
    start_time = time.time()
    scheduler = CSPScheduler(teams, dist, derbies)
    matches, total_dist, _ = scheduler.solve()
    exec_time = time.time() - start_time

    # 3. Save and Return
    run = ScheduleRepository.save_run(db, "CSP", matches, total_dist, exec_time)
    return ScheduleRepository.format_to_response(db, run)