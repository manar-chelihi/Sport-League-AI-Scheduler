from asyncio import run

from sqlalchemy.orm import Session
import models, schemas, time

class ScheduleRepository:
    @staticmethod
    def format_to_response(db: Session, run: models.AlgorithmRun) -> schemas.ScheduleResponse:
        # 1. Fetch all teams and derbies into memory once
        teams_map = {t.id: t.name for t in db.query(models.Team).all()}
        
        # Create a set of tuples for quick derby lookup
        derbies = db.query(models.Derby).all()
        derby_set = set()
        for d in derbies:
            derby_set.add(tuple(sorted((d.team1_id, d.team2_id))))

        matches_list = []
        for m in run.matches:
            # 2. Lookup values from memory instead of querying the DB
            is_derby = tuple(sorted((m.home_team_id, m.away_team_id))) in derby_set

            matches_list.append(schemas.MatchResponse(
                round=m.round_num,
                home_team=teams_map.get(m.home_team_id, "Unknown"),
                away_team=teams_map.get(m.away_team_id, "Unknown"),
                is_derby=is_derby,
                travel_distance=m.distance
        ))
    @staticmethod
    def save_run(db: Session, algo_name: str, matches_data: list, total_dist: float, exec_time: float):
        # 1. Create the AlgorithmRun entry
        db_run = models.AlgorithmRun(
            algo_name=algo_name,
            total_distance=total_dist,
            execution_time=exec_time
        )
        db.add(db_run)
        db.flush() # Get the run ID

        # 2. Bulk create ScheduledMatch entries
        db_matches = [
            models.ScheduledMatch(
                run_id=db_run.id,
                round_num=m['round'],
                home_team_id=m['home_id'],
                away_team_id=m['away_id'],
                distance=m.get('distance', 0)
            ) for m in matches_data
        ]
        db.add_all(db_matches)
        db.commit()
        db.refresh(db_run)
        return db_run

    @staticmethod
    def format_to_response(db: Session, run: models.AlgorithmRun) -> schemas.ScheduleResponse:
        """Helper to convert DB models to your specific ScheduleResponse schema."""
        matches_list = []
        # We join with Teams to get names and Derbies for the 'is_derby' flag
        for m in run.matches:
            home = db.query(models.Team).filter(models.Team.id == m.home_team_id).first()
            away = db.query(models.Team).filter(models.Team.id == m.away_team_id).first()
            
            # Check if this match is a derby
            is_derby = db.query(models.Derby).filter(
                ((models.Derby.team1_id == m.home_team_id) & (models.Derby.team2_id == m.away_team_id)) |
                ((models.Derby.team1_id == m.away_team_id) & (models.Derby.team2_id == m.home_team_id))
            ).first() is not None

            matches_list.append(schemas.MatchResponse(
                round=m.round_num,
                home_team=home.name,
                away_team=away.name,
                is_derby=is_derby,
                travel_distance=m.distance
            ))

        return schemas.ScheduleResponse(
            method=run.algo_name,
            total_travel_distance=run.total_distance,
            execution_time=run.execution_time,
            rounds=matches_list
        )