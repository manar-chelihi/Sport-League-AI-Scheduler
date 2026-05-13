from sqlalchemy.orm import Session
import models, schemas, time

class ScheduleRepository:
    @staticmethod
    def format_to_response(db: Session, run: models.AlgorithmRun) -> schemas.ScheduleResponse:
        """Converts DB models to the ScheduleResponse schema expected by the frontend."""
        
        # 1. Optimization: Fetch teams and derbies into memory once
        teams_map = {t.id: t.name for t in db.query(models.Team).all()}
        
        derbies = db.query(models.Derby).all()
        derby_set = set()
        for d in derbies:
            # Sort IDs to ensure (1, 2) matches (2, 1)
            derby_set.add(tuple(sorted((d.team1_id, d.team2_id))))

        matches_list = []
        
        # 2. Loop through the matches associated with this specific run
        for m in run.matches:
            match_pair = tuple(sorted((m.home_team_id, m.away_team_id)))
            is_derby = match_pair in derby_set

            matches_list.append(schemas.MatchResponse(
                round=m.round_num,
                home_team=teams_map.get(m.home_team_id, "Unknown"),
                away_team=teams_map.get(m.away_team_id, "Unknown"),
                is_derby=is_derby,
                travel_distance=m.distance
            ))

        # 3. CRITICAL: Return the object so the frontend receives it
        return schemas.ScheduleResponse(
            method=run.algo_name,
            total_travel_distance=run.total_distance,
            execution_time=run.execution_time,
            rounds=matches_list
        )

    @staticmethod
    def save_run(db: Session, algo_name: str, matches_data: list, total_dist: float, exec_time: float):
        # 1. Create the AlgorithmRun entry
        db_run = models.AlgorithmRun(
            algo_name=algo_name,
            total_distance=total_dist,
            execution_time=exec_time
        )
        db.add(db_run)
        db.flush() 

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