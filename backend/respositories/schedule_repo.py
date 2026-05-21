# repositories/schedule_repo.py

from sqlalchemy.orm import Session
import models
import schemas
import json


class ScheduleRepository:

    @staticmethod
    def save_run(
        db: Session,
        solver_type: str,
        schedule: list,
        metrics: dict,
        search_stats: dict = None,
        cost_history: list = None,
        blackout_rounds: list = None,
        hard_constraints_ok: bool = True,
        execution_time_sec: float = None,
        parameters: dict = None,
    ):

        # =====================================================
        # CREATE SOLVER RUN
        # =====================================================

        db_run = models.SolverRun(
            solver_type=solver_type,
            parameters=str(parameters) if parameters else None,
            blackout_rounds_used=str(blackout_rounds)
            if blackout_rounds else None,
            hard_constraints_ok=hard_constraints_ok,
            execution_time_sec=execution_time_sec,
        )

        db.add(db_run)
        db.flush()

        # =====================================================
        # SAVE MATCHES
        # =====================================================

        db_matches = []

        for match in schedule:

            db_matches.append(

                models.ScheduleMatch(
                    run_id=db_run.run_id,

                    round_number=match["round"],

                    match_index=match.get("match_index"),

                    home_team_id=match["home_id"],

                    away_team_id=match["away_id"],

                    is_derby=match.get(
                        "is_derby",
                        False
                    ),

                    distance_km=match.get(
                        "distance_km",
                        match.get("distance", 0)
                    ),

                    play_day=match.get("play_day"),

                    weekday=match.get("weekday"),
                )
            )

        db.add_all(db_matches)

        # =====================================================
        # SAVE METRICS
        # =====================================================

        db_metrics = models.RunMetric(

            run_id=db_run.run_id,

            total_travel_km=metrics.get(
                "total_travel_km",
                metrics.get("travel_distance_km",
                            metrics.get("travel_distance"))
            ),

            rest_imbalance=metrics.get(
                "rest_imbalance"
            ),

            away_break_violations=metrics.get(
                "away_break_violations",
                metrics.get("away_penalty")
            ),

            derby_blackout_violations=metrics.get(
                "derby_blackout_violations",
                metrics.get("derby_penalty")
            ),

            objective_cost=metrics.get(
                "objective_cost",
                metrics.get("objective_score")
            ),
        )

        db.add(db_metrics)

        # =====================================================
        # SAVE SEARCH STATISTICS
        # =====================================================

        if search_stats:

            db_stats = models.SearchStat(

                run_id=db_run.run_id,

                iterations_performed=search_stats.get(
                    "iterations_completed",
                    search_stats.get(
                        "iterations_performed"
                    )
                ),

                explored_neighbors=search_stats.get(
                    "explored_neighbors"
                ),

                accepted_moves=search_stats.get(
                    "accepted_moves"
                ),

                rejected_moves=search_stats.get(
                    "rejected_moves"
                ),

                improving_moves=search_stats.get(
                    "improving_moves"
                ),

                backtrack_calls=search_stats.get(
                    "backtrack_calls"
                ),

                constraint_checks=search_stats.get(
                    "constraint_checks"
                ),

                forward_check_failures=search_stats.get(
                    "forward_check_failures"
                ),
            )

            db.add(db_stats)

        # =====================================================
        # SAVE COST HISTORY
        # =====================================================

        if cost_history:

            history_rows = []

            for i, cost in enumerate(cost_history):

                history_rows.append(

                    models.CostHistory(
                        run_id=db_run.run_id,
                        iteration=i,
                        best_cost=float(cost),
                        current_cost=float(cost),
                    )
                )

            db.add_all(history_rows)

        db.commit()

        db.refresh(db_run)

        return db_run

    # =========================================================
    # FIX: Added missing get_full_result method.
    # Called by: sa_router, hc_router, csp_router, greedy_router,
    #            schedules.py — all routers that previously crashed
    #            with AttributeError because this method was absent.
    # =========================================================

    @staticmethod
    def get_full_result(
        db: Session,
        run_id: int
    ) -> schemas.FullSolverResult:

        run = db.query(
            models.SolverRun
        ).filter(
            models.SolverRun.run_id == run_id
        ).first()

        matches = db.query(
            models.ScheduleMatch
        ).filter(
            models.ScheduleMatch.run_id == run_id
        ).order_by(
            models.ScheduleMatch.round_number,
            models.ScheduleMatch.match_index
        ).all()

        metrics = db.query(
            models.RunMetric
        ).filter(
            models.RunMetric.run_id == run_id
        ).first()

        search_stats = db.query(
            models.SearchStat
        ).filter(
            models.SearchStat.run_id == run_id
        ).first()

        cost_history = db.query(
            models.CostHistory
        ).filter(
            models.CostHistory.run_id == run_id
        ).order_by(
            models.CostHistory.iteration
        ).all()

        return schemas.FullSolverResult(
            run=run,
            matches=matches,
            metrics=metrics,
            search_stats=search_stats,
            cost_history=cost_history,
        )