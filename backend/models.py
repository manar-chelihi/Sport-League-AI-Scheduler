# models.py
from sqlalchemy import (
    Column, ForeignKeyConstraint, Integer, String, Float, Boolean, ForeignKey, CheckConstraint,
    UniqueConstraint, PrimaryKeyConstraint, TIMESTAMP
)
from sqlalchemy.sql import func
from database import Base


class Team(Base):
    __tablename__ = "teams"

    team_id = Column(Integer, primary_key=True, index=True)
    team_name = Column(String, unique=True, nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)


class Distance(Base):
    __tablename__ = "distances"
    __table_args__ = (
        CheckConstraint("team1_id < team2_id", name="check_team1_lt_team2"),
        ForeignKeyConstraint(["team1_id"], ["teams.team_id"]),
        ForeignKeyConstraint(["team2_id"], ["teams.team_id"]),
        PrimaryKeyConstraint("team1_id", "team2_id"),
    )

    team1_id = Column(Integer, nullable=False)
    team2_id = Column(Integer, nullable=False)
    distance_km = Column(Float, nullable=False)


class Derby(Base):
    __tablename__ = "derbies"
    __table_args__ = (
        CheckConstraint("team1_id < team2_id", name="check_derby_team1_lt_team2"),
        UniqueConstraint("team1_id", "team2_id"),
        ForeignKeyConstraint(["team1_id"], ["teams.team_id"]),
        ForeignKeyConstraint(["team2_id"], ["teams.team_id"]),
    )

    derby_id = Column(Integer, primary_key=True, index=True)
    team1_id = Column(Integer, nullable=False)
    team2_id = Column(Integer, nullable=False)
    priority = Column(String, nullable=False)  # 'High', 'Medium', 'Low'


class BlackoutRound(Base):
    __tablename__ = "blackout_rounds"

    round_number = Column(Integer, primary_key=True)
    description = Column(String, nullable=True)


class SolverRun(Base):
    __tablename__ = "solver_runs"

    run_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    solver_type = Column(String, nullable=False)  # 'HillClimbing', 'SimulatedAnnealing', 'Greedy', 'CSP'
    start_time = Column(TIMESTAMP, server_default=func.now())
    end_time = Column(TIMESTAMP, nullable=True)
    parameters = Column(String, nullable=True)           # JSON string
    blackout_rounds_used = Column(String, nullable=True) # JSON list
    hard_constraints_ok = Column(Boolean, nullable=False, default=False)
    execution_time_sec = Column(Float, nullable=True)


class ScheduleMatch(Base):
    __tablename__ = "schedule_matches"
    __table_args__ = (
        UniqueConstraint("run_id", "round_number", "home_team_id", "away_team_id",
                         name="uq_match_per_run_round"),
        ForeignKeyConstraint(["run_id"], ["solver_runs.run_id"], ondelete="CASCADE"),
        ForeignKeyConstraint(["home_team_id"], ["teams.team_id"]),
        ForeignKeyConstraint(["away_team_id"], ["teams.team_id"]),
    )

    match_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, nullable=False)
    round_number = Column(Integer, nullable=False)
    match_index = Column(Integer, nullable=True)
    home_team_id = Column(Integer, nullable=False)
    away_team_id = Column(Integer, nullable=False)
    is_derby = Column(Boolean, nullable=False, default=False)
    distance_km = Column(Float, nullable=False)
    play_day = Column(Integer, nullable=True)
    weekday = Column(Integer, nullable=True)  # 1=Monday … 7=Sunday


class RunMetric(Base):
    __tablename__ = "run_metrics"

    run_id = Column(Integer, ForeignKey("solver_runs.run_id"), primary_key=True)
    total_travel_km = Column(Float, nullable=True)
    rest_imbalance = Column(Float, nullable=True)
    away_break_violations = Column(Integer, nullable=True)
    derby_blackout_violations = Column(Integer, nullable=True)
    objective_cost = Column(Float, nullable=True)


class SearchStat(Base):
    __tablename__ = "search_stats"

    run_id = Column(Integer, ForeignKey("solver_runs.run_id"), primary_key=True)
    iterations_performed = Column(Integer, nullable=True)
    explored_neighbors = Column(Integer, nullable=True)
    accepted_moves = Column(Integer, nullable=True)     # SA only
    rejected_moves = Column(Integer, nullable=True)     # SA only
    improving_moves = Column(Integer, nullable=True)    # SA only
    backtrack_calls = Column(Integer, nullable=True)    # CSP only
    constraint_checks = Column(Integer, nullable=True)  # CSP only
    forward_check_failures = Column(Integer, nullable=True)  # CSP only


class CostHistory(Base):
    __tablename__ = "cost_history"

    history_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("solver_runs.run_id"), nullable=False)
    iteration = Column(Integer, nullable=False)
    best_cost = Column(Float, nullable=True)
    current_cost = Column(Float, nullable=True)