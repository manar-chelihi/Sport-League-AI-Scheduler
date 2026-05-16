# schemas.py
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator
import ast


# ---------- Base schemas (used for reading/writing) ----------

class TeamBase(BaseModel):
    team_name: str
    latitude: float
    longitude: float

class TeamCreate(TeamBase):
    pass

class Team(TeamBase):
    team_id: int
    model_config = ConfigDict(from_attributes=True)


class DistanceBase(BaseModel):
    team1_id: int
    team2_id: int
    distance_km: float

class DistanceCreate(DistanceBase):
    pass

class Distance(DistanceBase):
    model_config = ConfigDict(from_attributes=True)


class DerbyBase(BaseModel):
    team1_id: int
    team2_id: int
    priority: str  # 'High', 'Medium', 'Low'

class DerbyCreate(DerbyBase):
    pass

class Derby(DerbyBase):
    derby_id: int
    model_config = ConfigDict(from_attributes=True)


class BlackoutRoundBase(BaseModel):
    round_number: int
    description: Optional[str] = None

class BlackoutRoundCreate(BlackoutRoundBase):
    pass

class BlackoutRound(BlackoutRoundBase):
    model_config = ConfigDict(from_attributes=True)


class SolverRunBase(BaseModel):
    solver_type: str
    parameters: Optional[Dict[str, Any]] = None
    blackout_rounds_used: Optional[List[int]] = None
    hard_constraints_ok: bool = False
    execution_time_sec: Optional[float] = None

class SolverRunCreate(SolverRunBase):
    pass

class SolverRun(SolverRunBase):
    run_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

    # FIX: schedule_repo.py stores blackout_rounds_used in the DB as a plain
    #      Python repr string (e.g. "[1, 2, 34]") via str(blackout_rounds).
    #      When SQLAlchemy reads it back, Pydantic receives a str but the field
    #      type is Optional[List[int]], causing a ValidationError.
    #      This validator intercepts the raw value before Pydantic validates it
    #      and parses it back to a list when needed.
    @field_validator("blackout_rounds_used", mode="before")
    @classmethod
    def parse_blackout_rounds(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # ast.literal_eval safely parses "[1, 2, 34]" → [1, 2, 34]
            parsed = ast.literal_eval(v)
            if isinstance(parsed, list):
                return parsed
        return v

    # FIX: schedule_repo.py also stores parameters as str(parameters).
    #      Same treatment: parse it back to a dict when it arrives as a string.
    @field_validator("parameters", mode="before")
    @classmethod
    def parse_parameters(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            parsed = ast.literal_eval(v)
            if isinstance(parsed, dict):
                return parsed
        return v


class ScheduleMatchBase(BaseModel):
    round_number: int
    match_index: Optional[int] = None
    home_team_id: int
    away_team_id: int
    is_derby: bool = False
    distance_km: float
    play_day: Optional[int] = None
    weekday: Optional[int] = Field(None, ge=1, le=7)

class ScheduleMatchCreate(ScheduleMatchBase):
    pass

class ScheduleMatch(ScheduleMatchBase):
    match_id: int
    run_id: int
    model_config = ConfigDict(from_attributes=True)


class RunMetricBase(BaseModel):
    total_travel_km: Optional[float] = None
    rest_imbalance: Optional[float] = None
    away_break_violations: Optional[int] = None
    derby_blackout_violations: Optional[int] = None
    objective_cost: Optional[float] = None

class RunMetricCreate(RunMetricBase):
    run_id: int

class RunMetric(RunMetricBase):
    run_id: int
    model_config = ConfigDict(from_attributes=True)


class SearchStatBase(BaseModel):
    iterations_performed: Optional[int] = None
    explored_neighbors: Optional[int] = None
    accepted_moves: Optional[int] = None
    rejected_moves: Optional[int] = None
    improving_moves: Optional[int] = None
    backtrack_calls: Optional[int] = None
    constraint_checks: Optional[int] = None
    forward_check_failures: Optional[int] = None

class SearchStatCreate(SearchStatBase):
    run_id: int

class SearchStat(SearchStatBase):
    run_id: int
    model_config = ConfigDict(from_attributes=True)


class CostHistoryBase(BaseModel):
    iteration: int
    best_cost: Optional[float] = None
    current_cost: Optional[float] = None

class CostHistoryCreate(CostHistoryBase):
    run_id: int

class CostHistory(CostHistoryBase):
    history_id: int
    run_id: int
    model_config = ConfigDict(from_attributes=True)


# ---------- Composite response schemas (for API convenience) ----------

class FullSolverResult(BaseModel):
    run: SolverRun
    matches: List[ScheduleMatch]
    metrics: Optional[RunMetric] = None
    search_stats: Optional[SearchStat] = None
    cost_history: List[CostHistory] = []