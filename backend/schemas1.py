from pydantic import BaseModel
from typing import List

class Team(BaseModel):
    id: int
    name: str
    strength: int
    
    class Config:
        from_attributes = True

class Distance(BaseModel):
    team1_id: int
    team2_id: int
    distance_km: float
    
    class Config:
        from_attributes = True

class Derby(BaseModel):
    id: int
    team1_id: int
    team2_id: int
    match_type: str
    priority: str
    
    class Config:
        from_attributes = True

class ScheduledMatch(BaseModel):
    id: int
    run_id: int
    round_num: int
    home_team_id: int
    away_team_id: int
    
    class Config:
        from_attributes = True

class AlgorithmRun(BaseModel):
    id: int
    algo_name: str
    total_distance: float
    execution_time: float
    matches: List[ScheduledMatch] = []
    
    class Config:
        from_attributes = True

class MatchResponse(BaseModel):
    round: int
    home_team: str
    away_team: str
    is_derby: bool
    travel_distance: float

class ScheduleResponse(BaseModel):
    method: str
    total_travel_distance: float
    execution_time: float
    rounds: List[MatchResponse]