from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base, engine, SessionLocal

class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    strength = Column(Integer, default=50)

class Distance(Base):
    __tablename__ = "distances"
    team1_id = Column(Integer, ForeignKey("teams.id"), primary_key=True)
    team2_id = Column(Integer, ForeignKey("teams.id"), primary_key=True)
    distance_km = Column(Float, nullable=False)

class Derby(Base):
    __tablename__ = "derbies"
    id = Column(Integer, primary_key=True, index=True)
    team1_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    team2_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    match_type = Column(String)
    priority = Column(String)

class AlgorithmRun(Base):
    __tablename__ = "algorithm_runs"
    id = Column(Integer, primary_key=True, index=True)
    algo_name = Column(String, nullable=False)
    total_distance = Column(Float, default=0)
    execution_time = Column(Float, default=0)

    matches = relationship("ScheduledMatch", back_populates="run", cascade="all, delete")

class ScheduledMatch(Base):
    __tablename__ = "scheduled_matches"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("algorithm_runs.id"))
    round_num = Column(Integer, nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    distance = Column(Float, default=0)

    run = relationship("AlgorithmRun", back_populates="matches")