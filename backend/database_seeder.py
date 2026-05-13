import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
import models

def seed_data():
    db: Session = SessionLocal()
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    try:
        # 1. Seed Teams
        if db.query(models.Team).count() == 0:
            print("Seeding teams...")
            teams_df = pd.read_csv("Data/teams.csv")
            for _, row in teams_df.iterrows():
                team = models.Team(
                    id=int(row['team_id']),
                    name=row['team_name'],
                    strength=row.get('strength', 50) # Fallback if strength column missing
                )
                db.add(team)
            db.commit()

        # 2. Seed Distances
        if db.query(models.Distance).count() == 0:
            print("Seeding distances...")
            dist_df = pd.read_csv("Data/distances.csv")
            for _, row in dist_df.iterrows():
                dist = models.Distance(
                    team1_id=int(row['team1_id']),
                    team2_id=int(row['team2_id']),
                    distance_km=float(row['distance_km'])
                )
                db.add(dist)
            db.commit()

        # 3. Seed Derbies
        if db.query(models.Derby).count() == 0:
            print("Seeding derbies...")
            derby_df = pd.read_csv("Data/derbies.csv")
            for _, row in derby_df.iterrows():
                derby = models.Derby(
                    team1_id=int(row['team1_id']),
                    team2_id=int(row['team2_id']),
                    match_type=row.get('match_type', 'Derby'),
                    priority=row.get('priority', 'Medium')
                )
                db.add(derby)
            db.commit()
            
        print("Database seeded successfully!")

    except Exception as e:
        print(f"Error seeding data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()