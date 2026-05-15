import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
import models


def seed_data():
    db: Session = SessionLocal()

    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    try:

        # =====================================================
        # 1. SEED TEAMS
        # FIX: model uses team_id (not id), team_name (not name),
        #      latitude + longitude (not strength — that column
        #      does not exist in models.Team).
        # =====================================================

        if db.query(models.Team).count() == 0:
            print("Seeding teams...")
            teams_df = pd.read_csv("Data/teams.csv")
            for _, row in teams_df.iterrows():
                team = models.Team(
                    team_id=int(row['team_id']),
                    team_name=str(row['team_name']),
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                )
                db.add(team)
            db.commit()
            print(f"  → {len(teams_df)} teams seeded.")

        # =====================================================
        # 2. SEED DISTANCES
        # No change needed — team1_id, team2_id, distance_km
        # all match models.Distance exactly.
        # =====================================================

        if db.query(models.Distance).count() == 0:
            print("Seeding distances...")
            dist_df = pd.read_csv("Data/distances.csv")
            for _, row in dist_df.iterrows():
                dist = models.Distance(
                    team1_id=int(row['team1_id']),
                    team2_id=int(row['team2_id']),
                    distance_km=float(row['distance_km']),
                )
                db.add(dist)
            db.commit()
            print(f"  → {len(dist_df)} distances seeded.")

        # =====================================================
        # 3. SEED DERBIES
        # FIX: removed match_type — that column does not exist
        #      in models.Derby. Only team1_id, team2_id, priority
        #      are valid. priority valid values: 'High'/'Medium'/'Low'.
        # =====================================================

        if db.query(models.Derby).count() == 0:
            print("Seeding derbies...")
            derby_df = pd.read_csv("Data/derbies.csv")
            for _, row in derby_df.iterrows():
                derby = models.Derby(
                    team1_id=int(row['team1_id']),
                    team2_id=int(row['team2_id']),
                    priority=str(row.get('priority', 'Medium')),
                )
                db.add(derby)
            db.commit()
            print(f"  → {len(derby_df)} derbies seeded.")

        # =====================================================
        # 4. SEED BLACKOUT ROUNDS
        # FIX: was missing entirely from the original seed file.
        #      models.BlackoutRound has round_number (PK) and
        #      an optional description.
        # =====================================================

        if db.query(models.BlackoutRound).count() == 0:
            print("Seeding blackout rounds...")
            try:
                blackout_df = pd.read_csv("Data/blackout_rounds.csv")
                for _, row in blackout_df.iterrows():
                    blackout = models.BlackoutRound(
                        round_number=int(row['round_number']),
                        description=str(row['description'])
                        if 'description' in row and pd.notna(row['description'])
                        else None,
                    )
                    db.add(blackout)
                db.commit()
                print(f"  → {len(blackout_df)} blackout rounds seeded.")
            except FileNotFoundError:
                print("  → Data/blackout_rounds.csv not found, skipping.")

        print("\nDatabase seeded successfully!")

    except Exception as e:
        print(f"Error seeding data: {e}")
        db.rollback()

    finally:
        db.close()


if __name__ == "__main__":
    seed_data()