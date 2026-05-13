# services/greedy_service.py
import pandas as pd
import random

class GreedyScheduler:
    def __init__(self, teams_df, distances_df, derbies_df):
        self.teams_df = teams_df
        self.distances_df = distances_df
        self.derbies_df = derbies_df
        
        self.team_ids = list(self.teams_df['team_id'])
        self.id_to_name = dict(zip(self.teams_df['team_id'], self.teams_df['team_name']))
        self.n = len(self.team_ids)
        self.num_rounds = 2 * (self.n - 1)

        self.dist_matrix = self._build_dist_lookup()
        self.derbies = self._build_derby_lookup()

    def _build_dist_lookup(self):
        dist = {}
        for _, row in self.distances_df.iterrows():
            dist[(row['team1_id'], row['team2_id'])] = row['distance_km']
            dist[(row['team2_id'], row['team1_id'])] = row['distance_km']
        return dist

    def _build_derby_lookup(self):
        derby_bonus = {}
        for _, row in self.derbies_df.iterrows():
            bonus = 1000 if row['priority'] == 'High' else 500
            derby_bonus[(row['team1_id'], row['team2_id'])] = bonus
            derby_bonus[(row['team2_id'], row['team1_id'])] = bonus
        return derby_bonus

    def solve(self):
        """Greedy Round-by-Round construction (Cell 13)"""
        history = set()
        current_city = {tid: tid for tid in self.team_ids}
        full_schedule = []
        total_dist = 0

        for r in range(1, self.num_rounds + 1):
            available = self.team_ids.copy()
            round_matches = []
            
            # Try to find matches for this round
            while len(available) >= 2:
                t1 = available.pop(0)
                best_opponent = None
                min_score = float('inf')
                
                for i, t2 in enumerate(available):
                    if (t1, t2) not in history:
                        # Cost = Travel Distance - Derby Bonus
                        dist = self.dist_matrix.get((current_city[t1], t2), 500)
                        bonus = self.derbies.get((t1, t2), 0)
                        score = dist - bonus
                        
                        if score < min_score:
                            min_score = score
                            best_opponent = i
                
                if best_opponent is not None:
                    t2 = available.pop(best_opponent)
                    history.add((t1, t2))
                    
                    # Track travel
                    total_dist += self.dist_matrix.get((current_city[t1], t1), 0) # Home travel
                    total_dist += self.dist_matrix.get((current_city[t2], t1), 0) # T2 travels to T1
                    current_city[t1] = t1
                    current_city[t2] = t1 # T2 is now at T1's city
                    
                    round_matches.append({
                        "round": r,
                        "home_id": t1,
                        "away_id": t2,
                        "is_derby": (t1, t2) in self.derbies
                    })
                else:
                    # If greedy fails to find a valid match, we reset and try again
                    # (Simplified for this version)
                    break
            
            full_schedule.extend(round_matches)

        return full_schedule, total_dist, 0