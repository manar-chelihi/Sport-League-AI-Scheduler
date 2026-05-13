# services/sa_service.py
import pandas as pd
import numpy as np
import random
import math
import copy

class SimulatedAnnealingScheduler:
    def __init__(self, teams_df, distances_df, derbies_df):
        self.teams_df = teams_df
        self.distances_df = distances_df
        self.derbies_df = derbies_df
        self.team_names = list(self.teams_df['team_name'])
        self.n = len(self.team_names)
        self.num_rounds = 2 * (self.n - 1)
        
        # Mappings
        self.name_to_id = dict(zip(self.teams_df['team_name'], self.teams_df['team_id']))
        self.id_to_name = dict(zip(self.teams_df['team_id'], self.teams_df['team_name']))
        self.name_to_idx = {name: i for i, name in enumerate(self.team_names)}
        
        # Lookups
        self.dist_matrix = self._build_dist_matrix()
        self.derby_pairs = self._build_derby_pairs()
        self.avoid_rounds = {0, 1, self.num_rounds - 1}

    def _build_dist_matrix(self):
        matrix = np.zeros((self.n, self.n))
        for _, row in self.distances_df.iterrows():
            t1_name = self.id_to_name.get(row['team1_id'])
            t2_name = self.id_to_name.get(row['team2_id'])
            if t1_name and t2_name:
                i, j = self.name_to_idx[t1_name], self.name_to_idx[t2_name]
                matrix[i][j] = matrix[j][i] = row['distance_km']
        return matrix

    def _build_derby_pairs(self):
        pairs = set()
        for _, row in self.derbies_df.iterrows():
            t1_name = self.id_to_name.get(row['team1_id'])
            t2_name = self.id_to_name.get(row['team2_id'])
            if t1_name and t2_name:
                i, j = self.name_to_idx[t1_name], self.name_to_idx[t2_name]
                # Store as index pairs for internal logic
                pairs.add((min(i, j), max(i, j)))
        return pairs

    def _generate_canonical_schedule(self):
        """Circle method logic: produces a valid Double Round-Robin schedule."""
        rounds_per_half = self.n - 1
        teams_idx = list(range(self.n))
        schedule = []

        # First half
        fixed = teams_idx[0]
        rotating = teams_idx[1:]
        for r in range(rounds_per_half):
            round_matches = []
            if r % 2 == 0:
                round_matches.append((fixed, rotating[-1]))
            else:
                round_matches.append((rotating[-1], fixed))
            for k in range(len(rotating)//2):
                home = rotating[k]
                away = rotating[-(k+2)]
                if r % 2 == 0:
                    round_matches.append((home, away))
                else:
                    round_matches.append((away, home))
            schedule.append(round_matches)
            rotating = [rotating[-1]] + rotating[:-1]

        # Second half: reverse home/away
        first_half = copy.deepcopy(schedule)
        for r in range(rounds_per_half):
            mirrored = [(away, home) for (home, away) in first_half[r]]
            schedule.append(mirrored)

        return schedule

    # --- Objective Function Components ---

    def _total_travel(self, schedule):
        total = 0.0
        for t in range(self.n):
            current_loc = t
            for round_matches in schedule:
                for (home, away) in round_matches:
                    if away == t:
                        total += self.dist_matrix[current_loc][home]
                        current_loc = home
                    elif home == t and current_loc != t:
                        total += self.dist_matrix[current_loc][t]
                        current_loc = t
            if current_loc != t:
                total += self.dist_matrix[current_loc][t]
        return total

    def _rest_imbalance(self, schedule, play_days):
        last_played = {t: 0 for t in range(self.n)}
        total_imb = 0
        for r in range(len(schedule)):
            for m_idx, (h, a) in enumerate(schedule[r]):
                day = play_days[r][m_idx]
                imb = abs((day - last_played[h]) - (day - last_played[a]))
                total_imb += imb
                last_played[h] = last_played[a] = day
        return total_imb

    def _count_away_breaks(self, schedule):
        violations = 0
        for t in range(self.n):
            consecutive_away = 0
            for round_matches in schedule:
                played_away = any(away == t for (home, away) in round_matches)
                if played_away:
                    consecutive_away += 1
                    if consecutive_away > 2: violations += 1
                else:
                    consecutive_away = 0
        return violations

    def _derby_penalty(self, schedule):
        penalty = 0
        for r_idx, round_matches in enumerate(schedule):
            if r_idx in self.avoid_rounds:
                for h, a in round_matches:
                    if (min(h, a), max(h, a)) in self.derby_pairs:
                        penalty += 1
        return penalty

    def _calculate_objective(self, schedule, w1=1.0, w2=0.01, w3=5000.0, w4=3000.0):
        slots = [5, 6, 7]
        play_days = []
        for r in range(len(schedule)):
            play_days.append([r * 7 + slots[m_idx % 3] for m_idx in range(len(schedule[r]))])
            
        travel = self._total_travel(schedule)
        imbalance = self._rest_imbalance(schedule, play_days)
        breaks = self._count_away_breaks(schedule)
        derbies = self._derby_penalty(schedule)
        return w1*travel + w2*imbalance + w3*breaks + w4*derbies

    def _get_neighbor(self, schedule):
        new_sched = copy.deepcopy(schedule)
        r1, r2 = random.sample(range(self.num_rounds), 2)
        m1_idx = random.randint(0, (self.n // 2) - 1)
        m2_idx = random.randint(0, (self.n // 2) - 1)
        new_sched[r1][m1_idx], new_sched[r2][m2_idx] = new_sched[r2][m2_idx], new_sched[r1][m1_idx]
        return new_sched

    def _is_valid(self, schedule):
        for r_matches in schedule:
            teams_in_round = set()
            for h, a in r_matches:
                if h in teams_in_round or a in teams_in_round:
                    return False
                teams_in_round.add(h)
                teams_in_round.add(a)
        return True

    def solve(self, initial_temp=1000, cooling_rate=0.995, iterations=5000):
        current_sched = self._generate_canonical_schedule()
        current_cost = self._calculate_objective(current_sched)
        best_sched = copy.deepcopy(current_sched)
        best_cost = current_cost
        
        temp = initial_temp

        for _ in range(iterations):
            neighbor = self._get_neighbor(current_sched)
            if self._is_valid(neighbor):
                neighbor_cost = self._calculate_objective(neighbor)
                delta = neighbor_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_sched = neighbor
                    current_cost = neighbor_cost
                    
                    if current_cost < best_cost:
                        best_sched = copy.deepcopy(current_sched)
                        best_cost = current_cost
            
            temp *= cooling_rate

        formatted_results = []
        for r_idx, round_matches in enumerate(best_sched):
            for h_idx, a_idx in round_matches:
                h_id = self.name_to_id[self.team_names[h_idx]]
                a_id = self.name_to_id[self.team_names[a_idx]]
                formatted_results.append({
                    "round": r_idx + 1,
                    "home_id": int(h_id),
                    "away_id": int(a_id),
                    "is_derby": (min(h_idx, a_idx), max(h_idx, a_idx)) in self.derby_pairs,
                    "distance": self.dist_matrix[h_idx][a_idx]
                })
        
        return formatted_results, self._total_travel(best_sched), best_cost