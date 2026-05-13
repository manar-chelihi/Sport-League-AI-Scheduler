# services/csp_service.py
import pandas as pd
import copy
import time
import sys
from itertools import combinations

# Increase recursion depth for CSP backtracking with large leagues
sys.setrecursionlimit(100000)

class CSPScheduler:
    def __init__(self, teams_df: pd.DataFrame, distances_df: pd.DataFrame, derbies_df: pd.DataFrame):
        """
        Initializes the CSP Solver with data from the database/CSVs.
        """
        # 1. Data Frames
        self.teams_df = teams_df
        self.distances_df = distances_df
        self.derbies_df = derbies_df

        # 2. Basic Mappings and Stats
        self.teams_list = list(self.teams_df['team_name'])
        self.num_teams = len(self.teams_list)
        self.num_rounds = 2 * (self.num_teams - 1)
        
        # Mappings for DB compatibility (Name <-> ID)
        self.name_to_id = dict(zip(self.teams_df['team_name'], self.teams_df['team_id']))
        self.id_to_name = dict(zip(self.teams_df['team_id'], self.teams_df['team_name']))
        
        # 3. Lookups for logic
        self.dist_lookup = self._build_distance_lookup()
        self.derby_matches, self.derby_prefs = self._build_derby_info()
        
        # 4. State
        self.backtrack_calls = 0

    def _build_distance_lookup(self):
        """Creates a (TeamA, TeamB) -> Distance mapping."""
        dist = {}
        for _, row in self.distances_df.iterrows():
            # Handle both name or ID based logic
            t1 = self.id_to_name.get(row['team1_id'], row['team1_id'])
            t2 = self.id_to_name.get(row['team2_id'], row['team2_id'])
            dist[(t1, t2)] = row['distance_km']
            dist[(t2, t1)] = row['distance_km']
        return dist

    def _build_derby_info(self):
        """Builds sets for derby detection and preferred rounds from notebook logic."""
        derbies = []
        prefs = {}
        for _, row in self.derbies_df.iterrows():
            t1 = self.id_to_name.get(row['team1_id'], row['team1_id'])
            t2 = self.id_to_name.get(row['team2_id'], row['team2_id'])
            derbies.extend([(t1, t2), (t2, t1)])
            
            # Logic from Notebook: High priority on rounds 11/22, others on 5-17
            rounds = [11, 22] if row['priority'] == 'High' else list(range(5, 18))
            prefs[(t1, t2)] = rounds
            prefs[(t2, t1)] = rounds
        return set(derbies), prefs

    def _is_consistent(self, assignment, match, round_num):
        """
        Hard Constraint Check (Cell 45):
        - C1: Each team plays at most once per round.
        - C2: Reverse fixture must be in a different round.
        """
        home, away = match
        for (h, a), r in assignment.items():
            if r == round_num:
                if h in (home, away) or a in (home, away):
                    return False
        
        reverse = (away, home)
        if reverse in assignment and assignment[reverse] == round_num:
            return False
        return True

    def _select_mrv_variable(self, variables, assignment, domains):
        """MRV Heuristic: Pick match with the fewest remaining legal rounds."""
        unassigned = [v for v in variables if v not in assignment]
        if not unassigned:
            return None
        return min(unassigned, key=lambda v: len(domains[v]))

    def _forward_check(self, variables, domains, assignment, match, round_num):
        """Prunes domains of related variables to speed up search."""
        new_domains = copy.deepcopy(domains)
        home, away = match
        for var in variables:
            if var in assignment:
                continue
            vh, va = var
            # Remove this round from any match involving these two teams
            if vh in (home, away) or va in (home, away) or var == (away, home):
                if round_num in new_domains[var]:
                    new_domains[var].remove(round_num)
                    if not new_domains[var]:
                        return None # Domain empty: Failure
        return new_domains

    def _backtrack(self, variables, assignment, domains):
        """Recursive backtracking solver."""
        self.backtrack_calls += 1
        
        if len(assignment) == len(variables):
            return assignment

        var = self._select_mrv_variable(variables, assignment, domains)
        if var is None: return None

        # Try rounds in order
        for round_num in domains[var]:
            if self._is_consistent(assignment, var, round_num):
                assignment[var] = round_num
                
                new_doms = self._forward_check(variables, domains, assignment, var, round_num)
                if new_doms is not None:
                    result = self._backtrack(variables, assignment, new_doms)
                    if result is not None:
                        return result
                
                del assignment[var]
        return None

    def _calculate_metrics(self, assignment):
        """Calculates Total Travel Distance and Soft Penalty Score (Cell 47 & 49)."""
        travel = 0
        penalty = 0
        
        # 1. Travel Distance (Sum of away team trips)
        for (home, away), _ in assignment.items():
            d = self.dist_lookup.get((home, away), 0)
            travel += 2 * d
            
        # 2. Penalty: Consecutive Away Games (>2)
        away_by_team = {}
        for (home, away), r in assignment.items():
            away_by_team.setdefault(away, []).append(r)

        for team, away_rounds in away_by_team.items():
            sorted_r = sorted(away_rounds)
            consec = 1
            for i in range(1, len(sorted_r)):
                if sorted_r[i] == sorted_r[i-1] + 1:
                    consec += 1
                    if consec > 2: penalty += 10
                else:
                    consec = 1

        # 3. Penalty: Derby timing
        for match, preferred in self.derby_prefs.items():
            r = assignment.get(match)
            if r is not None and r not in preferred:
                penalty += 5

        return travel, penalty

    def solve(self):
        """
        Main interface: Generates the schedule.
        Returns: (formatted_matches, total_distance, penalty_score)
        """
        # Initialize Variables (All ordered pairs)
        variables = []
        for a, b in combinations(self.teams_list, 2):
            variables.extend([(a, b), (b, a)])
        
        # Initialize Domains (1 to 22)
        domains = {v: list(range(1, self.num_rounds + 1)) for v in variables}
        
        start_time = time.time()
        solution = self._backtrack(variables, {}, domains)
        elapsed = time.time() - start_time

        if solution is None:
            return None, 0, 0

        # Transform to DB Structure
        formatted_results = []
        for (home, away), r in solution.items():
            formatted_results.append({
                "round": r,
                "home_id": self.name_to_id[home],
                "away_id": self.name_to_id[away],
                "is_derby": (home, away) in self.derby_matches
            })

        # Sort matches by round
        formatted_results.sort(key=lambda x: x['round'])
        
        dist, penalty = self._calculate_metrics(solution)
        
        print(f"CSP Finished: {self.backtrack_calls} calls, {elapsed:.2f}s")
        return formatted_results, dist, penalty