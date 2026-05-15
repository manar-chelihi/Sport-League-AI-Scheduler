# services/hc_service.py

import copy
import math
import random
import time
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd


class HillClimbingScheduler:
    """
    Enhanced Hill Climbing Sports League Scheduler.

    Supported constraints:
    ----------------------
    HARD-1:
        Each team plays exactly one match per round.

    HARD-2:
        Every pair of teams meets exactly twice:
            - once home
            - once away

    SOFT-3:
        No team should play more than 2 consecutive away games.

    SOFT-4:
        Derby matches should not occur in blackout rounds.

    Additional Improvements:
    ------------------------
    - Canonical round-robin schedule generation
    - Travel distance optimization
    - Rest fairness balancing
    - Dynamic play-day assignment
    - Multiple neighborhood moves
    - Reciprocal fixture consistency
    - Constraint verification utilities
    - Detailed metrics output
    - Configurable optimization weights
    - Configurable blackout rounds
    - Cost history tracking
    """

    def __init__(
        self,
        teams_df: pd.DataFrame,
        distances_df: pd.DataFrame,
        derbies_df: pd.DataFrame,
        blackout_rounds: Optional[Set[int]] = None,
    ):
        self.teams_df = teams_df.copy()
        self.distances_df = distances_df.copy()
        self.derbies_df = derbies_df.copy()

        self.team_names = list(self.teams_df["team_name"])
        self.n = len(self.team_names)

        if self.n % 2 != 0:
            raise ValueError("Number of teams must be even.")

        self.rounds_per_half = self.n - 1
        self.num_rounds = 2 * self.rounds_per_half

        # Mapping structures
        self.name_to_id = dict(
            zip(self.teams_df["team_name"], self.teams_df["team_id"])
        )

        self.id_to_name = dict(
            zip(self.teams_df["team_id"], self.teams_df["team_name"])
        )

        self.name_to_idx = {
            name: idx for idx, name in enumerate(self.team_names)
        }

        self.idx_to_name = {
            idx: name for idx, name in enumerate(self.team_names)
        }

        # Distances
        self.dist_matrix = self._build_distance_matrix()

        # Derby structures
        self.derby_matches = self._build_derby_pairs_by_id()
        self.derby_pairs_idx = self._build_derby_pairs_by_index()

        # Default blackout rounds:
        # first 2 + last 1
        if blackout_rounds is None:
            self.blackout_rounds = {
                0,
                1,
                self.num_rounds - 1,
            }
        else:
            self.blackout_rounds = blackout_rounds

    # =====================================================================
    # DISTANCE UTILITIES
    # =====================================================================

    @staticmethod
    def haversine(loc1: tuple, loc2: tuple) -> float:
        """
        Great-circle distance in km.
        """
        R = 6371

        lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
        lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        return 2 * R * math.asin(math.sqrt(a))

    def _build_distance_matrix(self) -> np.ndarray:
        """
        Build symmetric distance matrix.
        """
        matrix = np.zeros((self.n, self.n))

        for _, row in self.distances_df.iterrows():
            t1_name = self.id_to_name.get(row["team1_id"])
            t2_name = self.id_to_name.get(row["team2_id"])

            if t1_name and t2_name:
                i = self.name_to_idx[t1_name]
                j = self.name_to_idx[t2_name]

                matrix[i][j] = row["distance_km"]
                matrix[j][i] = row["distance_km"]

        return matrix

    # =====================================================================
    # DERBY UTILITIES
    # =====================================================================

    def _build_derby_pairs_by_id(self) -> Set[Tuple[int, int]]:
        pairs = set()

        for _, row in self.derbies_df.iterrows():
            t1 = int(row["team1_id"])
            t2 = int(row["team2_id"])

            pairs.add((t1, t2))
            pairs.add((t2, t1))

        return pairs

    def _build_derby_pairs_by_index(self) -> Set[Tuple[int, int]]:
        pairs = set()

        for _, row in self.derbies_df.iterrows():
            t1_id = int(row["team1_id"])
            t2_id = int(row["team2_id"])

            t1_name = self.id_to_name[t1_id]
            t2_name = self.id_to_name[t2_id]

            t1_idx = self.name_to_idx[t1_name]
            t2_idx = self.name_to_idx[t2_name]

            pairs.add((min(t1_idx, t2_idx), max(t1_idx, t2_idx)))

        return pairs

    # =====================================================================
    # INITIAL SCHEDULE
    # =====================================================================

    def generate_canonical_schedule(self) -> List[List[Tuple[int, int]]]:
        """
        Circle method double round-robin schedule.

        HARD-1 and HARD-2 are guaranteed by construction.
        """

        n = self.n
        rounds_per_half = n - 1

        teams = list(range(n))

        fixed = teams[0]
        rotating = teams[1:]

        schedule = []

        for r in range(rounds_per_half):
            round_matches = []

            # First pairing
            if r % 2 == 0:
                round_matches.append((fixed, rotating[-1]))
            else:
                round_matches.append((rotating[-1], fixed))

            # Remaining pairings
            for k in range(len(rotating) // 2):
                home = rotating[k]
                away = rotating[-(k + 2)]

                if r % 2 == 0:
                    round_matches.append((home, away))
                else:
                    round_matches.append((away, home))

            schedule.append(round_matches)

            # Rotation
            rotating = [rotating[-1]] + rotating[:-1]

        # Mirror second half
        first_half = copy.deepcopy(schedule)

        for r in range(rounds_per_half):
            mirrored_round = [
                (away, home)
                for (home, away) in first_half[r]
            ]

            schedule.append(mirrored_round)

        return schedule

    # =====================================================================
    # HARD CONSTRAINT VALIDATION
    # =====================================================================

    def verify_hard_constraints(
        self,
        schedule: List[List[Tuple[int, int]]]
    ) -> List[str]:
        """
        Validate HARD-1 and HARD-2.
        """

        violations = []

        # HARD-1
        for r_idx, round_matches in enumerate(schedule):
            seen = set()

            for home, away in round_matches:
                if home in seen:
                    violations.append(
                        f"Round {r_idx+1}: team {home} plays multiple matches"
                    )

                if away in seen:
                    violations.append(
                        f"Round {r_idx+1}: team {away} plays multiple matches"
                    )

                seen.add(home)
                seen.add(away)

            if len(seen) != self.n:
                violations.append(
                    f"Round {r_idx+1}: not all teams scheduled"
                )

        # HARD-2
        pair_count = {}

        for round_matches in schedule:
            for home, away in round_matches:
                pair_count[(home, away)] = (
                    pair_count.get((home, away), 0) + 1
                )

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue

                count = pair_count.get((i, j), 0)

                if count != 1:
                    violations.append(
                        f"Fixture ({i},{j}) appears {count} times"
                    )

        return violations

    def is_valid(self, schedule) -> bool:
        return len(self.verify_hard_constraints(schedule)) == 0

    # =====================================================================
    # PLAY DAY ASSIGNMENT
    # =====================================================================

    def assign_play_days(
        self,
        schedule: List[List[Tuple[int, int]]]
    ) -> List[List[int]]:
        """
        Assign play days while balancing:
        - derby priority
        - recovery time
        - weekday distribution
        """

        play_days = []
        last_played = {}

        for r_idx, round_matches in enumerate(schedule):
            round_days = []

            matches_per_day = {
                day: 0 for day in range(1, 8)
            }

            for home, away in round_matches:

                is_derby = (
                    min(home, away),
                    max(home, away)
                ) in self.derby_pairs_idx

                best_day = None
                best_penalty = float("inf")

                for weekday in range(1, 8):
                    absolute_day = r_idx * 7 + weekday
                    penalty = 0

                    # Rest fairness
                    for team in [home, away]:
                        if team in last_played:
                            rest_gap = absolute_day - last_played[team]

                            if rest_gap < 2:
                                penalty += 5000

                            previous_weekday = (
                                ((last_played[team] - 1) % 7) + 1
                            )

                            # Sunday -> Monday transition
                            if previous_weekday == 7 and weekday == 1:
                                penalty += 3000

                    # Derby weekend preference
                    if is_derby and weekday not in [5, 6, 7]:
                        penalty += 4000

                    # Match distribution penalty
                    penalty += (
                        matches_per_day[weekday] ** 2
                    ) * 500

                    if penalty < best_penalty:
                        best_penalty = penalty
                        best_day = absolute_day

                round_days.append(best_day)

                selected_weekday = ((best_day - 1) % 7) + 1
                matches_per_day[selected_weekday] += 1

                last_played[home] = best_day
                last_played[away] = best_day

            play_days.append(round_days)

        return play_days

    # =====================================================================
    # OBJECTIVE COMPONENTS
    # =====================================================================

    def total_travel(self, schedule) -> float:
        """
        Traveling Tournament Problem style travel model.
        """

        total = 0.0

        for team in range(self.n):
            current_location = team

            for round_matches in schedule:
                for home, away in round_matches:

                    if away == team:
                        total += self.dist_matrix[current_location][home]
                        current_location = home

                    elif home == team and current_location != team:
                        total += self.dist_matrix[current_location][team]
                        current_location = team

            # Return home after season
            if current_location != team:
                total += self.dist_matrix[current_location][team]

        return total

    def rest_imbalance(self, schedule, play_days) -> float:
        last_played = {
            team: 0 for team in range(self.n)
        }

        imbalance = 0

        for r_idx in range(len(schedule)):
            for m_idx, (home, away) in enumerate(schedule[r_idx]):

                day = play_days[r_idx][m_idx]

                home_rest = day - last_played[home]
                away_rest = day - last_played[away]

                imbalance += abs(home_rest - away_rest)

                last_played[home] = day
                last_played[away] = day

        return imbalance

    def count_away_breaks(self, schedule) -> int:
        """
        Count violations of:
            no more than 2 consecutive away games.
        """

        violations = 0

        for team in range(self.n):
            streak = 0

            for round_matches in schedule:
                played_away = any(
                    away == team
                    for home, away in round_matches
                )

                if played_away:
                    streak += 1

                    if streak > 2:
                        violations += 1
                else:
                    streak = 0

        return violations

    def derby_penalty(self, schedule) -> int:
        """
        Penalize derby matches in blackout rounds.
        """

        penalty = 0

        for r_idx, round_matches in enumerate(schedule):
            if r_idx in self.blackout_rounds:
                for home, away in round_matches:
                    if (
                        min(home, away),
                        max(home, away)
                    ) in self.derby_pairs_idx:
                        penalty += 1

        return penalty

    # =====================================================================
    # OBJECTIVE FUNCTION
    # =====================================================================

    def calculate_objective(
        self,
        schedule,
        play_days,
        w1=1.0,
        w2=0.01,
        w3=5000.0,
        w4=3000.0,
    ):
        """
        Objective:

        f(S) =
            w1 * travel
          + w2 * rest imbalance
          + w3 * away breaks
          + w4 * derby violations
        """

        travel = self.total_travel(schedule)
        imbalance = self.rest_imbalance(schedule, play_days)
        away_breaks = self.count_away_breaks(schedule)
        derby_violations = self.derby_penalty(schedule)

        return (
            w1 * travel
            + w2 * imbalance
            + w3 * away_breaks
            + w4 * derby_violations
        )

    # =====================================================================
    # NEIGHBOR GENERATION
    # =====================================================================

    def _round_swap_move(self, schedule):
        """
        Swap two rounds.

        HARD-1 and HARD-2 remain valid.
        """

        neighbor = copy.deepcopy(schedule)

        r1, r2 = random.sample(
            range(self.num_rounds),
            2
        )

        neighbor[r1], neighbor[r2] = (
            neighbor[r2],
            neighbor[r1],
        )

        return neighbor

    def _home_away_flip_move(self, schedule):
        """
        Flip home/away for one fixture AND its reciprocal fixture.

        Preserves HARD-2.
        """

        neighbor = copy.deepcopy(schedule)

        round_idx = random.randint(0, self.num_rounds - 1)
        match_idx = random.randint(0, (self.n // 2) - 1)

        home, away = neighbor[round_idx][match_idx]

        # Flip selected fixture
        neighbor[round_idx][match_idx] = (away, home)

        # Find reciprocal fixture
        reciprocal_found = False

        for r in range(self.num_rounds):
            for m in range(len(neighbor[r])):
                h, a = neighbor[r][m]

                if h == away and a == home:
                    neighbor[r][m] = (home, away)
                    reciprocal_found = True
                    break

            if reciprocal_found:
                break

        return neighbor

    def get_neighbor(self, schedule):
        """
        Random neighborhood move.
        """

        if random.random() < 0.5:
            return self._round_swap_move(schedule)

        return self._home_away_flip_move(schedule)

    # =====================================================================
    # HILL CLIMBING
    # =====================================================================

    def hill_climbing(
        self,
        initial_schedule,
        max_iterations=None,
        w1=1.0,
        w2=0.01,
        w3=5000.0,
        w4=3000.0,
    ):
        """
        Hill climbing optimization.
        """

        if max_iterations is None:
            max_iterations = self.n ** 2 * 50

        patience = max(250, self.n * 20)

        current = copy.deepcopy(initial_schedule)

        play_days = self.assign_play_days(current)

        current_cost = self.calculate_objective(
            current,
            play_days,
            w1,
            w2,
            w3,
            w4,
        )

        best_schedule = copy.deepcopy(current)
        best_cost = current_cost

        no_improvement_count = 0

        cost_history = [current_cost]

        for iteration in range(max_iterations):

            neighbor = self.get_neighbor(current)

            if not self.is_valid(neighbor):
                continue

            neighbor_play_days = self.assign_play_days(neighbor)

            neighbor_cost = self.calculate_objective(
                neighbor,
                neighbor_play_days,
                w1,
                w2,
                w3,
                w4,
            )

            # Accept improving solution
            if neighbor_cost < current_cost:
                current = neighbor
                current_cost = neighbor_cost

                no_improvement_count = 0

                if neighbor_cost < best_cost:
                    best_schedule = copy.deepcopy(neighbor)
                    best_cost = neighbor_cost
            else:
                no_improvement_count += 1

            cost_history.append(current_cost)

            if no_improvement_count >= patience:
                break

        return best_schedule, best_cost, cost_history

    # =====================================================================
    # SOLVER API
    # =====================================================================

    def solve(
        self,
        max_iterations=2000,
        w1=1.0,
        w2=0.01,
        w3=5000.0,
        w4=3000.0,
    ):
        """
        Main public API.
        """

        start_time = time.time()

        initial_schedule = self.generate_canonical_schedule()

        violations_before = self.verify_hard_constraints(
            initial_schedule
        )

        optimized_schedule, final_cost, cost_history = (
            self.hill_climbing(
                initial_schedule,
                max_iterations=max_iterations,
                w1=w1,
                w2=w2,
                w3=w3,
                w4=w4,
            )
        )

        violations_after = self.verify_hard_constraints(
            optimized_schedule
        )

        play_days = self.assign_play_days(optimized_schedule)

        formatted_results = []

        for r_idx, round_matches in enumerate(optimized_schedule):
            for m_idx, (home_idx, away_idx) in enumerate(round_matches):

                home_name = self.idx_to_name[home_idx]
                away_name = self.idx_to_name[away_idx]

                home_id = int(self.name_to_id[home_name])
                away_id = int(self.name_to_id[away_name])

                weekday = ((play_days[r_idx][m_idx] - 1) % 7) + 1

                formatted_results.append({
                    "round": r_idx + 1,
                    "match_index": m_idx + 1,
                    "home_id": home_id,
                    "home_name": home_name,
                    "away_id": away_id,
                    "away_name": away_name,
                    "is_derby": (
                        min(home_idx, away_idx),
                        max(home_idx, away_idx)
                    ) in self.derby_pairs_idx,
                    "distance_km": float(
                        self.dist_matrix[home_idx][away_idx]
                    ),
                    "play_day": int(play_days[r_idx][m_idx]),
                    "weekday": weekday,
                })

        execution_time = time.time() - start_time

        metrics = {
            "total_travel_km": float(
                self.total_travel(optimized_schedule)
            ),
            "rest_imbalance": float(
                self.rest_imbalance(
                    optimized_schedule,
                    play_days,
                )
            ),
            "away_break_violations": int(
                self.count_away_breaks(optimized_schedule)
            ),
            "derby_blackout_violations": int(
                self.derby_penalty(optimized_schedule)
            ),
            "objective_cost": float(final_cost),
            "execution_time_seconds": execution_time,
            "iterations_recorded": len(cost_history),
        }

        constraint_report = {
            "valid_before_optimization": len(violations_before) == 0,
            "valid_after_optimization": len(violations_after) == 0,
            "violations_before": violations_before,
            "violations_after": violations_after,
        }

        return {
            "schedule": formatted_results,
            "metrics": metrics,
            "constraint_report": constraint_report,
            "cost_history": cost_history,
        }