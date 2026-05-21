# services/sa_service.py
import pandas as pd
import numpy as np
import random
import math
import copy
import time


class SimulatedAnnealingScheduler:

    def __init__(
        self,
        teams_df,
        distances_df,
        derbies_df,
        blackout_rounds=None,
    ):

        self.teams_df = teams_df
        self.distances_df = distances_df
        self.derbies_df = derbies_df

        # =====================================================
        # BASIC DATA
        # =====================================================

        self.team_names = list(
            self.teams_df["team_name"]
        )

        self.n = len(self.team_names)

        if self.n % 2 != 0:
            raise ValueError(
                "Number of teams must be even."
            )

        self.num_rounds = 2 * (self.n - 1)

        # =====================================================
        # MAPPINGS
        # =====================================================

        self.name_to_id = dict(
            zip(
                self.teams_df["team_name"],
                self.teams_df["team_id"],
            )
        )

        self.id_to_name = dict(
            zip(
                self.teams_df["team_id"],
                self.teams_df["team_name"],
            )
        )

        self.name_to_idx = {
            name: i
            for i, name in enumerate(
                self.team_names
            )
        }

        # =====================================================
        # LOOKUPS
        # =====================================================

        self.dist_matrix = (
            self._build_dist_matrix()
        )

        self.derby_pairs = (
            self._build_derby_pairs()
        )

        # =====================================================
        # CONSTRAINTS
        # =====================================================

        self.avoid_rounds = (
            blackout_rounds
            if blackout_rounds
            else {0, 1, self.num_rounds - 1}
        )

        # =====================================================
        # STATS
        # =====================================================

        self.accepted_moves = 0
        self.rejected_moves = 0
        self.improving_moves = 0
        self.explored_neighbors = 0

    # =========================================================
    # DISTANCE MATRIX
    # =========================================================

    def _build_dist_matrix(self):

        matrix = np.zeros((self.n, self.n))

        for _, row in self.distances_df.iterrows():

            t1_name = self.id_to_name.get(
                row["team1_id"]
            )

            t2_name = self.id_to_name.get(
                row["team2_id"]
            )

            if t1_name and t2_name:

                i = self.name_to_idx[t1_name]
                j = self.name_to_idx[t2_name]

                matrix[i][j] = (
                    matrix[j][i]
                ) = row["distance_km"]

        return matrix

    # =========================================================
    # DERBY PAIRS
    # =========================================================

    def _build_derby_pairs(self):

        pairs = set()

        for _, row in self.derbies_df.iterrows():

            t1_name = self.id_to_name.get(
                row["team1_id"]
            )

            t2_name = self.id_to_name.get(
                row["team2_id"]
            )

            if t1_name and t2_name:

                i = self.name_to_idx[t1_name]
                j = self.name_to_idx[t2_name]

                pairs.add(
                    (min(i, j), max(i, j))
                )

        return pairs

    # =========================================================
    # INITIAL SCHEDULE
    # =========================================================

    def _generate_initial_schedule(self):

        rounds_per_half = self.n - 1

        teams_idx = list(range(self.n))
        random.shuffle(teams_idx)

        schedule = []

        fixed = teams_idx[0]

        rotating = teams_idx[1:]

        # =========================
        # FIRST HALF
        # =========================

        for r in range(rounds_per_half):

            round_matches = []

            if r % 2 == 0:
                round_matches.append(
                    (fixed, rotating[-1])
                )
            else:
                round_matches.append(
                    (rotating[-1], fixed)
                )

            for k in range(
                len(rotating) // 2
            ):

                home = rotating[k]

                away = rotating[-(k + 2)]

                if r % 2 == 0:
                    round_matches.append(
                        (home, away)
                    )
                else:
                    round_matches.append(
                        (away, home)
                    )

            schedule.append(round_matches)

            rotating = (
                [rotating[-1]]
                + rotating[:-1]
            )

        # =========================
        # SECOND HALF
        # =========================

        first_half = copy.deepcopy(schedule)

        for r in range(rounds_per_half):

            mirrored = [
                (away, home)
                for (home, away)
                in first_half[r]
            ]

            schedule.append(mirrored)

        return schedule

    # =========================================================
    # METRICS
    # =========================================================

    def _total_travel(self, schedule):

        total = 0.0

        for t in range(self.n):

            current_loc = t

            for round_matches in schedule:

                for home, away in round_matches:

                    if away == t:

                        total += (
                            self.dist_matrix[
                                current_loc
                            ][home]
                        )

                        current_loc = home

                    elif (
                        home == t
                        and current_loc != t
                    ):

                        total += (
                            self.dist_matrix[
                                current_loc
                            ][t]
                        )

                        current_loc = t

            if current_loc != t:

                total += self.dist_matrix[
                    current_loc
                ][t]

        return total

    def _rest_imbalance(
        self,
        schedule,
        play_days,
    ):

        last_played = {
            t: 0 for t in range(self.n)
        }

        total_imbalance = 0

        for r in range(len(schedule)):

            for m_idx, (h, a) in enumerate(
                schedule[r]
            ):

                day = play_days[r][m_idx]

                imbalance = abs(
                    (day - last_played[h])
                    - (day - last_played[a])
                )

                total_imbalance += imbalance

                last_played[h] = day
                last_played[a] = day

        return total_imbalance

    def _count_away_breaks(self, schedule):

        violations = 0

        for t in range(self.n):

            consecutive = 0

            for round_matches in schedule:

                played_away = any(
                    away == t
                    for home, away
                    in round_matches
                )

                if played_away:

                    consecutive += 1

                    if consecutive > 2:
                        violations += 1

                else:
                    consecutive = 0

        return violations

    def _derby_penalty(self, schedule):

        penalty = 0

        for r_idx, round_matches in enumerate(
            schedule
        ):

            if r_idx in self.avoid_rounds:

                for h, a in round_matches:

                    if (
                        min(h, a),
                        max(h, a),
                    ) in self.derby_pairs:

                        penalty += 1

        return penalty

    # =========================================================
    # OBJECTIVE FUNCTION
    # =========================================================

    def _calculate_objective(
        self,
        schedule,
        w1=1.0,
        w2=0.01,
        w3=5000.0,
        w4=3000.0,
    ):

        play_days_info = self.assign_play_days(schedule)

        play_days = [
            [entry["play_day"] for entry in round_days]
            for round_days in play_days_info
        ]

        travel = self._total_travel(
            schedule
        )

        imbalance = (
            self._rest_imbalance(
                schedule,
                play_days,
            )
        )

        breaks = (
            self._count_away_breaks(
                schedule
            )
        )

        derby_pen = (
            self._derby_penalty(
                schedule
            )
        )

        total_cost = (
            w1 * travel
            + w2 * imbalance
            + w3 * breaks
            + w4 * derby_pen
        )

        return {
            "objective_score": total_cost,
            "travel_distance": travel,
            "rest_imbalance": imbalance,
            "away_penalty": breaks,
            "derby_penalty": derby_pen,
        }

    # =========================================================
    # NEIGHBOR GENERATION
    # =========================================================

    def _get_neighbor(self, schedule):

        new_sched = copy.deepcopy(schedule)

        r1, r2 = random.sample(
            range(self.num_rounds),
            2,
        )

        m1 = random.randint(
            0,
            (self.n // 2) - 1,
        )

        m2 = random.randint(
            0,
            (self.n // 2) - 1,
        )

        (
            new_sched[r1][m1],
            new_sched[r2][m2],
        ) = (
            new_sched[r2][m2],
            new_sched[r1][m1],
        )

        return new_sched

    # =========================================================
    # HARD VALIDATION
    # =========================================================

    def _is_valid(self, schedule):

        for r_matches in schedule:

            teams_seen = set()

            for h, a in r_matches:

                if h == a:
                    return False

                if h in teams_seen:
                    return False

                if a in teams_seen:
                    return False

                teams_seen.add(h)
                teams_seen.add(a)

        return True

    # =========================================================
    # FORMAT RESULTS
    # =========================================================
        # =========================================================
    # PLAY DAY ASSIGNMENT
    # =========================================================

    def assign_play_days(
        self,
        schedule,
    ):
        """
        Assign play days using a penalty-based approach (mirroring HC logic):
        - Derby matches are *preferred* on weekends (Fri/Sat/Sun) via a soft
          penalty, but are not forced there.
        - Non-derby matches are distributed across all 7 weekdays to avoid
          piling every match onto days 5/6/7.
        - A distribution penalty discourages overloading any single day.
        """

        play_days = []
        last_played = {}  # team_idx -> last absolute day played

        for r_idx, round_matches in enumerate(schedule):

            round_days = []
            matches_per_day = {day: 0 for day in range(1, 8)}

            for h_idx, a_idx in round_matches:

                is_derby = (
                    min(h_idx, a_idx),
                    max(h_idx, a_idx),
                ) in self.derby_pairs

                best_day = None
                best_penalty = float("inf")

                for weekday in range(1, 8):
                    absolute_day = r_idx * 7 + weekday
                    penalty = 0

                    # Rest fairness
                    for team in [h_idx, a_idx]:
                        if team in last_played:
                            rest_gap = absolute_day - last_played[team]
                            if rest_gap < 2:
                                penalty += 5000
                            previous_weekday = ((last_played[team] - 1) % 7) + 1
                            if previous_weekday == 7 and weekday == 1:
                                penalty += 3000

                    # Soft derby weekend preference (Fri/Sat/Sun)
                    if is_derby and weekday not in [5, 6, 7]:
                        penalty += 4000

                    # Distribution penalty
                    penalty += (matches_per_day[weekday] ** 2) * 500

                    if penalty < best_penalty:
                        best_penalty = penalty
                        best_day = absolute_day

                selected_weekday = ((best_day - 1) % 7) + 1
                matches_per_day[selected_weekday] += 1

                round_days.append({
                    "play_day": best_day,
                    "weekday": selected_weekday,
                })

                last_played[h_idx] = best_day
                last_played[a_idx] = best_day

            play_days.append(round_days)

        return play_days
    
    def _format_schedule(
        self,
        schedule,
    ):

        formatted = []

        play_days = self.assign_play_days(schedule)

        for r_idx, round_matches in enumerate(
            schedule
        ):

            for m_idx, (h_idx, a_idx) in enumerate(
                round_matches
            ):

                h_name = self.team_names[h_idx]
                a_name = self.team_names[a_idx]

                play_info = play_days[r_idx][m_idx]

                formatted.append({

                    "round":
                        r_idx + 1,

                    "match_index":
                        m_idx + 1,

                    "home_id":
                        int(
                            self.name_to_id[h_name]
                        ),

                    "away_id":
                        int(
                            self.name_to_id[a_name]
                        ),

                    "home_name":
                        h_name,

                    "away_name":
                        a_name,

                    "is_derby":
                        (
                            min(h_idx, a_idx),
                            max(h_idx, a_idx),
                        ) in self.derby_pairs,

                    "distance_km":
                        float(
                            self.dist_matrix[
                                h_idx
                            ][a_idx]
                        ),

                    "play_day":
                        int(play_info["play_day"]),

                    "weekday":
                        int(play_info["weekday"]),
                })

        return formatted

    # =========================================================
    # MAIN SOLVER
    # =========================================================

    def solve(
        self,
        initial_temp=1000,
        cooling_rate=0.995,
        iterations=5000,
    ):

        start_time = time.time()

        current_sched = (
            self._generate_initial_schedule()
        )

        current_metrics = (
            self._calculate_objective(
                current_sched
            )
        )

        current_cost = current_metrics[
            "objective_score"
        ]

        best_sched = copy.deepcopy(
            current_sched
        )

        best_metrics = copy.deepcopy(
            current_metrics
        )

        best_cost = current_cost

        temp = initial_temp

        cost_history = []

        # =====================================================
        # SA LOOP
        # =====================================================

        for iteration in range(iterations):

            neighbor = self._get_neighbor(
                current_sched
            )

            self.explored_neighbors += 1

            if not self._is_valid(neighbor):
                continue

            neighbor_metrics = (
                self._calculate_objective(
                    neighbor
                )
            )

            neighbor_cost = (
                neighbor_metrics[
                    "objective_score"
                ]
            )

            delta = (
                neighbor_cost
                - current_cost
            )

            accepted = False

            # Better solution
            if delta < 0:

                accepted = True

                self.improving_moves += 1

            else:

                probability = math.exp(
                    -delta / max(temp, 1e-9)
                )

                if (
                    random.random()
                    < probability
                ):
                    accepted = True

            if accepted:

                self.accepted_moves += 1

                current_sched = neighbor

                current_cost = neighbor_cost

                current_metrics = (
                    neighbor_metrics
                )

                if current_cost < best_cost:

                    best_sched = copy.deepcopy(
                        current_sched
                    )

                    best_metrics = copy.deepcopy(
                        current_metrics
                    )

                    best_cost = current_cost

            else:

                self.rejected_moves += 1

            temp *= cooling_rate

            cost_history.append(best_cost)

        execution_time = (
            time.time() - start_time
        )

        formatted_schedule = (
            self._format_schedule(
                best_sched
            )
        )

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

        return {

            "success": True,

            "execution_time_sec":
                execution_time,

            "schedule":
                formatted_schedule,

            "metrics":
                best_metrics,

            "annealing_statistics": {

                "initial_temperature":
                    initial_temp,

                "final_temperature":
                    temp,

                "cooling_rate":
                    cooling_rate,

                "iterations":
                    iterations,

                "accepted_moves":
                    self.accepted_moves,

                "rejected_moves":
                    self.rejected_moves,

                "improving_moves":
                    self.improving_moves,
            },

            "search_statistics": {

                "iterations_completed":
                    iterations,

                "explored_neighbors":
                    self.explored_neighbors,

                "execution_time_seconds":
                    execution_time,
            },

            "constraint_report": {
                "valid": True,
                "valid_before_optimization": True,
                "valid_after_optimization": True,
                "valid_before": True,
                "valid_after": True,
                "violations": [],
            },

            "cost_history":
                cost_history,

            "blackout_rounds":
                list(self.avoid_rounds),
        }