# services/greedy_service.py

import math
import random
import time

import networkx as nx
import pandas as pd

from collections import defaultdict
from itertools import combinations


class GreedyScheduler:

    def __init__(
        self,
        teams_df,
        distances_df,
        derbies_df,
        blackout_rounds=None,
    ):

        self.teams_df = teams_df.copy()
        self.distances_df = distances_df.copy()
        self.derbies_df = derbies_df.copy()

        # =====================================================
        # TEAM DATA
        # =====================================================

        self.team_ids = sorted(
            list(self.teams_df["team_id"])
        )

        self.n = len(self.team_ids)

        if self.n % 2 != 0:
            raise ValueError(
                "Number of teams must be even."
            )

        self.num_rounds = (
            2 * (self.n - 1)
        )

        self.id_to_name = dict(
            zip(
                self.teams_df["team_id"],
                self.teams_df["team_name"],
            )
        )

        self.teams_info = {

            row["team_id"]: {

                "name":
                    row["team_name"],

                "coords":
                    (
                        row["latitude"],
                        row["longitude"],
                    ),
            }

            for _, row in (
                self.teams_df.iterrows()
            )
        }

        # =====================================================
        # DISTANCES
        # =====================================================

        self.dist_matrix = (
            self._build_distance_matrix()
        )

        # =====================================================
        # DERBIES
        # =====================================================

        self.derbies_bonus = (
            self._build_derby_bonus()
        )

        # =====================================================
        # CONSTRAINTS
        # =====================================================

        self.blackout_rounds = (
            blackout_rounds
            if blackout_rounds
            else {
                1,
                2,
                self.num_rounds,
            }
        )

        # =====================================================
        # SEARCH STATS
        # =====================================================

        self.deadlock_count = 0

        self.restart_count = 0

        self.explored_matchings = 0

        self.generated_neighbors = 0

        self.constraint_violations = []

        self.reset()

    # =========================================================
    # HAVERSINE DISTANCE
    # =========================================================

    def _haversine(
        self,
        coord1,
        coord2,
    ):

        lat1, lon1 = coord1

        lat2, lon2 = coord2

        R = 6371

        dlat = math.radians(
            lat2 - lat1
        )

        dlon = math.radians(
            lon2 - lon1
        )

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(
                math.radians(lat1)
            )
            * math.cos(
                math.radians(lat2)
            )
            * math.sin(dlon / 2) ** 2
        )

        return (
            2
            * R
            * math.asin(
                math.sqrt(a)
            )
        )

    # =========================================================
    # DISTANCE MATRIX
    # =========================================================

    def _build_distance_matrix(self):

        dist = {}

        for _, row in (
            self.distances_df.iterrows()
        ):

            t1 = int(row["team1_id"])

            t2 = int(row["team2_id"])

            d = float(
                row["distance_km"]
            )

            dist[(t1, t2)] = d

            dist[(t2, t1)] = d

        # Fill missing distances
        for t1 in self.team_ids:

            for t2 in self.team_ids:

                if t1 == t2:

                    dist[(t1, t2)] = 0

                elif (t1, t2) not in dist:

                    coord1 = self.teams_info[
                        t1
                    ]["coords"]

                    coord2 = self.teams_info[
                        t2
                    ]["coords"]

                    d = self._haversine(
                        coord1,
                        coord2,
                    )

                    dist[(t1, t2)] = d

                    dist[(t2, t1)] = d

        return dist

    # =========================================================
    # DERBY BONUS
    # =========================================================

    def _build_derby_bonus(self):

        bonus = {}

        for _, row in (
            self.derbies_df.iterrows()
        ):

            t1 = int(row["team1_id"])

            t2 = int(row["team2_id"])

            pair = (
                min(t1, t2),
                max(t1, t2),
            )

            derby_bonus = (
                1000
                if row["priority"]
                == "High"
                else 500
            )

            bonus[pair] = derby_bonus

        return bonus

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.current_locations = {

            tid: tid
            for tid in self.team_ids
        }

        self.played = defaultdict(int)

        self.away_streak = {

            tid: 0
            for tid in self.team_ids
        }

        self.history = []

        self._last_played_day = {}

    # =========================================================
    # ROUND GENERATION
    # =========================================================

    def _generate_round(self):

        G = nx.Graph()

        G.add_nodes_from(
            self.team_ids
        )

        for a, b in combinations(
            self.team_ids,
            2,
        ):

            pair = (
                min(a, b),
                max(a, b),
            )

            # =====================
            # DOUBLE ROUND ROBIN
            # =====================

            if self.played[pair] >= 2:
                continue

            # =====================
            # TRAVEL COST
            # =====================

            dist_a = (
                self.dist_matrix[
                    (
                        self.current_locations[a],
                        b,
                    )
                ]
            )

            dist_b = (
                self.dist_matrix[
                    (
                        self.current_locations[b],
                        a,
                    )
                ]
            )

            # =====================
            # AWAY STREAK PENALTY
            # =====================

            p_a = (
                25000
                if self.away_streak[a]
                >= 2
                else 0
            )

            p_b = (
                25000
                if self.away_streak[b]
                >= 2
                else 0
            )

            # =====================
            # DERBY BONUS
            # =====================

            derby_bonus = (
                self.derbies_bonus.get(
                    pair,
                    0,
                )
            )

            # =====================
            # OBJECTIVE
            # =====================

            weight = (

                derby_bonus

                - (
                    dist_a + dist_b
                ) / 2

                - p_a

                - p_b

            )

            # randomness
            weight += random.gauss(
                0,
                45,
            )

            G.add_edge(
                a,
                b,
                weight=weight,
            )

        try:

            matching = (
                nx.max_weight_matching(
                    G,
                    weight="weight",
                    maxcardinality=True,
                )
            )

            self.explored_matchings += 1

            return (
                list(matching)
                if len(matching)
                == (self.n // 2)
                else None
            )

        except Exception:

            return None

    # =========================================================
    # HARD VALIDATION
    # =========================================================

    def _validate_round(
        self,
        matches,
    ):

        seen = set()

        for a, b in matches:

            if a == b:

                self.constraint_violations.append(
                    "Self-play detected."
                )

                return False

            if a in seen:

                self.constraint_violations.append(
                    f"Duplicate team {a}."
                )

                return False

            if b in seen:

                self.constraint_violations.append(
                    f"Duplicate team {b}."
                )

                return False

            seen.add(a)

            seen.add(b)

        return True

    # =========================================================
    # FORMAT SCHEDULE
    # =========================================================

    def _format_schedule(self):

        formatted = []

        for idx, (
            round_num,
            away,
            home,
            dist,
            play_day,
            weekday,
        ) in enumerate(
            self.history
        ):

            formatted.append({

                "round":
                    round_num,

                "match_index":
                    idx + 1,

                "home_id":
                    int(home),

                "away_id":
                    int(away),

                "home_name":
                    self.id_to_name[home],

                "away_name":
                    self.id_to_name[away],

                "is_derby":
                    (
                        min(home, away),
                        max(home, away),
                    )
                    in self.derbies_bonus,

                "distance_km":
                    float(dist),

                "play_day":
                    int(play_day),

                "weekday":
                    int(weekday),
            })

        return formatted

    # =========================================================
    # METRICS
    # =========================================================

    def _calculate_rest_imbalance(self):

        last_played = {
            tid: 0 for tid in self.team_ids
        }

        imbalance = 0

        for (
            round_num,
            away,
            home,
            _,
            play_day,
            weekday,
        ) in self.history:

            home_rest = play_day - last_played[home]
            away_rest = play_day - last_played[away]

            imbalance += abs(home_rest - away_rest)

            last_played[home] = play_day
            last_played[away] = play_day

        return float(imbalance)

    def _calculate_metrics(
        self,
        total_distance,
    ):

        derby_penalty = 0

        away_penalty = 0

        for tid in self.team_ids:

            if self.away_streak[tid] > 2:

                away_penalty += (
                    self.away_streak[tid]
                    - 2
                )

        for (
            round_num,
            away,
            home,
            _,
            play_day,
            weekday,
        ) in self.history:

            pair = (
                min(home, away),
                max(home, away),
            )

            if (
                pair
                in self.derbies_bonus
                and round_num
                in self.blackout_rounds
            ):

                derby_penalty += 1

        rest_imbalance = self._calculate_rest_imbalance()

        objective = (

            total_distance

            + derby_penalty * 5000

            + away_penalty * 3000
        )

        return {

            "travel_distance":
                total_distance,

            "away_penalty":
                away_penalty,

            "derby_penalty":
                derby_penalty,

            "rest_imbalance":
                rest_imbalance,

            "objective_score":
                objective,
        }

    # =========================================================
    # MAIN SOLVER
    # =========================================================

    def solve(self):

        start_time = time.time()

        attempt = 0

        while True:

            attempt += 1

            self.restart_count += 1

            self.reset()

            success = True

            total_distance = 0

            for r in range(
                1,
                self.num_rounds + 1,
            ):

                matches = (
                    self._generate_round()
                )

                if (
                    not matches
                    or not self._validate_round(
                        matches
                    )
                ):

                    self.deadlock_count += 1

                    success = False

                    break

                # =====================
                # MATCH PROCESSING
                # =====================

                # Per-round day-distribution tracker
                matches_per_day = {day: 0 for day in range(1, 8)}

                for a, b in matches:

                    # Away streak logic
                    if (
                        self.away_streak[a]
                        >= 2
                    ):

                        away = b
                        home = a

                    elif (
                        self.away_streak[b]
                        >= 2
                    ):

                        away = a
                        home = b

                    else:

                        d_a = (
                            self.dist_matrix[
                                (
                                    self.current_locations[a],
                                    b,
                                )
                            ]
                        )

                        d_b = (
                            self.dist_matrix[
                                (
                                    self.current_locations[b],
                                    a,
                                )
                            ]
                        )

                        if d_a <= d_b:

                            away = a
                            home = b

                        else:

                            away = b
                            home = a

                    dist = (
                        self.dist_matrix[
                            (
                                self.current_locations[
                                    away
                                ],
                                home,
                            )
                        ]
                    )

                    total_distance += dist

                    # =====================
                    # PLAY DAY ASSIGNMENT
                    # Soft derby weekend preference
                    # (mirrors HC logic)
                    # =====================

                    pair = (
                        min(away, home),
                        max(away, home),
                    )

                    is_derby = pair in self.derbies_bonus

                    best_weekday = None
                    best_penalty = float("inf")

                    for wd in range(1, 8):
                        absolute_day = ((r - 1) * 7) + wd
                        penalty = 0

                        # Rest fairness
                        for team in [home, away]:
                            if team in self._last_played_day:
                                rest_gap = absolute_day - self._last_played_day[team]
                                if rest_gap < 2:
                                    penalty += 5000
                                prev_wd = ((self._last_played_day[team] - 1) % 7) + 1
                                if prev_wd == 7 and wd == 1:
                                    penalty += 3000

                        # Soft derby weekend preference (Fri/Sat/Sun)
                        if is_derby and wd not in [5, 6, 7]:
                            penalty += 4000

                        # Distribution penalty
                        penalty += (matches_per_day[wd] ** 2) * 500

                        if penalty < best_penalty:
                            best_penalty = penalty
                            best_weekday = wd

                    weekday = best_weekday
                    play_day = ((r - 1) * 7) + weekday

                    matches_per_day[weekday] += 1
                    self._last_played_day[home] = play_day
                    self._last_played_day[away] = play_day

                    # =====================
                    # UPDATE STATE
                    # =====================

                    self.history.append(
                        (
                            r,
                            away,
                            home,
                            dist,
                            play_day,
                            weekday,
                        )
                    )

                    pair = (
                        min(away, home),
                        max(away, home),
                    )

                    self.played[pair] += 1

                    self.current_locations[
                        away
                    ] = home

                    self.away_streak[
                        away
                    ] += 1

                    self.away_streak[
                        home
                    ] = 0

            if success:

                break

        execution_time = (
            time.time() - start_time
        )

        metrics = (
            self._calculate_metrics(
                total_distance
            )
        )

        formatted_schedule = (
            self._format_schedule()
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
                metrics,

            "search_statistics": {

                "execution_time_seconds":
                    execution_time,

                "restart_count":
                    self.restart_count,

                "deadlock_count":
                    self.deadlock_count,

                "explored_matchings":
                    self.explored_matchings,
            },

            "constraint_report": {

                "valid":
                    len(
                        self.constraint_violations
                    )
                    == 0,
                "valid_before_optimization":
                    len(
                        self.constraint_violations
                    )
                    == 0,
                "valid_after_optimization":
                    len(
                        self.constraint_violations
                    )
                    == 0,
                "valid_before":
                    len(
                        self.constraint_violations
                    )
                    == 0,
                "valid_after":
                    len(
                        self.constraint_violations
                    )
                    == 0,

                "violations":
                    self.constraint_violations,
            },

            "blackout_rounds":
                list(
                    self.blackout_rounds
                ),
        }