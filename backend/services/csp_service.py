# services/csp_service.py

import copy
import sys
import time
from itertools import combinations
from typing import Dict, List, Tuple, Set, Optional

import pandas as pd


# Increase recursion limit for deep CSP search trees
sys.setrecursionlimit(100000)


class CSPScheduler:
    """
    Enhanced CSP-based Sports League Scheduler.

    Features:
    ----------
    - Backtracking CSP solver
    - MRV heuristic
    - Forward checking
    - Hard constraint verification
    - Derby preference handling
    - Consecutive away-game control
    - Rest fairness metrics
    - Distance optimization metrics
    - Configurable blackout rounds
    - Detailed schedule output
    - Search statistics
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

        # Teams
        self.teams_list = list(self.teams_df["team_name"])

        self.num_teams = len(self.teams_list)

        if self.num_teams % 2 != 0:
            raise ValueError(
                "Number of teams must be even."
            )

        self.num_rounds = 2 * (self.num_teams - 1)

        # ID mappings
        self.name_to_id = dict(
            zip(
                self.teams_df["team_name"],
                self.teams_df["team_id"]
            )
        )

        self.id_to_name = dict(
            zip(
                self.teams_df["team_id"],
                self.teams_df["team_name"]
            )
        )

        # Distances
        self.dist_lookup = self._build_distance_lookup()

        # Derby info
        (
            self.derby_matches,
            self.derby_preferences
        ) = self._build_derby_info()

        # Blackout rounds
        if blackout_rounds is None:
            self.blackout_rounds = {
                1,
                2,
                self.num_rounds,
            }
        else:
            self.blackout_rounds = blackout_rounds

        # Search stats
        self.backtrack_calls = 0
        self.constraint_checks = 0
        self.forward_check_failures = 0

    # ============================================================
    # DISTANCE LOOKUP
    # ============================================================

    def _build_distance_lookup(self):
        """
        Creates:
            (TeamA, TeamB) -> distance_km
        """

        lookup = {}

        for _, row in self.distances_df.iterrows():

            t1 = self.id_to_name.get(
                row["team1_id"],
                row["team1_id"]
            )

            t2 = self.id_to_name.get(
                row["team2_id"],
                row["team2_id"]
            )

            d = float(row["distance_km"])

            lookup[(t1, t2)] = d
            lookup[(t2, t1)] = d

        return lookup

    # ============================================================
    # DERBY PROCESSING
    # ============================================================

    def _build_derby_info(self):
        """
        Build derby structures and preferred rounds.
        """

        derby_matches = set()
        derby_preferences = {}

        for _, row in self.derbies_df.iterrows():

            t1 = self.id_to_name.get(
                row["team1_id"],
                row["team1_id"]
            )

            t2 = self.id_to_name.get(
                row["team2_id"],
                row["team2_id"]
            )

            derby_matches.add((t1, t2))
            derby_matches.add((t2, t1))

            priority = str(
                row.get("priority", "Medium")
            ).lower()

            if priority == "high":
                preferred_rounds = [
                    self.num_rounds // 2,
                    self.num_rounds,
                ]

            elif priority == "medium":
                preferred_rounds = list(
                    range(5, 18)
                )

            else:
                preferred_rounds = list(
                    range(3, self.num_rounds)
                )

            derby_preferences[(t1, t2)] = preferred_rounds
            derby_preferences[(t2, t1)] = preferred_rounds

        return derby_matches, derby_preferences

    # ============================================================
    # HARD CONSTRAINTS
    # ============================================================

    def _is_consistent(
        self,
        assignment,
        match,
        round_num,
    ):
        """
        HARD CONSTRAINTS

        C1:
            Each team plays at most once per round.

        C2:
            Reverse fixture cannot occur
            in the same round.

        C3:
            A fixture appears only once.

        C4:
            No self-play.
        """

        self.constraint_checks += 1

        home, away = match

        # C4
        if home == away:
            return False

        # C1
        for (h, a), r in assignment.items():

            if r != round_num:
                continue

            if h in (home, away):
                return False

            if a in (home, away):
                return False

        # C2
        reverse = (away, home)

        if reverse in assignment:
            if assignment[reverse] == round_num:
                return False

        # C3
        if match in assignment:
            return False

        return True

    # ============================================================
    # MRV HEURISTIC
    # ============================================================

    def _select_mrv_variable(
        self,
        variables,
        assignment,
        domains,
    ):
        """
        Minimum Remaining Values heuristic.
        """

        unassigned = [
            v for v in variables
            if v not in assignment
        ]

        if not unassigned:
            return None

        return min(
            unassigned,
            key=lambda v: len(domains[v])
        )

    # ============================================================
    # FORWARD CHECKING
    # ============================================================

    def _forward_check(
        self,
        variables,
        domains,
        assignment,
        match,
        round_num,
    ):
        """
        Forward checking:
        Remove invalid rounds from neighbors.
        """

        new_domains = copy.deepcopy(domains)

        home, away = match

        for var in variables:

            if var in assignment:
                continue

            vh, va = var

            related = (
                vh in (home, away)
                or va in (home, away)
                or var == (away, home)
            )

            if related:

                if round_num in new_domains[var]:
                    new_domains[var].remove(round_num)

                if not new_domains[var]:
                    self.forward_check_failures += 1
                    return None

        return new_domains

    # ============================================================
    # BACKTRACKING SEARCH
    # ============================================================

    def _backtrack(
        self,
        variables,
        assignment,
        domains,
    ):
        """
        Recursive CSP solver.
        """

        self.backtrack_calls += 1

        # Complete assignment
        if len(assignment) == len(variables):
            return assignment

        # MRV
        var = self._select_mrv_variable(
            variables,
            assignment,
            domains,
        )

        if var is None:
            return assignment

        # Least constraining order
        ordered_rounds = sorted(domains[var])

        for round_num in ordered_rounds:

            if self._is_consistent(
                assignment,
                var,
                round_num,
            ):

                assignment[var] = round_num

                new_domains = self._forward_check(
                    variables,
                    domains,
                    assignment,
                    var,
                    round_num,
                )

                if new_domains is not None:

                    result = self._backtrack(
                        variables,
                        assignment,
                        new_domains,
                    )

                    if result is not None:
                        return result

                del assignment[var]

        return None

    # ============================================================
    # HARD VALIDATION
    # ============================================================

    def verify_constraints(self, assignment):
        """
        Validate full schedule integrity.
        """

        violations = []

        rounds = {}

        for match, r in assignment.items():
            rounds.setdefault(r, []).append(match)

        # Per round validation
        for r, matches in rounds.items():

            teams_seen = set()

            for home, away in matches:

                if home == away:
                    violations.append(
                        f"Self-play in round {r}"
                    )

                if home in teams_seen:
                    violations.append(
                        f"{home} duplicated in round {r}"
                    )

                if away in teams_seen:
                    violations.append(
                        f"{away} duplicated in round {r}"
                    )

                teams_seen.add(home)
                teams_seen.add(away)

        # Reverse fixtures
        for home, away in assignment:

            reverse = (away, home)

            if reverse not in assignment:
                violations.append(
                    f"Missing reverse fixture for "
                    f"{home} vs {away}"
                )

        return violations

    # ============================================================
    # METRICS
    # ============================================================

    def _calculate_travel_distance(
        self,
        assignment,
    ):
        """
        Sum away-team travel.
        """

        total = 0.0

        for (home, away), _ in assignment.items():

            d = self.dist_lookup.get(
                (home, away),
                0
            )

            total += 2 * d

        return total

    def _calculate_away_penalty(
        self,
        assignment,
    ):
        """
        Penalize >2 consecutive away games.
        """

        penalty = 0

        away_by_team = {}

        for (home, away), r in assignment.items():
            away_by_team.setdefault(
                away,
                []
            ).append(r)

        for team, rounds in away_by_team.items():

            rounds = sorted(rounds)

            streak = 1

            for i in range(1, len(rounds)):

                if rounds[i] == rounds[i - 1] + 1:

                    streak += 1

                    if streak > 2:
                        penalty += 10

                else:
                    streak = 1

        return penalty

    def _calculate_derby_penalty(
        self,
        assignment,
    ):
        """
        Penalize derbies outside preferred rounds.
        """

        penalty = 0

        for match, preferred_rounds in (
            self.derby_preferences.items()
        ):

            assigned_round = assignment.get(match)

            if assigned_round is None:
                continue

            if assigned_round not in preferred_rounds:
                penalty += 5

            if assigned_round in self.blackout_rounds:
                penalty += 20

        return penalty

    def _calculate_rest_imbalance(
        self,
        assignment,
    ):
        """
        Rest fairness metric.
        """

        matches_by_round = {}

        for match, r in assignment.items():
            matches_by_round.setdefault(
                r,
                []
            ).append(match)

        last_played = {
            t: 0 for t in self.teams_list
        }

        imbalance = 0

        for r in sorted(matches_by_round.keys()):

            for home, away in matches_by_round[r]:

                home_rest = r - last_played[home]
                away_rest = r - last_played[away]

                imbalance += abs(
                    home_rest - away_rest
                )

                last_played[home] = r
                last_played[away] = r

        return imbalance

    def calculate_metrics(
        self,
        assignment,
    ):
        """
        Full metrics report.
        """

        travel = self._calculate_travel_distance(
            assignment
        )

        away_penalty = self._calculate_away_penalty(
            assignment
        )

        derby_penalty = self._calculate_derby_penalty(
            assignment
        )

        rest_imbalance = (
            self._calculate_rest_imbalance(
                assignment
            )
        )

        objective = (
            travel
            + away_penalty
            + derby_penalty
            + rest_imbalance
        )

        return {
            "travel_distance_km": travel,
            "away_penalty_violations": away_penalty,
            "derby_penalty": derby_penalty,
            "rest_imbalance": rest_imbalance,
            "objective_score": objective,
        }

    # ============================================================
    # SCHEDULE FORMATTER
    # ============================================================
        # ============================================================
    # PLAY DAY ASSIGNMENT
    # ============================================================

    def assign_play_days(self, assignment):
        """
        Assign play days using a penalty-based approach (mirroring HC logic):
        - Derby matches are *preferred* on weekends (Fri/Sat/Sun) via a soft
          penalty, but are not forced there.
        - Non-derby matches are distributed across all 7 weekdays to avoid
          piling every match onto days 5/6/7.
        - A distribution penalty discourages overloading any single day.
        """

        rounds = {}
        for (home, away), round_num in assignment.items():
            rounds.setdefault(round_num, []).append((home, away))

        play_day_map = {}
        last_played = {}  # team -> last absolute day played

        for round_num in sorted(rounds.keys()):

            matches = rounds[round_num]
            matches_per_day = {day: 0 for day in range(1, 8)}

            for home, away in matches:

                is_derby = (
                    (home, away) in self.derby_matches
                    or (away, home) in self.derby_matches
                )

                best_day = None
                best_penalty = float("inf")

                for weekday in range(1, 8):
                    absolute_day = ((round_num - 1) * 7) + weekday
                    penalty = 0

                    # Rest fairness: discourage < 2 days rest and
                    # awkward Sunday->Monday transitions
                    for team in [home, away]:
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

                    # Distribution penalty: avoid piling matches on one day
                    penalty += (matches_per_day[weekday] ** 2) * 500

                    if penalty < best_penalty:
                        best_penalty = penalty
                        best_day = absolute_day

                selected_weekday = ((best_day - 1) % 7) + 1
                matches_per_day[selected_weekday] += 1

                play_day_map[(home, away)] = {
                    "play_day": best_day,
                    "weekday": selected_weekday,
                }

                last_played[home] = best_day
                last_played[away] = best_day

        return play_day_map
    
    def _format_results(
        self,
        assignment,
    ):

        formatted = []

        play_days = self.assign_play_days(assignment)

        for (home, away), round_num in assignment.items():

            distance = self.dist_lookup.get(
                (home, away),
                0
            )

            play_info = play_days.get(
                (home, away),
                {}
            )

            formatted.append({

                "round":
                    int(round_num),

                "home_id":
                    int(self.name_to_id[home]),

                "home_name":
                    home,

                "away_id":
                    int(self.name_to_id[away]),

                "away_name":
                    away,

                "is_derby":
                    (home, away)
                    in self.derby_matches,

                "distance_km":
                    float(distance),

                "play_day":
                    int(play_info.get("play_day", 0)),

                "weekday":
                    int(play_info.get("weekday", 1)),
            })

        formatted.sort(
            key=lambda x: (
                x["round"],
                x["home_name"]
            )
        )

        return formatted

    # ============================================================
    # MAIN SOLVER
    # ============================================================

    def solve(self):
        """
        Main public interface.
        """

        # Variables:
        # every directed fixture
        variables = []

        for a, b in combinations(
            self.teams_list,
            2
        ):

            variables.append((a, b))
            variables.append((b, a))

        # Domains:
        # all rounds
        domains = {
            v: list(
                range(
                    1,
                    self.num_rounds + 1
                )
            )
            for v in variables
        }

        start_time = time.time()

        solution = self._backtrack(
            variables,
            {},
            domains,
        )

        execution_time = (
            time.time() - start_time
        )

        if solution is None:

            return {
                "success": False,
                "schedule": [],
                "metrics": {},
                "violations": [
                    "No feasible solution found."
                ],
            }

        violations = self.verify_constraints(
            solution
        )

        metrics = self.calculate_metrics(
            solution
        )

        formatted_results = self._format_results(
            solution
        )

        return {
            "success": True,

            "execution_time_sec": execution_time,

            "schedule": formatted_results,

            "metrics": metrics,

            "constraint_report": {
                "valid":
                    len(violations) == 0,

                "violations":
                    violations,
            },

            "search_statistics": {
                "backtrack_calls":
                    self.backtrack_calls,

                "constraint_checks":
                    self.constraint_checks,

                "forward_check_failures":
                    self.forward_check_failures,

                "execution_time_seconds":
                    execution_time,
            },
        }