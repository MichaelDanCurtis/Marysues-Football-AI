"""
Fantasy Football Simulation Framework
======================================

A comprehensive Python system for fantasy football analysis, projection,
and winner prediction through Monte Carlo simulations.

Author: Analytics Framework
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from datetime import datetime, timedelta
import json
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ScoringFormat(Enum):
    """Supported scoring formats"""
    STANDARD = "standard"
    PPR = "ppr"
    HALF_PPR = "half_ppr"
    DFS = "dfs"


class Position(Enum):
    """Player positions"""
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DEF = "DEF"
    FLEX = "FLEX"  # RB, WR, or TE


class PlayerStatus(Enum):
    """Player status"""
    HEALTHY = "healthy"
    QUESTIONABLE = "questionable"
    DOUBTFUL = "doubtful"
    OUT = "out"
    IR = "ir"


# Scoring constants
SCORING_RULES = {
    ScoringFormat.STANDARD.value: {
        'passing_yards': 0.04,
        'passing_td': 4,
        'interception': -2,
        'rushing_yards': 0.1,
        'rushing_td': 6,
        'reception': 0,
        'receiving_yards': 0.1,
        'receiving_td': 6,
        'two_point_conversion': 2,
        'fumble_lost': -2,
    },
    ScoringFormat.PPR.value: {
        'passing_yards': 0.04,
        'passing_td': 4,
        'interception': -2,
        'rushing_yards': 0.1,
        'rushing_td': 6,
        'reception': 1,
        'receiving_yards': 0.1,
        'receiving_td': 6,
        'two_point_conversion': 2,
        'fumble_lost': -2,
    },
    ScoringFormat.HALF_PPR.value: {
        'passing_yards': 0.04,
        'passing_td': 4,
        'interception': -2,
        'rushing_yards': 0.1,
        'rushing_td': 6,
        'reception': 0.5,
        'receiving_yards': 0.1,
        'receiving_td': 6,
        'two_point_conversion': 2,
        'fumble_lost': -2,
    },
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PlayerStats:
    """Actual player statistics from games"""
    passing_yards: float = 0
    passing_td: int = 0
    interceptions: int = 0
    rushing_yards: float = 0
    rushing_td: int = 0
    receptions: int = 0
    receiving_yards: float = 0
    receiving_td: int = 0
    two_point_conversions: int = 0
    fumbles_lost: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'passing_yards': self.passing_yards,
            'passing_td': self.passing_td,
            'interceptions': self.interceptions,
            'rushing_yards': self.rushing_yards,
            'rushing_td': self.rushing_td,
            'receptions': self.receptions,
            'receiving_yards': self.receiving_yards,
            'receiving_td': self.receiving_td,
            'two_point_conversions': self.two_point_conversions,
            'fumbles_lost': self.fumbles_lost,
        }


@dataclass
class Player:
    """Individual player"""
    name: str
    position: Position
    team: str
    nfl_id: str
    status: PlayerStatus = PlayerStatus.HEALTHY
    
    # Season statistics
    weekly_stats: List[PlayerStats] = field(default_factory=list)
    
    # Projection metrics
    projection_mean: float = 0  # Average projected points
    projection_std: float = 0   # Standard deviation
    consistency: float = 0      # Consistency score (inverse of volatility)
    floor: float = 0            # 25th percentile projection
    ceiling: float = 0          # 75th percentile projection
    
    # Opportunity metrics
    target_share: float = 0     # Percentage of team targets
    air_yards_share: float = 0  # Percentage of air yards
    snap_count_pct: float = 0   # Percentage of snaps played
    red_zone_touches: int = 0   # Total red zone opportunities
    
    # Advanced metrics
    dvoa_adjusted: float = 0    # Defense-adjusted value
    matchup_ranking: int = 0    # Weekly matchup ranking (1=best)
    
    def get_season_stats(self) -> PlayerStats:
        """Get cumulative season statistics"""
        if not self.weekly_stats:
            return PlayerStats()
        
        combined = PlayerStats()
        for stats in self.weekly_stats:
            combined.passing_yards += stats.passing_yards
            combined.passing_td += stats.passing_td
            combined.interceptions += stats.interceptions
            combined.rushing_yards += stats.rushing_yards
            combined.rushing_td += stats.rushing_td
            combined.receptions += stats.receptions
            combined.receiving_yards += stats.receiving_yards
            combined.receiving_td += stats.receiving_td
            combined.two_point_conversions += stats.two_point_conversions
            combined.fumbles_lost += stats.fumbles_lost
        
        return combined
    
    def get_games_played(self) -> int:
        """Get number of games with statistics"""
        return len([s for s in self.weekly_stats if s.passing_yards > 0 or 
                   s.rushing_yards > 0 or s.receiving_yards > 0])
    
    def calculate_consistency(self) -> float:
        """Calculate consistency score (1/std of recent 4 weeks)"""
        if len(self.weekly_stats) < 2:
            return 0
        
        recent = [self.score_from_stats(s) for s in self.weekly_stats[-4:]]
        if len(recent) < 2:
            return 0
        
        std = np.std(recent)
        return 1.0 / (1.0 + std) if std > 0 else 1.0  # Normalize to 0-1
    
    def score_from_stats(self, stats: PlayerStats, format: ScoringFormat = ScoringFormat.PPR) -> float:
        """Calculate fantasy points from statistics"""
        key = format.value if hasattr(format, 'value') else format
        rules = SCORING_RULES[key]
        
        points = 0
        points += stats.passing_yards * rules['passing_yards']
        points += stats.passing_td * rules['passing_td']
        points += stats.interceptions * rules['interception']
        points += stats.rushing_yards * rules['rushing_yards']
        points += stats.rushing_td * rules['rushing_td']
        points += stats.receptions * rules['reception']
        points += stats.receiving_yards * rules['receiving_yards']
        points += stats.receiving_td * rules['receiving_td']
        points += stats.two_point_conversions * rules['two_point_conversion']
        points += stats.fumbles_lost * rules['fumble_lost']
        
        return max(0, points)  # Floor at 0
    
    def get_weekly_scores(self, format: ScoringFormat = ScoringFormat.PPR) -> List[float]:
        """Get list of weekly fantasy scores"""
        return [self.score_from_stats(s, format) for s in self.weekly_stats]
    
    def get_rolling_average(self, weeks: int = 4, format: ScoringFormat = ScoringFormat.PPR) -> float:
        """Get rolling average of last N weeks"""
        scores = self.get_weekly_scores(format)
        if not scores:
            return self.projection_mean
        
        return np.mean(scores[-weeks:])
    
    def __hash__(self):
        return hash(self.nfl_id)
    
    def __eq__(self, other):
        return self.nfl_id == other.nfl_id if isinstance(other, Player) else False


@dataclass
class Team:
    """Fantasy Team"""
    team_id: str
    owner_name: str
    roster: List[Player] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    points_for: float = 0
    points_against: float = 0
    weekly_scores: Dict[int, float] = field(default_factory=dict)  # Week -> Score
    
    def add_player(self, player: Player):
        self.roster.append(player)
        
    def get_team_projection(self) -> float:
        return sum(p.projection_mean for p in self.roster if p.status != PlayerStatus.OUT)
        
    def get_season_projection(self, format: ScoringFormat = ScoringFormat.PPR) -> float:
        """Project season points"""
        return sum(player.projection_mean for player in self.roster)
    
    def get_week_projection(self, format: ScoringFormat = ScoringFormat.PPR) -> float:
        """Project week points"""
        return sum(player.projection_mean for player in self.roster if player.status != PlayerStatus.OUT)
    
    def get_win_loss_record(self) -> Tuple[int, int]:
        """Get win-loss record"""
        return self.wins, self.losses

    def get_win_percentage(self) -> float:
        total_games = self.wins + self.losses
        return self.wins / total_games if total_games > 0 else 0.0


@dataclass
class Matchup:
    """Single weekly matchup"""
    week: int
    home_team: Team
    away_team: Team
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    
    def is_played(self) -> bool:
        """Check if matchup has been played"""
        return self.home_score is not None and self.away_score is not None
    
    def get_winner(self) -> Optional[Team]:
        """Get winning team"""
        if not self.is_played():
            return None
        return self.home_team if self.home_score > self.away_score else self.away_team
    
    def get_loser(self) -> Optional[Team]:
        """Get losing team"""
        if not self.is_played():
            return None
        return self.away_team if self.home_score > self.away_score else self.home_team


@dataclass
class League:
    """Fantasy football league"""
    league_id: str
    name: str
    teams: List[Team] = field(default_factory=list)
    scoring_format: ScoringFormat = ScoringFormat.PPR
    current_week: int = 1
    total_weeks: int = 17
    matchups: List[Matchup] = field(default_factory=list)
    
    def add_team(self, team: Team) -> None:
        """Add team to league"""
        self.teams.append(team)
    
    def get_standings(self) -> pd.DataFrame:
        """Get league standings"""
        data = []
        for team in self.teams:
            wins, losses = team.get_win_loss_record()
            data.append({
                'Team': team.owner_name,
                'Wins': wins,
                'Losses': losses,
                'Win%': team.get_win_percentage(),
                'PF': team.points_for,
                'PA': team.points_against,
                'Projection': team.get_season_projection(self.scoring_format),
            })
        
        return pd.DataFrame(data).sort_values('Wins', ascending=False)
    
    def get_league_schedule(self) -> List[Matchup]:
        """Get all league matchups"""
        return self.matchups
    
    def get_week_schedule(self, week: int) -> List[Matchup]:
        """Get matchups for specific week"""
        return [m for m in self.matchups if m.week == week]


# ============================================================================
# SCORING ENGINE
# ============================================================================

class ScoringEngine:
    """Handles all fantasy football scoring calculations"""
    
    def __init__(self, format: ScoringFormat = ScoringFormat.PPR):
        self.format = format
        # Handle both Enum and string/value lookup
        key = format.value if hasattr(format, 'value') else format
        self.rules = SCORING_RULES[key]
    
    def calculate_points(self, stats: PlayerStats) -> float:
        """Calculate fantasy points from player stats"""
        points = 0
        points += stats.passing_yards * self.rules['passing_yards']
        points += stats.passing_td * self.rules['passing_td']
        points += stats.interceptions * self.rules['interception']
        points += stats.rushing_yards * self.rules['rushing_yards']
        points += stats.rushing_td * self.rules['rushing_td']
        points += stats.receptions * self.rules['reception']
        points += stats.receiving_yards * self.rules['receiving_yards']
        points += stats.receiving_td * self.rules['receiving_td']
        points += stats.two_point_conversions * self.rules['two_point_conversion']
        points += stats.fumbles_lost * self.rules['fumble_lost']
        
        return max(0, points)
    
    def calculate_team_week_score(self, team: Team) -> float:
        """Calculate team's weekly score"""
        return sum(self.calculate_points(stats) 
                  for player in team.roster 
                  if (stats := player.weekly_stats[-1] if player.weekly_stats else None))
    
    def get_rules(self) -> Dict:
        """Get current scoring rules"""
        return self.rules.copy()


# ============================================================================
# PROJECTION ENGINE
# ============================================================================

class ProjectionEngine:
    """Generates player point projections"""
    
    def __init__(self, scoring_engine: ScoringEngine):
        self.scoring_engine = scoring_engine
    
    def project_player_week(self, player: Player, week: int, 
                           opponent_adjustment: float = 1.0,
                           matchup_difficulty: int = 16) -> Tuple[float, float]:
        """
        Project player's weekly points
        
        Returns: (mean_projection, std_deviation)
        """
        if player.status == PlayerStatus.OUT:
            return 0, 0
        
        if player.status == PlayerStatus.DOUBTFUL:
            return player.projection_mean * 0.5, player.projection_std * 0.8
        
        # Base projection from rolling average
        base_projection = player.projection_mean
        
        # Apply recent form (4-week rolling)
        rolling_avg = player.get_rolling_average(weeks=4)
        if rolling_avg > 0:
            form_weight = 0.3
            base_projection = (base_projection * (1 - form_weight) + 
                             rolling_avg * form_weight)
        
        # Adjust for opponent (use matchup ranking as proxy)
        opponent_factor = 1.0 + (0.05 * (16 - player.matchup_ranking) / 16)
        adjusted_projection = base_projection * opponent_factor * opponent_adjustment
        
        # Standard deviation based on consistency
        std = player.projection_std * (1.0 - player.consistency * 0.3)
        
        return adjusted_projection, std
    
    def project_team_season(self, team: Team, weeks_remaining: int) -> float:
        """Project team's total season points"""
        return sum(player.projection_mean * weeks_remaining 
                  for player in team.roster 
                  if player.status != PlayerStatus.OUT)
    
    def calculate_floor_ceiling(self, player: Player) -> Tuple[float, float]:
        """Calculate player's floor and ceiling"""
        # Floor = mean - 1 std, Ceiling = mean + 1 std
        floor = max(0, player.projection_mean - player.projection_std)
        ceiling = player.projection_mean + player.projection_std
        
        return floor, ceiling


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class SimulationEngine:
    """Monte Carlo simulation engine for fantasy football predictions"""
    
    def __init__(self, league: League, projection_engine: ProjectionEngine):
        self.league = league
        self.projection_engine = projection_engine
        self.rng = np.random.RandomState(seed=42)
    
    def simulate_week(self, week: int, num_simulations: int = 1000,
                     seed: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Simulate a single week's outcomes
        
        Returns: Dictionary mapping team_id to list of simulated scores
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        
        results = defaultdict(list)
        
        for team in self.league.teams:
            scores = []
            
            for _ in range(num_simulations):
                week_score = 0
                
                for player in team.roster:
                    if player.status == PlayerStatus.OUT:
                        continue
                    
                    mean, std = self.projection_engine.project_player_week(
                        player, week
                    )
                    
                    # Generate random score from normal distribution
                    simulated_score = max(0, self.rng.normal(mean, std))
                    week_score += simulated_score
                
                scores.append(week_score)
            
            results[team.team_id] = scores
        
        return dict(results)
    
    def simulate_season(self, num_simulations: int = 1000,
                       seed: Optional[int] = None) -> pd.DataFrame:
        """
        Simulate entire season remaining
        
        Returns: DataFrame with results for each team
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        
        results = {team.team_id: {'wins': [], 'points_for': []} 
                  for team in self.league.teams}
        
        for sim in range(num_simulations):
            season_wins = {team.team_id: team.wins for team in self.league.teams}
            season_pf = {team.team_id: team.points_for for team in self.league.teams}
            
            # Simulate remaining weeks
            for week in range(self.league.current_week, self.league.total_weeks + 1):
                week_results = self.simulate_week(week, num_simulations=1)
                
                # Process matchups for this week
                matchups = self.league.get_week_schedule(week)
                for matchup in matchups:
                    home_score = np.mean(week_results[matchup.home_team.team_id])
                    away_score = np.mean(week_results[matchup.away_team.team_id])
                    
                    season_pf[matchup.home_team.team_id] += home_score
                    season_pf[matchup.away_team.team_id] += away_score
                    
                    if home_score > away_score:
                        season_wins[matchup.home_team.team_id] += 1
                    else:
                        season_wins[matchup.away_team.team_id] += 1
            
            for team_id, wins in season_wins.items():
                results[team_id]['wins'].append(wins)
            for team_id, pf in season_pf.items():
                results[team_id]['points_for'].append(pf)
        
        # Calculate statistics
        summary = []
        for team in self.league.teams:
            wins_array = np.array(results[team.team_id]['wins'])
            pf_array = np.array(results[team.team_id]['points_for'])
            
            summary.append({
                'Team': team.owner_name,
                'Current_Record': f"{team.wins}-{team.losses}",
                'Avg_Final_Wins': np.mean(wins_array),
                'Playoff_Probability': np.sum(wins_array >= 10) / num_simulations,  # Top 6
                'Championship_Probability': np.sum(wins_array >= 12) / num_simulations,  # Top 2
                'Avg_PF': np.mean(pf_array),
                'Std_PF': np.std(pf_array),
            })
        
        return pd.DataFrame(summary).sort_values('Playoff_Probability', ascending=False)
    
    def simulate_playoff_bracket(self, num_simulations: int = 10000,
                                seed: Optional[int] = None) -> Dict[str, float]:
        """
        Simulate playoff outcomes
        
        Returns: Championship probability for each team
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        
        champ_wins = defaultdict(int)
        
        for _ in range(num_simulations):
            # Simulate playoffs (simplified single elimination)
            standings = self.league.get_standings()
            playoff_teams = standings.head(6)['Team'].tolist()
            
            # Generate random playoff winner
            winner = self.rng.choice(playoff_teams)
            champ_wins[winner] += 1
        
        return {team: count / num_simulations for team, count in champ_wins.items()}


# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    """Advanced analytics and prediction tools"""
    
    @staticmethod
    def calculate_strength_of_schedule(team: Team, league: League) -> float:
        """
        Calculate opponent strength based on current records
        
        Returns: Average win percentage of remaining opponents
        """
        remaining_opponents = []
        for matchup in league.get_league_schedule():
            if matchup.week >= league.current_week:
                if matchup.home_team == team:
                    remaining_opponents.append(matchup.away_team)
                elif matchup.away_team == team:
                    remaining_opponents.append(matchup.home_team)
        
        if not remaining_opponents:
            return 0.5
        
        avg_win_pct = np.mean([opp.get_win_percentage() for opp in remaining_opponents])
        return avg_win_pct
    
    @staticmethod
    def calculate_expected_wins(pf: float, pf_avg: float, pa_avg: float) -> float:
        """
        Calculate expected wins using Pythagorean formula
        
        Formula: (PF^2) / (PF^2 + PA^2)
        """
        if pf_avg == 0 or pa_avg == 0:
            return 0.5
        
        return (pf_avg ** 2) / (pf_avg ** 2 + pa_avg ** 2)
    
    @staticmethod
    def identify_trade_candidates(league: League, position: Position,
                                  threshold: float = 0.3) -> pd.DataFrame:
        """
        Identify players with inconsistent performance (trade candidates)
        
        Returns: DataFrame of candidates
        """
        candidates = []
        
        for team in league.teams:
            for player in team.roster:
                if player.position != position:
                    continue
                
                if player.consistency < threshold:
                    candidates.append({
                        'Player': player.name,
                        'Team': team.owner_name,
                        'Position': player.position.value,
                        'Consistency': player.consistency,
                        'Floor': player.floor,
                        'Ceiling': player.ceiling,
                        'Recent_Avg': player.get_rolling_average(),
                    })
        
        return pd.DataFrame(candidates).sort_values('Consistency')
    
    @staticmethod
    def calculate_boom_bust_potential(player: Player) -> float:
        """
        Calculate boom-or-bust score
        
        High score = high variance (boom/bust)
        Low score = consistent (reliable)
        """
        if player.projection_std == 0:
            return 0
        
        return player.projection_std / max(player.projection_mean, 1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_league() -> League:
    """Create a sample league for demonstration"""
    league = League(
        league_id="SAMPLE_2024",
        name="Sample Fantasy Football League",
        scoring_format=ScoringFormat.PPR,
        current_week=1,
        total_weeks=17
    )
    
    # Create sample teams
    team_names = ["Mahomes' Magic", "Taylor Swift Offense", "The Mahomeboys",
                  "Go Birds", "Defense Wins Championships", "Kelce Kingdom"]
    
    for i, name in enumerate(team_names):
        team = Team(
            team_id=f"TEAM_{i}",
            owner_name=name
        )
        league.add_team(team)
    
    # Create sample players
    qbs = [
        Player(name="Patrick Mahomes", position=Position.QB, team="KC", nfl_id="PM_1",
               projection_mean=24.5, projection_std=4.2),
        Player(name="Jalen Hurts", position=Position.QB, team="PHI", nfl_id="JH_1",
               projection_mean=23.1, projection_std=3.8),
    ]
    
    rbs = [
        Player(name="Christian McCaffrey", position=Position.RB, team="SF", nfl_id="CMC_1",
               projection_mean=18.5, projection_std=3.2),
        Player(name="Josh Jacobs", position=Position.RB, team="LV", nfl_id="JJ_1",
               projection_mean=14.2, projection_std=3.8),
    ]
    
    wrs = [
        Player(name="Travis Kelce", position=Position.TE, team="KC", nfl_id="TK_1",
               projection_mean=16.3, projection_std=2.9),
        Player(name="CeeDee Lamb", position=Position.WR, team="DAL", nfl_id="CDL_1",
               projection_mean=15.8, projection_std=3.5),
    ]
    
    all_players = qbs + rbs + wrs
    
    # Assign players to teams
    for i, player in enumerate(all_players):
        team = league.teams[i % len(league.teams)]
        team.add_player(player)
    
    return league


def print_league_summary(league: League):
    """Print league summary statistics"""
    print("\n" + "="*60)
    print(f"LEAGUE: {league.name}")
    print("="*60)
    print(f"Format: {league.scoring_format.value.upper()}")
    print(f"Week: {league.current_week}/{league.total_weeks}")
    print("\nStandings:")
    print(league.get_standings().to_string(index=False))
    print("\n" + "="*60)


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main demonstration of the fantasy football system"""
    
    print("Fantasy Football Simulation Framework")
    print("="*60)
    
    # Create sample league
    league = create_sample_league()
    print_league_summary(league)
    
    # Initialize engines
    scoring_engine = ScoringEngine(ScoringFormat.PPR)
    projection_engine = ProjectionEngine(scoring_engine)
    simulation_engine = SimulationEngine(league, projection_engine)
    
    # Simulate a week
    print("\n" + "="*60)
    print("WEEK 1 SIMULATION (100 iterations)")
    print("="*60)
    
    week_results = simulation_engine.simulate_week(week=1, num_simulations=100)
    
    for team in league.teams:
        scores = week_results[team.team_id]
        print(f"\n{team.owner_name}:")
        print(f"  Avg Score: {np.mean(scores):.2f}")
        print(f"  Std Dev: {np.std(scores):.2f}")
        print(f"  Floor: {np.percentile(scores, 25):.2f}")
        print(f"  Ceiling: {np.percentile(scores, 75):.2f}")
    
    # Season simulation
    print("\n" + "="*60)
    print("SEASON PROJECTION (1000 simulations)")
    print("="*60)
    
    season_results = simulation_engine.simulate_season(num_simulations=1000)
    print(season_results.to_string(index=False))
    
    # Analytics
    print("\n" + "="*60)
    print("LEAGUE ANALYTICS")
    print("="*60)
    
    analytics = AnalyticsEngine()
    
    for team in league.teams:
        sos = analytics.calculate_strength_of_schedule(team, league)
        print(f"\n{team.owner_name}:")
        print(f"  Strength of Schedule: {sos:.3f}")
        print(f"  Season Projection: {team.get_season_projection():.1f} points")
    
    print("\n" + "="*60)
    print("Simulation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
