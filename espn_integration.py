from espn_api.football import League as EspnLeague
from fantasy_football_simulation import (
    League, Team, Player, Position, ScoringFormat, PlayerStatus
)
import numpy as np
from collections import defaultdict

def map_position(pos_str: str) -> Position:
    """Map ESPN position string to Position enum"""
    pos_map = {
        'QB': Position.QB,
        'RB': Position.RB,
        'WR': Position.WR,
        'TE': Position.TE,
        'D/ST': Position.DEF,
        'K': Position.K
    }
    # Default to FLEX if unknown or multiple (ESPN sometimes has 'RB/WR')
    return pos_map.get(pos_str, Position.FLEX)

def map_status(status_str: str) -> PlayerStatus:
    """Map ESPN status string to PlayerStatus enum"""
    status_map = {
        'ACTIVE': PlayerStatus.HEALTHY,
        'QUESTIONABLE': PlayerStatus.QUESTIONABLE,
        'DOUBTFUL': PlayerStatus.DOUBTFUL,
        'OUT': PlayerStatus.OUT,
        'IR': PlayerStatus.IR,
        'INJURY_RESERVE': PlayerStatus.IR
    }
    return status_map.get(status_str, PlayerStatus.HEALTHY)

def get_advanced_stats(espn_league):
    """
    Fetch historical scores and current week projections from box scores.
    This provides much higher accuracy than simple averages.
    """
    projections = {}
    history = defaultdict(list)
    
    # 1. Get History (Weeks 1 to current-1)
    # Limit to current season context
    current_week = espn_league.current_week
    
    # Fetch history
    for w in range(1, current_week):
        try:
            box_scores = espn_league.box_scores(week=w)
            for match in box_scores:
                # Combine home and away lineups
                all_players = match.home_lineup + match.away_lineup
                for player in all_players:
                    if hasattr(player, 'points'):
                        history[player.playerId].append(player.points)
        except Exception:
            continue

    # 2. Get Current Week Projections
    try:
        # If it's the offseason or pre-season, this might fail or return empty
        current_box = espn_league.box_scores(week=current_week)
        for match in current_box:
            all_players = match.home_lineup + match.away_lineup
            for player in all_players:
                # Use projected_points if available
                proj = getattr(player, 'projected_points', 0)
                # Handle cases where projection is None (e.g. BYE)
                if proj is None: proj = 0
                projections[player.playerId] = proj
    except Exception:
        pass
        
    return projections, history

def import_espn_league(league_id: int, year: int, swid: str = None, espn_s2: str = None) -> League:
    """
    Import a league from ESPN API and convert to simulation League format
    """
    try:
        espn_league = EspnLeague(league_id=league_id, year=year, swid=swid, espn_s2=espn_s2)
    except Exception as e:
        raise ValueError(f"Failed to connect to ESPN League: {str(e)}")

    # Create Simulation League
    sim_league = League(
        league_id=str(league_id),
        name=espn_league.settings.name if hasattr(espn_league, 'settings') else f"ESPN League {league_id}",
        scoring_format=ScoringFormat.PPR, # Defaulting to PPR, could try to infer from settings
        current_week=espn_league.current_week,
        total_weeks=17 # Standard NFL season
    )

    # Fetch Advanced Stats (Projections & History)
    # This takes a bit longer but provides "most accurate" data
    projections_map, history_map = get_advanced_stats(espn_league)

    # Import Teams
    sim_teams_map = {}
    for espn_team in espn_league.teams:
        sim_team = Team(
            team_id=f"TEAM_{espn_team.team_id}",
            owner_name=str(espn_team.owner),
            wins=espn_team.wins,
            losses=espn_team.losses,
            points_for=espn_team.points_for,
            points_against=espn_team.points_against
        )
        sim_teams_map[espn_team.team_id] = sim_team

        # Import Roster
        for espn_player in espn_team.roster:
            pid = espn_player.playerId
            
            # 1. Determine Projection Mean
            # Priority: Current Week Projection > Season Avg > 0
            proj_mean = 0
            if pid in projections_map:
                proj_mean = projections_map[pid]
            else:
                # Fallback to simple average if not in this week's box score (e.g. BYE or Bench?)
                # Actually bench players are in box scores usually, but let's be safe
                games_played = max(1, espn_league.current_week - 1)
                if espn_player.total_points > 0:
                    proj_mean = espn_player.total_points / games_played

            # 2. Determine Projection Std (Volatility)
            # Calculate from history if available
            proj_std = 0
            consistency = 0.5
            
            if pid in history_map and len(history_map[pid]) > 1:
                scores = history_map[pid]
                # Calculate standard deviation of actual scores
                proj_std = np.std(scores)
                # Calculate consistency (inverse of coefficient of variation, roughly)
                # Simple metric: 1 / (1 + std/mean)
                avg_score = np.mean(scores)
                if avg_score > 0:
                    cv = proj_std / avg_score
                    consistency = 1.0 / (1.0 + cv)
                else:
                    consistency = 0.5
            else:
                # Default fallback
                proj_std = max(2.0, proj_mean * 0.25)
                consistency = 0.7

            sim_player = Player(
                name=espn_player.name,
                position=map_position(espn_player.position),
                team=espn_player.proTeam,
                nfl_id=f"{espn_player.name}_{espn_player.playerId}",
                status=map_status(espn_player.injuryStatus),
                projection_mean=float(proj_mean),
                projection_std=float(proj_std),
                consistency=float(consistency)
            )
            
            sim_team.add_player(sim_player)
        
        sim_league.add_team(sim_team)

    # Populate Weekly Scores for Power Rankings
    # We need to fetch box scores for all completed weeks
    for w in range(1, espn_league.current_week):
        try:
            box_scores = espn_league.box_scores(week=w)
            for match in box_scores:
                if match.home_team and match.home_team.team_id in sim_teams_map:
                    sim_teams_map[match.home_team.team_id].weekly_scores[w] = match.home_score
                
                if match.away_team and match.away_team.team_id in sim_teams_map:
                    sim_teams_map[match.away_team.team_id].weekly_scores[w] = match.away_score
        except Exception:
            pass

    return sim_league
