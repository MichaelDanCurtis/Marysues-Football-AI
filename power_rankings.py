import pandas as pd
import numpy as np
from fantasy_football_simulation import League

def calculate_luck_adjusted_power_rankings(league: League) -> pd.DataFrame:
    """
    Calculate Luck-Adjusted Power Rankings.
    
    Metrics:
    - Actual Wins: Real record.
    - Expected Wins: (Wins against all other teams each week) / (Total Teams - 1).
    - Luck: Actual Wins - Expected Wins.
    - Power Score: Combination of Expected Wins and Points For.
    """
    
    data = []
    
    # Get all completed weeks
    # Assuming all teams have the same weeks played
    if not league.teams:
        return pd.DataFrame()
        
    completed_weeks = sorted(list(league.teams[0].weekly_scores.keys()))
    
    for team in league.teams:
        actual_wins = team.wins
        expected_wins = 0
        
        for week in completed_weeks:
            my_score = team.weekly_scores.get(week, 0)
            
            # Compare against all other teams this week
            wins_this_week = 0
            opponents_count = 0
            
            for opponent in league.teams:
                if opponent.team_id == team.team_id:
                    continue
                
                opp_score = opponent.weekly_scores.get(week, 0)
                if my_score > opp_score:
                    wins_this_week += 1
                opponents_count += 1
            
            if opponents_count > 0:
                expected_wins += (wins_this_week / opponents_count)
                
        luck = actual_wins - expected_wins
        
        data.append({
            "Team": team.owner_name,
            "Record": f"{team.wins}-{team.losses}",
            "Points For": team.points_for,
            "Expected Wins": round(expected_wins, 2),
            "Luck": round(luck, 2),
            "Luck Rating": "🍀 Lucky" if luck > 1 else ("💀 Unlucky" if luck < -1 else "😐 Neutral")
        })
        
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values("Expected Wins", ascending=False)
        
    return df
