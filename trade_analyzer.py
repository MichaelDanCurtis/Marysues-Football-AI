import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from fantasy_football_simulation import League, Team, Player, Position, SimulationEngine, ProjectionEngine, ScoringEngine

class TradeAnalyzer:
    def __init__(self, league: League, simulation_engine: SimulationEngine):
        self.league = league
        self.sim_engine = simulation_engine
        self.projection_engine = simulation_engine.projection_engine

    def calculate_roster_value(self, team: Team) -> Dict[str, float]:
        """
        Calculate the total projected value of a roster for the rest of the season.
        Returns a dictionary of value by position.
        """
        values = {pos: 0.0 for pos in Position}
        weeks_remaining = self.league.total_weeks - self.league.current_week + 1
        
        for player in team.roster:
            # Simple projection: mean * weeks remaining
            # In a real scenario, we'd use the simulation engine, but this is a fast heuristic
            val = player.projection_mean * weeks_remaining
            values[player.position] += val
            
        return values

    def get_position_rankings(self) -> Dict[Position, pd.DataFrame]:
        """
        Rank all players in the league by position based on rest-of-season projection.
        """
        rankings = {}
        weeks_remaining = max(1, self.league.total_weeks - self.league.current_week + 1)
        
        for pos in Position:
            players = []
            for team in self.league.teams:
                for player in team.roster:
                    if player.position == pos:
                        players.append({
                            'Player': player,
                            'Team': team,
                            'ROS_Points': player.projection_mean * weeks_remaining
                        })
            
            df = pd.DataFrame(players)
            if not df.empty:
                df = df.sort_values('ROS_Points', ascending=False).reset_index(drop=True)
                df['Rank'] = df.index + 1
                rankings[pos] = df
            else:
                rankings[pos] = pd.DataFrame()
                
        return rankings

    def identify_team_needs(self, team: Team) -> List[Dict]:
        """
        Identify team needs by comparing starters to league average.
        Assumes standard roster settings (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX).
        """
        rankings = self.get_position_rankings()
        needs = []
        
        # Define standard starting roster slots
        slots = {
            Position.QB: 1,
            Position.RB: 2,
            Position.WR: 2,
            Position.TE: 1
        }
        
        league_size = len(self.league.teams)
        
        for pos, count in slots.items():
            if pos not in rankings or rankings[pos].empty:
                continue
                
            df = rankings[pos]
            
            # Find team's players at this position
            team_players = df[df['Team'] == team]
            
            if team_players.empty:
                needs.append({'Position': pos, 'Severity': 'Critical', 'Reason': 'No players'})
                continue
                
            # Check if top players are "starters" (e.g., top 12 QB, top 24 RB)
            # A simple heuristic: Is the Nth best player on the team worse than the league average starter?
            
            # Get the team's starters at this position
            starters = team_players.head(count)
            
            # Calculate "Starter Strength" relative to league
            # League average starter rank for QB is ~6.5 (in 12 team), RB1 is ~6.5, RB2 is ~18.5
            
            for i in range(count):
                if i < len(starters):
                    player_rank = starters.iloc[i]['Rank']
                    # Threshold: If your RB1 is ranked > 12, you are weak. If RB2 > 24, you are weak.
                    threshold = league_size * (i + 1)
                    
                    if player_rank > threshold:
                        needs.append({
                            'Position': pos, 
                            'Severity': 'High' if player_rank > threshold * 1.5 else 'Moderate',
                            'Reason': f"Starter #{i+1} ranked #{player_rank} (Target < {threshold})"
                        })
                else:
                     needs.append({'Position': pos, 'Severity': 'Critical', 'Reason': f'Missing starter #{i+1}'})

        return needs

    def find_trade_partners(self, my_team: Team) -> List[Dict]:
        """
        Find teams that match my needs and have surplus in my areas of strength.
        """
        my_needs = self.identify_team_needs(my_team)
        my_strengths = [] # Inverse of needs, essentially
        
        # Identify my strengths (surplus)
        rankings = self.get_position_rankings()
        league_size = len(self.league.teams)
        
        for pos in [Position.QB, Position.RB, Position.WR, Position.TE]:
            if pos not in rankings or rankings[pos].empty: continue
            
            df = rankings[pos]
            team_players = df[df['Team'] == my_team]
            
            # Check for bench strength (e.g., having a top 24 RB on the bench)
            # Standard starters: QB=1, RB=2, WR=2, TE=1
            starter_count = 1 if pos in [Position.QB, Position.TE] else 2
            
            if len(team_players) > starter_count:
                bench_player = team_players.iloc[starter_count] # The best bench player
                # If bench player is "startable" (e.g., top 36 RB/WR, top 15 QB/TE)
                threshold = league_size * (starter_count + 1) # e.g. 36 for RB
                
                if bench_player['Rank'] <= threshold:
                    my_strengths.append({
                        'Position': pos,
                        'Player': bench_player['Player'],
                        'Rank': bench_player['Rank']
                    })

        suggestions = []
        
        for other_team in self.league.teams:
            if other_team == my_team: continue
            
            other_needs = self.identify_team_needs(other_team)
            
            # Match logic:
            # 1. They need what I have (My Strength matches Their Need)
            # 2. I need what they have (Their Strength matches My Need) - Harder to calculate without full scan
            
            score = 0
            details = []
            
            # Check if I can help them
            for strength in my_strengths:
                for need in other_needs:
                    if strength['Position'] == need['Position']:
                        score += 2 if need['Severity'] == 'Critical' else 1
                        details.append(f"You give {strength['Player'].name} ({strength['Position'].value}) to fix their {need['Severity']} need.")
            
            # Check if they can help me (simplified: do they have a surplus at my need?)
            for my_need in my_needs:
                # Check their surplus at this position
                if my_need['Position'] not in rankings: continue
                df = rankings[my_need['Position']]
                their_players = df[df['Team'] == other_team]
                
                starter_count = 1 if my_need['Position'] in [Position.QB, Position.TE] else 2
                if len(their_players) > starter_count:
                     # They have depth
                     surplus_player = their_players.iloc[starter_count] # Their best bench player
                     # Or even one of their starters if they are super deep
                     
                     score += 1
                     details.append(f"Target {surplus_player['Player'].name} ({my_need['Position'].value}) to fix your {my_need['Severity']} need.")

            if score > 0:
                suggestions.append({
                    'Team': other_team.owner_name,
                    'Score': score,
                    'Details': details
                })
                
        return sorted(suggestions, key=lambda x: x['Score'], reverse=True)

    def evaluate_trade(self, team_a: Team, team_b: Team, 
                      players_a_gives: List[Player], players_b_gives: List[Player]) -> Dict:
        """
        Simulate the season with and without the trade to determine impact.
        """
        # 1. Run baseline simulation (current rosters)
        # We can use a smaller number of sims for speed
        baseline_results = self.sim_engine.simulate_season(num_simulations=100)
        
        team_a_base_wins = baseline_results[baseline_results['Team'] == team_a.owner_name]['Avg_Final_Wins'].values[0]
        team_b_base_wins = baseline_results[baseline_results['Team'] == team_b.owner_name]['Avg_Final_Wins'].values[0]
        
        # 2. Apply trade temporarily
        # Helper to swap players
        original_roster_a = list(team_a.roster)
        original_roster_b = list(team_b.roster)
        
        # Execute swap in memory
        for p in players_a_gives:
            if p in team_a.roster: team_a.roster.remove(p)
            team_b.roster.append(p)
            
        for p in players_b_gives:
            if p in team_b.roster: team_b.roster.remove(p)
            team_a.roster.append(p)
            
        # 3. Run post-trade simulation
        trade_results = self.sim_engine.simulate_season(num_simulations=100)
        
        team_a_trade_wins = trade_results[trade_results['Team'] == team_a.owner_name]['Avg_Final_Wins'].values[0]
        team_b_trade_wins = trade_results[trade_results['Team'] == team_b.owner_name]['Avg_Final_Wins'].values[0]
        
        # 4. Revert rosters
        team_a.roster = original_roster_a
        team_b.roster = original_roster_b
        
        return {
            'team_a_delta': team_a_trade_wins - team_a_base_wins,
            'team_b_delta': team_b_trade_wins - team_b_base_wins,
            'fairness_score': abs((team_a_trade_wins - team_a_base_wins) - (team_b_trade_wins - team_b_base_wins))
        }
