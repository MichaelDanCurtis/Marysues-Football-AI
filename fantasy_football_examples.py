"""
Fantasy Football Framework - Advanced Examples
==============================================

Practical use cases demonstrating the framework for:
- Weekly prediction and optimization
- Trade analysis
- Playoff probability calculations
- Player consistency analysis
- Real-time decision support
"""

import numpy as np
import pandas as pd
from fantasy_football_simulation import (
    League, Team, Player, Position, ScoringFormat, PlayerStatus,
    ScoringEngine, ProjectionEngine, SimulationEngine, AnalyticsEngine,
    PlayerStats, Matchup
)
from typing import Dict, List, Tuple
import json


# ============================================================================
# EXAMPLE 1: BUILDING A REALISTIC LEAGUE WITH SEASON DATA
# ============================================================================

def example_realistic_league() -> League:
    """Build a realistic league with actual season data"""
    
    league = League(
        league_id="2024_COMPETITIVE",
        name="12-Team Competitive League",
        scoring_format=ScoringFormat.PPR,
        current_week=8,  # Midseason
        total_weeks=17
    )
    
    # Create 12 teams
    owners = [
        "The Mahomes Cronies", "Swift Justice", "Kelce's Kingdom",
        "Burrow Power", "Allen Island", "Lamar's Legends",
        "Tank Commander", "CeeDee's Army", "Saquon's Squad",
        "Travis Strikes Back", "Sunshine State", "The Underdogs"
    ]
    
    teams = {}
    for i, owner in enumerate(owners):
        team = Team(team_id=f"TEAM_{i:02d}", owner_name=owner)
        teams[i] = team
        league.add_team(team)
    
    # Add elite QBs
    elite_qbs = [
        ("Patrick Mahomes", "KC", 24.5, 4.2, 95),
        ("Josh Allen", "BUF", 23.8, 4.1, 92),
        ("Jalen Hurts", "PHI", 23.1, 3.9, 88),
        ("Lamar Jackson", "BAL", 22.9, 4.0, 87),
    ]
    
    for i, (name, team, mean, std, games) in enumerate(elite_qbs):
        player = Player(
            name=name, position=Position.QB, team=team,
            nfl_id=f"QB_{i}", status=PlayerStatus.HEALTHY,
            projection_mean=mean, projection_std=std,
            snap_count_pct=0.95, consistency=0.85
        )
        teams[i].add_player(player)
    
    # Add elite RBs with varying consistency
    elite_rbs = [
        ("Christian McCaffrey", "SF", 18.5, 3.2, 0.90),  # Very consistent
        ("Josh Jacobs", "LV", 14.2, 3.8, 0.75),  # Some variance
        ("Saquon Barkley", "PHI", 16.8, 2.9, 0.88),
        ("Derrick Henry", "TEN", 15.2, 4.5, 0.65),  # Boom/bust
        ("Kenneth Walker", "SEA", 12.1, 3.5, 0.70),
        ("De'Von Achane", "MIA", 13.4, 3.2, 0.72),
    ]
    
    for i, (name, team, mean, std, consistency) in enumerate(elite_rbs):
        player = Player(
            name=name, position=Position.RB, team=team,
            nfl_id=f"RB_{i}", status=PlayerStatus.HEALTHY,
            projection_mean=mean, projection_std=std,
            snap_count_pct=0.85, consistency=consistency,
            target_share=0.22, red_zone_touches=8
        )
        teams[4 + i].add_player(player)
    
    # Add elite WRs
    elite_wrs = [
        ("Travis Kelce", "KC", 16.3, 2.9, 0.87),
        ("CeeDee Lamb", "DAL", 15.8, 3.5, 0.80),
        ("Tyreek Hill", "MIA", 15.2, 2.8, 0.85),
        ("Justin Jefferson", "MIN", 14.9, 3.3, 0.82),
        ("Stefon Diggs", "HOU", 14.5, 3.6, 0.78),
        ("AJ Brown", "PHI", 16.1, 3.2, 0.83),
    ]
    
    for i, (name, team, mean, std, consistency) in enumerate(elite_wrs):
        player = Player(
            name=name, position=Position.WR if "Kelce" not in name else Position.TE,
            team=team, nfl_id=f"WR_{i}",
            status=PlayerStatus.HEALTHY,
            projection_mean=mean, projection_std=std,
            snap_count_pct=0.90, consistency=consistency,
            target_share=0.28, air_yards_share=0.32
        )
        teams[10 + (i % 2)].add_player(player)
    
    # Simulate some historical games (week 1-7)
    for team in league.teams:
        team.wins = np.random.randint(2, 7)
        team.losses = 7 - team.wins
        team.points_for = np.random.uniform(80, 140)
        team.points_against = np.random.uniform(80, 140)
    
    return league, teams


# ============================================================================
# EXAMPLE 2: WEEKLY PREDICTION & LINEUP OPTIMIZATION
# ============================================================================

def example_weekly_prediction(league: League):
    """Demonstrate weekly prediction and decision-making"""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: WEEKLY PREDICTION & DECISION SUPPORT")
    print("="*70)
    
    scoring_engine = ScoringEngine(league.scoring_format)
    projection_engine = ProjectionEngine(scoring_engine)
    
    # Get the first team
    team = league.teams[0]
    
    print(f"\nTeam: {team.owner_name}")
    print(f"Record: {team.wins}-{team.losses}")
    print(f"\nWeekly Projections for Week {league.current_week}:")
    print("-" * 70)
    
    player_projections = []
    
    for player in team.roster:
        if player.status == PlayerStatus.OUT:
            continue
        
        # Get projection for this week
        mean, std = projection_engine.project_player_week(player, league.current_week)
        floor, ceiling = projection_engine.calculate_floor_ceiling(player)
        
        # Calculate consistency score
        consistency = player.consistency
        
        player_projections.append({
            'Player': player.name,
            'Position': player.position.value,
            'Projection': mean,
            'Floor': floor,
            'Ceiling': ceiling,
            'Std_Dev': std,
            'Consistency': consistency,
            'Risk_Level': 'Low' if consistency > 0.7 else 'Medium' if consistency > 0.4 else 'High'
        })
    
    df_projections = pd.DataFrame(player_projections).sort_values('Projection', ascending=False)
    print(df_projections.to_string(index=False))
    
    total_projection = df_projections['Projection'].sum()
    print(f"\nTeam Weekly Projection: {total_projection:.2f} points")
    print(f"Range: {df_projections['Floor'].sum():.2f} - {df_projections['Ceiling'].sum():.2f}")
    
    return df_projections


# ============================================================================
# EXAMPLE 3: TRADE ANALYSIS
# ============================================================================

def example_trade_analysis(league: League):
    """Analyze trade scenarios and value exchanges"""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: TRADE ANALYSIS")
    print("="*70)
    
    analytics = AnalyticsEngine()
    
    # Find high-ceiling, high-variance players (sell high candidates)
    print("\nSell High Candidates (High Ceiling, Variable Performance):")
    print("-" * 70)
    
    candidates = []
    for team in league.teams:
        for player in team.roster:
            if player.position in [Position.RB, Position.WR]:
                boom_bust = analytics.calculate_boom_bust_potential(player)
                candidates.append({
                    'Player': player.name,
                    'Team': team.owner_name,
                    'Position': player.position.value,
                    'Projection': player.projection_mean,
                    'Ceiling': player.projection_mean + player.projection_std,
                    'Boom_Bust': boom_bust,
                    'Consistency': player.consistency,
                })
    
    df_trade = pd.DataFrame(candidates).sort_values('Boom_Bust', ascending=False).head(5)
    print(df_trade.to_string(index=False))
    
    # Analyze replacement value at each position
    print("\n\nPosition-by-Position Analysis:")
    print("-" * 70)
    
    position_stats = {}
    for position in [Position.QB, Position.RB, Position.WR, Position.TE]:
        players_at_pos = []
        for team in league.teams:
            for player in team.roster:
                if player.position == position:
                    players_at_pos.append({
                        'name': player.name,
                        'projection': player.projection_mean,
                        'team': team.owner_name
                    })
        
        if players_at_pos:
            projections = [p['projection'] for p in players_at_pos]
            position_stats[position.value] = {
                'avg': np.mean(projections),
                'std': np.std(projections),
                'top': max(projections),
                'count': len(projections)
            }
    
    print(f"{'Position':<8} {'Count':<6} {'Avg Proj':<12} {'Std Dev':<12} {'Top':<8}")
    print("-" * 70)
    for pos, stats in position_stats.items():
        print(f"{pos:<8} {stats['count']:<6} {stats['avg']:<12.1f} {stats['std']:<12.2f} {stats['top']:<8.1f}")


# ============================================================================
# EXAMPLE 4: PLAYOFF PROBABILITY SIMULATION
# ============================================================================

def example_playoff_probability(league: League):
    """Simulate playoff probability for each team"""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: PLAYOFF PROBABILITY ANALYSIS")
    print("="*70)
    
    scoring_engine = ScoringEngine(league.scoring_format)
    projection_engine = ProjectionEngine(scoring_engine)
    simulation_engine = SimulationEngine(league, projection_engine)
    
    print(f"\nSimulating season outcomes from Week {league.current_week} forward...")
    print("Running 2000 Monte Carlo simulations...\n")
    
    results = simulation_engine.simulate_season(num_simulations=2000)
    
    print(results.to_string(index=False))
    
    print("\n\nKey Insights:")
    print("-" * 70)
    
    strong_contenders = results[results['Playoff_Probability'] > 0.75]
    underdogs = results[results['Playoff_Probability'] < 0.25]
    middle = results[(results['Playoff_Probability'] >= 0.25) & 
                     (results['Playoff_Probability'] <= 0.75)]
    
    print(f"Strong Contenders (>75% playoff chance): {len(strong_contenders)} teams")
    if len(strong_contenders) > 0:
        print(f"  - {', '.join(strong_contenders['Team'].tolist())}")
    
    print(f"\nMiddle of Pack (25-75% playoff chance): {len(middle)} teams")
    
    print(f"\nUnderdogs (<25% playoff chance): {len(underdogs)} teams")
    if len(underdogs) > 0:
        print(f"  - {', '.join(underdogs['Team'].tolist())}")
    
    return results


# ============================================================================
# EXAMPLE 5: CONSISTENCY ANALYSIS
# ============================================================================

def example_consistency_analysis(league: League):
    """Analyze player consistency and variability"""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: PLAYER CONSISTENCY & VOLATILITY ANALYSIS")
    print("="*70)
    
    analytics = AnalyticsEngine()
    
    # Collect all players
    all_players = []
    for team in league.teams:
        for player in team.roster:
            all_players.append({
                'name': player.name,
                'position': player.position.value,
                'team': team.owner_name,
                'projection': player.projection_mean,
                'consistency': player.consistency,
                'std_dev': player.projection_std,
                'boom_bust': analytics.calculate_boom_bust_potential(player)
            })
    
    df_consistency = pd.DataFrame(all_players)
    
    # Find most consistent players
    print("\nMost Consistent Players (Low Variance):")
    print("-" * 70)
    top_consistent = df_consistency.nlargest(5, 'consistency')
    print(top_consistent[['name', 'position', 'projection', 'consistency', 'std_dev']].to_string(index=False))
    
    # Find most volatile players
    print("\n\nMost Volatile Players (High Variance - Boom/Bust):")
    print("-" * 70)
    top_volatile = df_consistency.nlargest(5, 'boom_bust')
    print(top_volatile[['name', 'position', 'projection', 'consistency', 'boom_bust']].to_string(index=False))
    
    # Average by position
    print("\n\nConsistency by Position:")
    print("-" * 70)
    position_consistency = df_consistency.groupby('position').agg({
        'consistency': 'mean',
        'std_dev': 'mean',
        'boom_bust': 'mean'
    }).round(3)
    print(position_consistency)


# ============================================================================
# EXAMPLE 6: SCENARIO PLANNING
# ============================================================================

def example_scenario_planning(league: League, team_idx: int = 0):
    """Analyze different injury/trade scenarios"""
    
    print("\n" + "="*70)
    print("EXAMPLE 6: SCENARIO PLANNING & IMPACT ANALYSIS")
    print("="*70)
    
    team = league.teams[team_idx]
    
    print(f"\nTeam: {team.owner_name}")
    print(f"Current Projection: {team.get_season_projection():.1f} points")
    
    # Scenario 1: Loss of top RB to injury
    print("\n\nScenario 1: Top RB Gets Injured")
    print("-" * 70)
    
    rbs = [p for p in team.roster if p.position == Position.RB]
    if rbs:
        top_rb = max(rbs, key=lambda x: x.projection_mean)
        print(f"Impact of losing {top_rb.name}:")
        print(f"  Loss of projection: {top_rb.projection_mean:.1f} points/week")
        print(f"  Season impact: {top_rb.projection_mean * 9:.1f} points (9 weeks remaining)")
        print(f"  New projection: {team.get_season_projection() - top_rb.projection_mean * 9:.1f} points")
    
    # Scenario 2: Trade impact
    print("\n\nScenario 2: Trade Analysis")
    print("-" * 70)
    
    wrs = [p for p in team.roster if p.position == Position.WR]
    if len(wrs) > 1:
        wrs_sorted = sorted(wrs, key=lambda x: x.projection_mean, reverse=True)
        trade_from = wrs_sorted[0]
        replacement_value = sum(p.projection_mean for p in wrs_sorted[1:]) / len(wrs_sorted[1:])
        
        print(f"Trading away {trade_from.name}:")
        print(f"  Loss: {trade_from.projection_mean:.1f} points/week")
        print(f"  Replacement value: {replacement_value:.1f} points/week")
        print(f"  Net impact: {replacement_value - trade_from.projection_mean:.1f} points/week")
        print(f"  To break even, acquire player with {trade_from.projection_mean:.1f}+ projection")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print("FANTASY FOOTBALL FRAMEWORK - ADVANCED EXAMPLES")
    print("="*70)
    
    # Create realistic league
    league, teams = example_realistic_league()
    
    print(f"\nLeague Setup Complete!")
    print(f"Teams: {len(league.teams)}")
    print(f"Current Week: {league.current_week}/{league.total_weeks}")
    print(f"Format: {league.scoring_format.value.upper()}")
    
    # Run examples
    example_weekly_prediction(league)
    example_trade_analysis(league)
    example_consistency_analysis(league)
    example_playoff_probability(league)
    example_scenario_planning(league)
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE!")
    print("="*70)
    
    # Summary of framework capabilities
    print("\n\nFramework Capabilities Summary:")
    print("-" * 70)
    print("""
✓ Weekly point projections with confidence intervals
✓ Trade analysis and value calculations
✓ Playoff probability simulations (Monte Carlo)
✓ Player consistency and boom/bust analysis
✓ Strength of schedule calculations
✓ Scenario planning and impact analysis
✓ Season-long projections with uncertainty quantification
✓ Support for multiple scoring formats (PPR, Standard, Half-PPR, DFS)
✓ Customizable player metrics and projections
✓ Flexible league configuration
    """)


if __name__ == "__main__":
    main()
