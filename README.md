# Fantasy Football Simulation & Prediction Framework

A comprehensive Python-based system for fantasy football analysis, projections, and winner prediction using statistical modeling and Monte Carlo simulations.

## Overview

This framework provides everything you need to:
- Understand fantasy football rulesets and scoring systems
- Project player and team performance
- Simulate season outcomes and playoff scenarios
- Analyze trade opportunities and player consistency
- Predict winners through advanced statistical methods
- Run comprehensive league analytics

## Contents

### 1. **fantasy_football_guide.md**
The complete knowledge base covering:
- Fantasy football rulesets and scoring systems (Standard, PPR, Half-PPR, IDP)
- League formats (Redraft, Dynasty, DFS, Keeper)
- Outcome metrics (Win Probability, Expected Wins, Consistency)
- Advanced analysis tools (WOPR, Target Share, DVOA)
- Prediction frameworks and methodologies
- Best practices for building winning systems

### 2. **fantasy_football_simulation.py**
The core framework containing:
- **Data Structures**: Player, Team, League, Matchup classes
- **Scoring Engine**: Calculates fantasy points from player statistics
- **Projection Engine**: Generates weekly and seasonal projections
- **Simulation Engine**: Monte Carlo simulations for outcome prediction
- **Analytics Engine**: Advanced metrics and analysis tools

### 3. **fantasy_football_examples.py**
Practical examples demonstrating:
- Building a realistic league with player data
- Weekly prediction and decision support
- Trade analysis and value calculations
- Playoff probability analysis
- Player consistency analysis
- Scenario planning and impact analysis

## Quick Start

### Basic Usage

```python
from fantasy_football_simulation import (
    League, Team, Player, Position, ScoringFormat,
    ScoringEngine, ProjectionEngine, SimulationEngine
)

# Create a league
league = League(
    league_id="2024_LEAGUE",
    name="My League",
    scoring_format=ScoringFormat.PPR,
    current_week=1,
    total_weeks=17
)

# Create teams
team1 = Team(team_id="T1", owner_name="Manager 1")
team2 = Team(team_id="T2", owner_name="Manager 2")
league.add_team(team1)
league.add_team(team2)

# Add players
player = Player(
    name="Patrick Mahomes",
    position=Position.QB,
    team="KC",
    nfl_id="PM_1",
    projection_mean=24.5,
    projection_std=4.2
)
team1.add_player(player)

# Project and simulate
scoring_engine = ScoringEngine(ScoringFormat.PPR)
projection_engine = ProjectionEngine(scoring_engine)
simulation_engine = SimulationEngine(league, projection_engine)

# Get playoff probabilities
results = simulation_engine.simulate_season(num_simulations=1000)
print(results)
```

## Framework Components

### Data Structures

#### Player
Represents an individual player with:
- Basic info: name, position, team, NFL ID
- Status: healthy, questionable, out, etc.
- Statistics: weekly game logs
- Projections: mean, std dev, floor, ceiling
- Metrics: consistency, target share, snap count, etc.

```python
player = Player(
    name="Travis Kelce",
    position=Position.TE,
    team="KC",
    nfl_id="TK_1",
    projection_mean=16.3,
    projection_std=2.9,
    consistency=0.87,
    target_share=0.25
)
```

#### Team
Represents a fantasy team with:
- Roster of players
- Historical performance (wins, losses, points)
- Season projections

```python
team = Team(team_id="TEAM_1", owner_name="John Doe")
team.add_player(player)
team.wins = 5
team.losses = 2
```

#### League
Contains all teams and matchups:
- Teams and their rosters
- Scoring format and settings
- Current week and schedule
- Historical matchups

```python
league = League(
    league_id="2024_COMPETITIVE",
    name="Competitive League",
    scoring_format=ScoringFormat.PPR,
    current_week=8,
    total_weeks=17
)
```

### Scoring Formats

**Standard Scoring:**
- Passing: 0.04 pts/yard, 4 pts/TD, -2 pts/INT
- Rushing: 0.1 pts/yard, 6 pts/TD
- Receiving: 0.1 pts/yard, 6 pts/TD (0 pts/reception)

**PPR (Points Per Reception):**
- Same as Standard, but 1 pt/reception

**Half-PPR:**
- Same as Standard, but 0.5 pts/reception

**DFS:**
- Custom salary cap constraints

## Analysis Tools

### Weekly Projections

```python
projection_engine = ProjectionEngine(scoring_engine)

# Project single player
mean, std = projection_engine.project_player_week(
    player=player,
    week=8,
    opponent_adjustment=1.05  # Playing worse defense
)

# Get floor/ceiling
floor, ceiling = projection_engine.calculate_floor_ceiling(player)
```

### Simulations

```python
# Week simulation (100 iterations)
week_results = simulation_engine.simulate_week(
    week=8,
    num_simulations=100
)

# Season simulation (1000 iterations)
season_results = simulation_engine.simulate_season(
    num_simulations=1000
)
```

### Advanced Analytics

```python
analytics = AnalyticsEngine()

# Strength of schedule
sos = analytics.calculate_strength_of_schedule(team, league)

# Boom/bust potential
volatility = analytics.calculate_boom_bust_potential(player)

# Trade candidates
candidates = analytics.identify_trade_candidates(league, Position.WR)
```

## Key Metrics Explained

### Consistency Score (0-1)
- **High (0.8+)**: Reliable, low variance week-to-week
- **Medium (0.4-0.8)**: Some variance
- **Low (<0.4)**: High variance, boom/bust potential

### Projection Metrics
- **Mean**: Expected points per week
- **Standard Deviation**: Variability around the mean
- **Floor**: 25th percentile (conservative estimate)
- **Ceiling**: 75th percentile (optimistic estimate)

### Advanced Metrics
- **Target Share**: % of team's targets a receiver gets
- **Air Yards**: Total yards thrown to player
- **Snap Count %**: % of team's offensive snaps player participates in
- **Red Zone Touches**: Opportunities within 20-yard line

## Prediction Methodology

### 1. Player-Level Projections
- Start with season average
- Adjust for recent form (4-week rolling average)
- Apply opponent adjustment based on defense strength
- Apply situation factors (home/away, weather)
- Generate confidence intervals from consistency

### 2. Team-Level Projections
- Sum player projections
- Account for player status (out, questionable)
- Adjust for bye weeks
- Project season total based on remaining schedule

### 3. Monte Carlo Simulations
- Generate random outcomes from projection distributions
- Simulate each remaining week independently
- Track wins, losses, and points scored
- Run 1000+ iterations for statistical validity
- Calculate playoff/championship probability

### 4. Accuracy Expectations
- Best case: 15-25% better than baseline
- Typical: 5-10% improvement
- Regression to mean as season progresses
- Uncertainty increases with fewer games remaining

## Common Use Cases

### Weekly Start/Sit Decisions
```python
# Get week 8 projections for your team
projections = []
for player in team.roster:
    mean, std = projection_engine.project_player_week(player, 8)
    projections.append({
        'player': player.name,
        'projection': mean,
        'confidence': player.consistency
    })

# Sort by projection and consistency
df = pd.DataFrame(projections).sort_values('projection', ascending=False)
```

### Trade Analysis
```python
# Calculate replacement value at each position
from fantasy_football_simulation import AnalyticsEngine

# Identify inconsistent performers (sell high)
candidates = AnalyticsEngine.identify_trade_candidates(
    league, Position.RB, threshold=0.4
)

# Project impact of trades
current_proj = team.get_season_projection()
after_trade_proj = current_proj - player_out + player_in
print(f"Impact: {after_trade_proj - current_proj:+.1f} points")
```

### Playoff Probability
```python
# How likely to make playoffs?
results = simulation_engine.simulate_season(num_simulations=2000)
playoff_prob = results[results['Team'] == 'My Team']['Playoff_Probability'].values[0]
print(f"Playoff probability: {playoff_prob:.1%}")
```

### Strength of Schedule
```python
# Which team has the easiest remaining schedule?
for team in league.teams:
    sos = AnalyticsEngine.calculate_strength_of_schedule(team, league)
    print(f"{team.owner_name}: SOS = {sos:.3f}")
```

## Advanced Features

### Custom Scoring
```python
# Create custom scoring format
custom_rules = {
    'passing_yards': 0.05,  # 1 pt per 20 yards
    'passing_td': 6,        # 6 pts instead of 4
    'reception': 1.5,       # Premium PPR
    # ... other settings
}

scoring_engine = ScoringEngine(ScoringFormat.PPR)
scoring_engine.rules = custom_rules
```

### Injury Impact Analysis
```python
# Simulate losing a key player
player.status = PlayerStatus.OUT

# Get new projections
old_proj = team.get_season_projection()
new_proj = team.get_season_projection()
impact = old_proj - new_proj

print(f"Loss of {player.name}: {impact:.1f} point impact")
```

### Matchup Analysis
```python
# Get specific week's matchups
week_matchups = league.get_week_schedule(week=8)

for matchup in week_matchups:
    print(f"{matchup.home_team.owner_name} vs {matchup.away_team.owner_name}")
    h_proj = matchup.home_team.get_week_projection()
    a_proj = matchup.away_team.get_week_projection()
    print(f"  {h_proj:.1f} - {a_proj:.1f}")
```

## Data Input

### Load Actual Player Stats
```python
# You can populate player stats from your league's API
stats = PlayerStats(
    passing_yards=298,
    passing_td=2,
    interceptions=1,
    rushing_yards=25,
    rushing_td=0,
    receptions=8,
    receiving_yards=89,
    receiving_td=0
)

player.weekly_stats.append(stats)

# Calculate points
points = scoring_engine.calculate_points(stats)
```

### Populate from External Sources
```python
# Load from ESPN, Yahoo, NFL.com, etc.
# Update player stats after each week
# Recalculate projections based on actual performance
# Run simulations with updated data
```

## Performance Considerations

- **Simulation Speed**: ~1000 iterations per second on modern hardware
- **Optimal Simulation Count**: 1000-2000 for playoffs, 5000+ for early season
- **Memory Usage**: Minimal (< 100MB for typical league)
- **Accuracy**: Improves with more games played (more historical data)

## Limitations & Assumptions

1. **Scoring Consistency**: Assumes recent performance predicts future performance
2. **Injury Probability**: Uses status flags but doesn't predict injuries
3. **Trades**: Doesn't account for mid-season roster changes
4. **Coaching Changes**: Can't anticipate role changes or offensive adjustments
5. **Regression**: Players regress toward mean (especially QBs with variable scoring)
6. **Variance**: Real-world randomness means individual games won't match projections

## Best Practices

1. **Update Weekly**: Refresh projections after each week's results
2. **Cross-Validate**: Compare your projections to expert consensus
3. **Account for Uncertainty**: Use confidence intervals, not point projections
4. **Consider Context**: Check injury reports, bye weeks, trade deadlines
5. **Manage Risk**: Balance upside (ceiling) with downside (floor) protection
6. **Diversity**: Don't rely on single prediction method
7. **Backtest**: Validate accuracy against historical seasons

## Extensions & Customization

You can extend the framework to:
- Add Vegas odds integration for matchup quality
- Incorporate weather data (wind affects passing/kicking)
- Add snap count tracking for IDP scoring
- Implement machine learning models (XGBoost, neural networks)
- Create DFS lineup optimization algorithms
- Build interactive dashboards and visualizations
- Connect to league APIs for real-time data

## Examples

Run the included examples:
```bash
python fantasy_football_simulation.py      # Basic demo
python fantasy_football_examples.py        # Advanced examples
```

## Support & Resources

- Fantasy Football Analytics: https://fantasyfootballanalytics.net/
- Draft Sharks: https://www.draftsharks.com/
- Fantasy Points: https://www.fantasypoints.com/
- Football Outsiders (DVOA): https://www.footballoutsiders.com/

## License

This framework is provided as-is for educational and personal use.

## Disclaimer

This framework provides projections and simulations based on statistical analysis. 
Actual results will vary. Use this as one tool among many in your decision-making 
process, not as a replacement for expert analysis, injury reports, or other 
contextual information.

---

**Version**: 1.0.0  
**Last Updated**: November 2024  
**Author**: Fantasy Football Analytics Framework
