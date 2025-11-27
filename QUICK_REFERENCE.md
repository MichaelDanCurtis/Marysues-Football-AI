# Fantasy Football Framework - Quick Reference

## Import Essentials
```python
from fantasy_football_simulation import (
    League, Team, Player, Position, ScoringFormat, PlayerStatus,
    ScoringEngine, ProjectionEngine, SimulationEngine, AnalyticsEngine,
    PlayerStats, Matchup
)
```

---

## 1. CREATE LEAGUE & TEAMS

```python
# Create league
league = League(
    league_id="2024_LEAGUE",
    name="My Fantasy League",
    scoring_format=ScoringFormat.PPR,
    current_week=1,
    total_weeks=17
)

# Create team
team = Team(team_id="T1", owner_name="John Doe")
league.add_team(team)
```

---

## 2. CREATE PLAYERS

```python
# Basic player
player = Player(
    name="Patrick Mahomes",
    position=Position.QB,
    team="KC",
    nfl_id="PM_1"
)

# Player with projections
player = Player(
    name="Travis Kelce",
    position=Position.TE,
    team="KC",
    nfl_id="TK_1",
    projection_mean=16.3,      # Expected points/week
    projection_std=2.9,         # Variability
    consistency=0.87,           # Reliability score (0-1)
    snap_count_pct=0.90,       # % of snaps
    target_share=0.25,         # % of targets
    status=PlayerStatus.HEALTHY
)

# Add to team
team.add_player(player)
```

---

## 3. SCORING SETUP

```python
# Initialize scoring engine
scoring_engine = ScoringEngine(ScoringFormat.PPR)

# Or custom format
scoring_engine = ScoringEngine(ScoringFormat.STANDARD)

# Or customize rules
scoring_engine.rules['passing_td'] = 6  # 6 pts instead of 4
```

---

## 4. GENERATE PROJECTIONS

```python
# Initialize projection engine
projection_engine = ProjectionEngine(scoring_engine)

# Single player week projection
mean, std = projection_engine.project_player_week(
    player=player,
    week=8,
    opponent_adjustment=1.05  # Worse defense = higher proj
)
# Returns: (mean_points, std_dev)

# Get floor and ceiling
floor, ceiling = projection_engine.calculate_floor_ceiling(player)

# Team weekly projection
team_proj = team.get_week_projection()

# Team season projection
season_proj = team.get_season_projection()
```

---

## 5. MONTE CARLO SIMULATIONS

```python
# Initialize simulation engine
simulation_engine = SimulationEngine(league, projection_engine)

# Simulate single week (100 iterations)
week_results = simulation_engine.simulate_week(
    week=8,
    num_simulations=100
)
# Returns: Dict[team_id] -> List[scores]

# Simulate entire season (1000 iterations)
season_results = simulation_engine.simulate_season(
    num_simulations=1000
)
# Returns: DataFrame with playoff/championship probability

# Access results
for team_id, scores in week_results.items():
    avg = np.mean(scores)
    std = np.std(scores)
    floor = np.percentile(scores, 25)
    ceiling = np.percentile(scores, 75)
```

---

## 6. ANALYSIS & ANALYTICS

```python
analytics = AnalyticsEngine()

# Strength of schedule (0-1, higher = harder)
sos = analytics.calculate_strength_of_schedule(team, league)

# Expected wins (actual vs luck)
exp_wins = analytics.calculate_expected_wins(pf=1000, pf_avg=950, pa_avg=940)

# Boom/bust potential (higher = more volatile)
volatility = analytics.calculate_boom_bust_potential(player)

# Find inconsistent players (trade candidates)
trade_candidates = analytics.identify_trade_candidates(
    league=league,
    position=Position.WR,
    threshold=0.4  # Consistency < 0.4
)
```

---

## 7. PLAYER STATISTICS & SCORING

```python
# Add game statistics
stats = PlayerStats(
    passing_yards=298,
    passing_td=2,
    interceptions=1,
    rushing_yards=25,
    rushing_td=0,
    receptions=8,
    receiving_yards=89,
    receiving_td=1
)
player.weekly_stats.append(stats)

# Calculate points from statistics
points = scoring_engine.calculate_points(stats)

# Get player's weekly scores
scores = player.get_weekly_scores(format=ScoringFormat.PPR)

# Get rolling average (last 4 weeks)
avg = player.get_rolling_average(weeks=4)

# Get season total
season_stats = player.get_season_stats()
games_played = player.get_games_played()
```

---

## 8. LEAGUE INFORMATION

```python
# Get standings
standings = league.get_standings()
# DataFrame: Team, Wins, Losses, Win%, PF, PA, Projection

# Get schedule
all_matchups = league.get_league_schedule()

# Get week schedule
week_matchups = league.get_week_schedule(week=8)

# Access team info
for team in league.teams:
    print(f"{team.owner_name}: {team.wins}-{team.losses}")
    print(f"  Points For: {team.points_for:.1f}")
    print(f"  Points Against: {team.points_against:.1f}")
```

---

## 9. COMMON WORKFLOWS

### Weekly Start/Sit Decision
```python
# Get projections for your team
projections = []
for player in team.roster:
    if player.status == PlayerStatus.OUT:
        continue
    mean, std = projection_engine.project_player_week(player, current_week)
    projections.append({
        'player': player.name,
        'position': player.position.value,
        'projection': mean,
        'floor': mean - std,
        'ceiling': mean + std,
        'consistency': player.consistency
    })

# Sort by projection
df = pd.DataFrame(projections).sort_values('projection', ascending=False)
print(df)  # Start the top players
```

### Trade Value Analysis
```python
# Current projection without trade
current_proj = team.get_season_projection()

# Remove and add player
team.roster.remove(give_player)
team.roster.add(get_player))

# New projection
new_proj = team.get_season_projection()

# Impact
impact = new_proj - current_proj
print(f"Trade impact: {impact:+.1f} points")  # Should be +30+ to justify disruption
```

### Identify Playoff Contenders
```python
results = simulation_engine.simulate_season(num_simulations=2000)

# Strong contenders
strong = results[results['Playoff_Probability'] > 0.75]
print(f"Will likely make playoffs: {len(strong)} teams")

# Long shots
long_shots = results[results['Playoff_Probability'] < 0.10]
print(f"Long shots: {', '.join(long_shots['Team'].tolist())}")
```

### Injury Impact
```python
# Simulate key player going out
player.status = PlayerStatus.OUT

# Get new projection
new_projection = team.get_season_projection()

# Restore and compare
player.status = PlayerStatus.HEALTHY
old_projection = team.get_season_projection()

print(f"Loss of {player.name}: {old_projection - new_projection:.1f} point impact")
```

---

## 10. KEY PLAYER ATTRIBUTES

```python
Player(
    # Identification
    name: str,                    # Player's name
    position: Position,           # QB, RB, WR, TE, K, DEF
    team: str,                    # NFL team (KC, NYG, etc)
    nfl_id: str,                  # Unique identifier
    status: PlayerStatus,         # HEALTHY, OUT, QUESTIONABLE, etc
    
    # Statistics
    weekly_stats: List[PlayerStats],  # Weekly game logs
    
    # Projections
    projection_mean: float,       # Expected points/week
    projection_std: float,        # Standard deviation
    consistency: float,           # Reliability 0-1
    floor: float,                 # 25th percentile
    ceiling: float,               # 75th percentile
    
    # Metrics
    target_share: float,          # % of team targets (WR/TE/RB)
    air_yards_share: float,       # % of air yards
    snap_count_pct: float,        # % of snaps played
    red_zone_touches: int,        # Opportunities in RZ
    
    # Advanced
    dvoa_adjusted: float,         # Defense-adjusted value
    matchup_ranking: int,         # Weekly matchup rank (1-17)
)
```

---

## 11. SCORING RULES BY FORMAT

### Standard Scoring
```python
{
    'passing_yards': 0.04,       # 1 pt per 25 yards
    'passing_td': 4,
    'interception': -2,
    'rushing_yards': 0.1,        # 1 pt per 10 yards
    'rushing_td': 6,
    'reception': 0,              # NO points per reception
    'receiving_yards': 0.1,
    'receiving_td': 6,
}
```

### PPR (Points Per Reception)
```python
# Same as Standard, but:
'reception': 1,                  # 1 pt per reception
```

### Half-PPR
```python
# Same as Standard, but:
'reception': 0.5,                # 0.5 pt per reception
```

---

## 12. SIMULATION TIPS

```python
# More simulations = more accurate but slower
# Recommended iterations:
# Early season (week 1-4): 5000+ iterations
# Mid season (week 5-13): 2000-3000 iterations
# Late season (week 14-17): 1000-2000 iterations
# Playoff simulations: 10000+ iterations

# Set random seed for reproducibility
simulation_engine = SimulationEngine(league, projection_engine)
results = simulation_engine.simulate_season(
    num_simulations=2000,
    seed=42  # Reproducible results
)
```

---

## 13. ERROR HANDLING

```python
# Check player status
if player.status == PlayerStatus.OUT:
    projection = 0
    
# Check games played (avoid small sample)
if player.get_games_played() < 4:
    # Use season average, not recent average
    proj = player.projection_mean
else:
    # Use recent form
    proj = player.get_rolling_average(weeks=4)

# Validate data
if team.points_for == 0 or team.points_against == 0:
    # No games played yet
    standings_valid = False
```

---

## 14. DATA EXPORT

```python
# Export to CSV
standings.to_csv('standings.csv', index=False)
projections.to_csv('week_projections.csv', index=False)

# Export to JSON
results.to_json('simulation_results.json', orient='records')

# Export to Excel
standings.to_excel('standings.xlsx', index=False)
```

---

## 15. DEBUGGING & VALIDATION

```python
# Check league structure
print(f"Teams: {len(league.teams)}")
print(f"Players: {sum(len(t.roster) for t in league.teams)}")
print(f"Current week: {league.current_week}")

# Check player
print(f"Games played: {player.get_games_played()}")
print(f"Recent average: {player.get_rolling_average():.2f}")
print(f"Consistency: {player.consistency:.2f}")

# Check projections reasonableness
assert 0 < player.projection_mean < 50  # Should be in range
assert player.projection_std > 0         # Should have variance
assert 0 <= player.consistency <= 1      # Should be 0-1
```

---

## QUICK REFERENCE METRICS

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| Consistency | 0-1 | 0.8+ = reliable, <0.5 = volatile |
| Target Share | 0-1 | >0.25 = elite, 0.15-0.20 = good |
| Snap Count % | 0-1 | >0.90 = locked in, 0.70-0.80 = part-time |
| Ceiling/Floor Gap | varies | Larger = more risky, smaller = safer |
| Boom/Bust Score | 0+ | >0.25 = high variance, <0.15 = consistent |
| Playoff Prob | 0-1 | >0.75 = lock, <0.25 = long shot |

---

## FORMULA REFERENCE

**Projection = Base_Mean × Recent_Form_Weight × Opponent_Multiplier × Situation_Factor**

**Floor = Projection - (1 × Standard_Deviation)**

**Ceiling = Projection + (1 × Standard_Deviation)**

**Expected_Wins = (PF²) / (PF² + PA²)**

**Boom_Bust_Score = Standard_Deviation / Mean**

---

**Last Updated**: November 2024  
**Framework Version**: 1.0.0
