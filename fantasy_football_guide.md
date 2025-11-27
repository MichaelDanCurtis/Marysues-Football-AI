# Fantasy Football: Complete Knowledge Base

## Table of Contents
1. [Introduction](#introduction)
2. [Rulesets & Scoring Systems](#rulesets--scoring-systems)
3. [League Formats](#league-formats)
4. [Outcomes & Performance Metrics](#outcomes--performance-metrics)
5. [Analysis Tools & Methodologies](#analysis-tools--methodologies)
6. [Prediction Frameworks](#prediction-frameworks)

---

## Introduction

Fantasy football is a game where participants draft real NFL players and earn points based on their actual statistical performance in NFL games. Managers compete weekly and seasonally, earning points from multiple scoring categories such as yardage, touchdowns, receptions, and defensive plays.

The evolution of fantasy football has shifted from gut-driven picks to data-driven analytics, with advanced statistical modeling now providing competitive edges through predictive modeling, situational analysis, and machine learning algorithms.

---

## Rulesets & Scoring Systems

### Standard Scoring Format (Non-PPR)

**Passing Stats:**
- Passing yards: 1 point per 25 yards (0.04 points per yard)
- Passing touchdowns: 4 points each
- Interceptions: -2 points each

**Rushing Stats:**
- Rushing yards: 1 point per 10 yards (0.1 points per yard)
- Rushing touchdowns: 6 points each

**Receiving Stats:**
- Receptions: 0 points (non-PPR)
- Receiving yards: 1 point per 10 yards (0.1 points per yard)
- Receiving touchdowns: 6 points each

**Special Stats:**
- 2-point conversions: 2 points
- Fumbles lost: -2 points

### PPR (Points Per Reception) Scoring

PPR has become the most common format. It's identical to standard scoring except:
- Receptions: 1 point per reception
- Half-PPR awards 0.5 points per reception (compromise format)

### Defense/Special Teams (DST) Scoring

- Sacks: 1 point each
- Interceptions: 2 points each
- Fumble recoveries: 2 points each
- Defensive touchdowns: 6 points each
- Safety: 2 points
- Tackle for loss: 1 point each
- Blocked kick/punt: 1 point each

**Points allowed scoring (variable by league):**
- 0 points allowed: 10 points
- 1-6 points allowed: 7 points
- 7-13 points allowed: 4 points
- 14-20 points allowed: 1 point
- 21+ points allowed: 0 points

### Kicker Scoring

- Field goal (0-39 yards): 3 points
- Field goal (40-49 yards): 4 points
- Field goal (50+ yards): 5 points
- Extra point (PAT): 1 point

### Individual Defensive Player (IDP) Scoring

Alternative to team defense where individual defensive players are drafted:
- Solo tackles: 1 point
- Assisted tackles: 0.5 points
- Sacks: 2 points
- Sack yards: 1 point per 10 yards
- Tackles for loss: 1 point
- Quarterback hits: 1 point
- Passes defended: 1 point
- Interceptions: 3 points
- Fumbles forced: 3 points
- Fumbles recovered: 3 points

---

## League Formats

### Redraft Leagues
- Annual draft each season
- Standard 12-team format
- Snake draft order (alternating picks per round)
- Waiver wire management throughout season
- 17-week regular season with playoffs

### Dynasty Leagues
- Players retained season-to-season
- Multiple-year dynasty (often 3-5 years)
- Creates long-term team building strategy
- Rookie drafts occurring annually
- Higher emphasis on age/longevity

### Daily Fantasy Sports (DFS)
- Weekly salary cap based contests
- New lineups drafted each week
- Contests vary: GPP (large prize pools), H2H (head-to-head)
- Focus on matchups and weekly trends
- Multiple entry options

### Keeper Leagues
- Hybrid between redraft and dynasty
- Retain select players (typically 2-4)
- Auction draft or snake draft for remaining players
- Opportunity cost analysis critical

### Superflex/2QB Leagues
- Additional flex position requiring QB start
- QBs become more valuable
- Deeper roster requirements
- Different draft value distributions

### PPR vs Non-PPR Trade-offs
- PPR: Values consistency, receivers, slot receivers, and pass-catching backs
- Standard: Emphasizes touchdowns and yardage volume
- PPR typically increases scoring by 20-30% overall

---

## Outcomes & Performance Metrics

### Win Probability Metrics

**Win Probability Score (WPS):**
- Cumulative probability of defeating all opponents
- Accounts for schedule difficulty
- Projects playoff likelihoods

**Expected Wins (xW):**
- Statistical projection based on point differential
- Accounts for luck and randomness
- More predictive than actual wins in early season

### Consistency Metrics

**Consistency Score:**
- Standard deviation of weekly scoring
- Lower variance indicates more reliable player
- Critical for playoff predictability

**Floor vs Ceiling:**
- Floor: Projected minimum expected points (25th percentile)
- Ceiling: Projected maximum expected points (75th percentile)
- Range indicates risk vs reward

### Player Performance Outcomes

**Boom-or-Bust:**
- High variance between performances
- Large gap between floor and ceiling
- RBs in time-share situations, rookie WRs

**Reliable/Safe:**
- Consistent week-to-week output
- Narrow floor-ceiling range
- Established WR1s, pass-catching backs

**Upside:**
- Projected ceiling substantially higher than average
- Typically younger players or role-poaching situations

### Team Performance Outcomes

**Strength of Schedule (SOS):**
- Quality of opponents remaining
- DVOA-based rankings preferred over points allowed
- Critical for playoff matchups

**Playoff Position:**
- Bracket seeding determines path to championship
- Early seeding heavily favors team
- Bye weeks impact roster construction

---

## Analysis Tools & Methodologies

### Data Collection Sources

**Public APIs & Datasets:**
- Sports-Reference.com (NFL historical stats)
- NFL.com (official statistics)
- Vegas odds (spreads, over/unders)
- Air yards data (advanced metrics)
- Red zone metrics (situational data)

**Web Scraping Tools:**
- BeautifulSoup (Python)
- Scrapy (large-scale scraping)
- Selenium (dynamic content)

### Advanced Metrics

**WOPR (Weighted Opportunity Rating):**
- Measures receiver opportunity quality
- Combines air yards and target share
- More predictive than volume alone
- >8% WOPR signals elite receiver

**Target Share & Air Yards:**
- Target share: Percentage of team's targets
- Air yards: Total yards thrown to player
- 25%+ target share indicates alpha player

**Rushing Share & Red Zone Touches:**
- Rushing attempts percentage of team
- Red zone touches highly predictive
- TD regression analysis critical

**Defense-Adjusted Value Over Average (DVOA):**
- NFL's advanced efficiency metric
- Accounts for opponent strength
- More predictive than points allowed
- Available from Football Outsiders

### Statistical Analysis Techniques

**Regression Analysis:**
- Linear regression for baseline projections
- Multiple regression incorporating multiple variables
- Identifies strongest predictive features
- Foundation of most projection systems

**Time Series Analysis:**
- Moving averages (4-week, 8-week)
- Trend identification and momentum
- Accounts for mid-season changes
- Captures form entering playoffs

**Clustering Analysis:**
- Groups similar players by profile
- Identifies comparable players
- Trade evaluation framework
- Values adjustment for replacements

### Machine Learning Approaches

**Gradient Boosting Models:**
- XGBoost, LightGBM for ensemble predictions
- Captures non-linear relationships
- Typically 5-10% improvement over regression
- Requires substantial training data

**Neural Networks:**
- Deep learning for complex patterns
- Requires large datasets
- Marginal improvements over boosting
- Computationally expensive

**Random Forest Ensembles:**
- Multiple decision trees
- Feature importance analysis
- Handles categorical data well
- Robust to outliers

### Bayesian Methods

**Prior Beliefs:**
- Start with league-wide averages as priors
- Update with new information (injuries, matchups)
- Quantifies uncertainty bands
- Reduces false confidence

---

## Prediction Frameworks

### Week-to-Week Projections

**Input Variables:**
- Historical player performance (season average, rolling average)
- Opponent defense metrics (DVOA, points allowed, pace)
- Matchup factors (weather, home/away, rest)
- Vegas over/under (correlates with pace and scoring)
- Injury status and snap count percentages
- QB change impacts
- Weather data

**Process:**
1. Collect baseline league-wide scoring distribution
2. Adjust for player-specific recent form
3. Apply opponent adjustment (multiplicative factor)
4. Add positional trend factors
5. Adjust for situation (home/away, rest, weather)
6. Generate confidence intervals (standard error)

**Accuracy Expectations:**
- Best case: 15-25% better than baseline
- Most analysts: 5-10% improvement
- Regression to mean as season progresses

### Season Projections

**Methodology:**
1. Project weekly output for entire remaining schedule
2. Sum projections across season
3. Incorporate consistency (volatility adjustments)
4. Account for injury likelihood
5. Adjust for role changes and regression

**Key Variables:**
- Remaining strength of schedule
- Role stability probability
- Age and career trajectory
- Contract/incentive structures
- Team offensive tendency changes

### Win Probability Simulations

**Monte Carlo Simulation Approach:**
1. Generate player performance distributions (normal or empirical)
2. Randomly sample from distributions for each player for each week
3. Calculate weekly scores across all matchups
4. Simulate entire remaining season (1000-10000 iterations)
5. Calculate win probability from simulation outcomes
6. Identify championship probability

**Process:**
- Each player has mean (projection) and standard deviation (consistency)
- Weekly outcomes drawn from distribution
- Playoff bracket outcomes simulated
- Championship distribution calculated

### Predictive Modeling Best Practices

**Data Quality:**
- Remove outliers (injuries, extreme performances)
- Handle missing data appropriately
- Normalize features for regression
- Ensure data independence (no lookahead bias)

**Model Validation:**
- Train/test split (80/20 or temporal)
- Cross-validation across seasons
- Out-of-sample testing
- Backtesting on historical seasons

**Handling Uncertainty:**
- Projection confidence intervals
- Standard error estimates
- Sensitivity analysis
- Scenario planning

**Avoiding Common Pitfalls:**
- Overconfidence bias (regression matters)
- Recency bias (small sample noise)
- Narrative bias (story over stats)
- Survivor bias (successful teams only)

### Integration with Lineup Optimization

**Salary Cap Optimization (DFS):**
- Input: Player projections and salary
- Constraint: Max salary cap (typically $50,000)
- Constraints: Position requirements
- Output: Optimal lineup using integer linear programming
- Tools: PuLP, Gurobi, or specialized DFS software

**Weekly Start/Sit Decisions:**
- Compare to bench alternatives
- Consider volatility and ceiling
- Account for bye weeks
- Evaluate stack potential

**Trade Analysis:**
- Calculate replacement level at positions
- Project remaining season value
- Account for league depth at position
- Consider playoff schedule advantage

---

## Summary: Building a Winning System

1. **Data Foundation**: Collect comprehensive historical and current player data
2. **Statistical Model**: Start with regression, validate cross-season
3. **Advanced Metrics**: Incorporate targets, air yards, DVOA, snap counts
4. **Flexibility**: Adjust for injuries, role changes, matchups weekly
5. **Simulation**: Use Monte Carlo for uncertainty quantification
6. **Optimization**: Optimize lineups/rosters within constraints
7. **Continuous Learning**: Backtest, validate, and refine throughout season
8. **Risk Management**: Understand your confidence intervals and limits

The combination of rigorous statistical analysis, advanced metrics, and simulation-based prediction creates a foundation for competitive advantage in fantasy football.
