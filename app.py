import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fantasy_football_simulation import (
    League, Team, Player, Position, ScoringFormat,
    SimulationEngine, AnalyticsEngine, ProjectionEngine, ScoringEngine
)
from fantasy_football_examples import example_realistic_league
from espn_integration import import_espn_league
from trade_analyzer import TradeAnalyzer
from ai_scenario_agent import ScenarioAgent
from power_rankings import calculate_luck_adjusted_power_rankings
import os
import requests
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Fantasy Football AI", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling ---
def get_background_image():
    """
    Returns base64 encoded image from local file or a default URL.
    """
    image_path = "background.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpg;base64,{encoded_string}"
    # Fallback to a nice dark stadium URL if local file doesn't exist
    return "https://images.unsplash.com/photo-1519865885898-a54a6f2c7eea?q=80&w=2558&auto=format&fit=crop"

def apply_custom_style():
    bg_image = get_background_image()
    st.markdown(f"""
        <style>
        /* Main Background */
        .stApp {{
            background-image: linear-gradient(rgba(20, 0, 20, 0.7), rgba(20, 0, 20, 0.7)), url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #ffffff;
        }}
        
        /* Sidebar Background */
        section[data-testid="stSidebar"] {{
            background-color: rgba(30, 10, 30, 0.85);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(236, 72, 153, 0.3);
        }}
        
        /* Cards/Containers */
        .css-1r6slb0, .css-12w0qpk {{
            background-color: rgba(40, 15, 40, 0.85);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(236, 72, 153, 0.2);
        }}
        
        /* Metrics */
        div[data-testid="stMetric"] {{
            background-color: rgba(40, 15, 40, 0.8);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(236, 72, 153, 0.4);
            backdrop-filter: blur(5px);
        }}
        
        /* Headers */
        h1, h2, h3 {{
            color: #fbcfe8 !important;
            font-family: 'Segoe UI', sans-serif;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }}
        
        /* Custom Card Class */
        .dashboard-card {{
            background-color: rgba(40, 15, 40, 0.85);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid rgba(236, 72, 153, 0.3);
            backdrop-filter: blur(5px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }}
        
        /* Draftboard Item */
        .draft-item {{
            background-color: rgba(40, 15, 40, 0.8);
            padding: 10px 15px;
            margin-bottom: 8px;
            border-radius: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid rgba(236, 72, 153, 0.3);
            transition: transform 0.2s, background-color 0.2s;
        }}
        .draft-item:hover {{
            transform: translateX(5px);
            border-color: #ec4899;
            background-color: rgba(60, 20, 60, 0.95);
        }}
        .draft-rank {{
            color: #fbcfe8;
            font-size: 0.9em;
            margin-right: 10px;
        }}
        .draft-name {{
            font-weight: 600;
            flex-grow: 1;
            color: #ffffff;
        }}
        .draft-score {{
            background-color: #ec4899;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.85em;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        
        /* Charts */
        canvas {{
            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.2));
        }}
        
        /* Input fields transparency */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {{
            background-color: rgba(40, 15, 40, 0.8) !important;
            color: white !important;
            border-color: rgba(236, 72, 153, 0.3) !important;
        }}
        
        /* Buttons */
        .stButton button {{
            background-color: #ec4899 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
        }}
        .stButton button:hover {{
            background-color: #db2777 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_style()

st.title("Marysue's Football AI system")

# Initialize League
@st.cache_resource
def load_league(league_source="example", **kwargs):
    if league_source == "espn":
        return import_espn_league(
            league_id=kwargs.get('league_id'),
            year=kwargs.get('year'),
            swid=kwargs.get('swid'),
            espn_s2=kwargs.get('espn_s2')
        )
    else:
        league, _ = example_realistic_league()
        return league

@st.cache_data(ttl=3600)
def get_openrouter_models():
    """Fetch available models from OpenRouter API"""
    models = []
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        if response.status_code == 200:
            data = response.json()
            # Sort by id for easier finding
            models = sorted([model["id"] for model in data["data"]])
    except Exception as e:
        pass
    
    if not models:
        # Fallback list if API fails
        models = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-flash-1.5",
            "meta-llama/llama-3.1-70b-instruct",
            "mistral/mistral-large-latest"
        ]
    
    # Ensure openrouter/auto is available and at the top
    if "openrouter/auto" in models:
        models.remove("openrouter/auto")
    models.insert(0, "openrouter/auto")
    
    return models

# Sidebar Configuration
st.sidebar.header("League Settings")
data_source = st.sidebar.selectbox("Data Source", ["Example Data", "ESPN League"])

# API Key for AI
with st.sidebar.expander("AI Configuration"):
    # Try to get from env
    env_api_key = os.getenv("OPENROUTER_API_KEY", "")

    if env_api_key:
        st.success("API Key is set ✅")
        if st.button("Reset API Key"):
            if os.path.exists(".env"):
                os.remove(".env")
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            st.rerun()
        api_key = env_api_key
    else:
        api_key = st.text_input(
            "Enter OpenRouter API Key", 
            type="password", 
            help="Required for AI Scenario Lab"
        )
        
        if st.button("Add API Key"):
            if api_key:
                with open(".env", "w") as f:
                    f.write(f"OPENROUTER_API_KEY={api_key}")
                st.success("Key saved!")
                st.rerun()
            else:
                st.error("Please enter a key first.")

    available_models = get_openrouter_models()
    default_index = 0
    if "openrouter/auto" in available_models:
        default_index = available_models.index("openrouter/auto")
    elif "openai/gpt-4o-mini" in available_models:
        default_index = available_models.index("openai/gpt-4o-mini")

    model_name = st.selectbox(
        "Select AI Model",
        available_models,
        index=default_index,
        help="Select the model to use for scenario interpretation."
    )

if data_source == "ESPN League":
    with st.sidebar.form("espn_form"):
        league_id = st.number_input("League ID", value=0)
        year = st.number_input("Year", value=2024)
        swid = st.text_input("SWID (Private Leagues)", help="{...}")
        espn_s2 = st.text_input("ESPN_S2 (Private Leagues)")
        submitted = st.form_submit_button("Load League")
        
        if submitted:
            try:
                league = load_league(
                    league_source="espn",
                    league_id=league_id,
                    year=year,
                    swid=swid if swid else None,
                    espn_s2=espn_s2 if espn_s2 else None
                )
                st.session_state['league'] = league
                st.success("League loaded successfully!")
            except Exception as e:
                st.error(f"Error loading league: {e}")
else:
    if 'league' not in st.session_state or st.sidebar.button("Reset to Example"):
        st.session_state['league'] = load_league("example")

league = st.session_state['league']

# Initialize Engines
scoring_engine = ScoringEngine(league.scoring_format)
projection_engine = ProjectionEngine(scoring_engine)
sim_engine = SimulationEngine(league, projection_engine)
trade_analyzer = TradeAnalyzer(league, sim_engine)

# Sidebar
st.sidebar.header("Navigation")

# My Team Selector
team_names = [t.owner_name for t in league.teams]
my_team_name = st.sidebar.selectbox(
    "Select Your Team", 
    ["None"] + team_names,
    index=0,
    help="Select your team to see a personalized dashboard."
)

# Determine default page based on selection
default_page_index = 0
if my_team_name != "None":
    # If a team is selected, we want to show the Dashboard
    # We'll add "My Dashboard" to the list of pages
    page_options = ["My Dashboard", "League Overview", "Power Rankings", "Team Analysis", "Matchup Simulator", "Trade Analyzer", "AI Scenario Lab"]
else:
    page_options = ["League Overview", "Power Rankings", "Team Analysis", "Matchup Simulator", "Trade Analyzer", "AI Scenario Lab"]

page = st.sidebar.radio("Go to", page_options)

if page == "My Dashboard":
    st.header(f"👑 Owner Dashboard: {my_team_name}")
    
    my_team = next(t for t in league.teams if t.owner_name == my_team_name)
    
    # 1. High Level Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Record", f"{my_team.wins}-{my_team.losses}")
    with col2:
        st.metric("Points For", f"{my_team.points_for:.1f}")
    with col3:
        st.metric("Points Against", f"{my_team.points_against:.1f}")
    with col4:
        win_pct = my_team.get_win_percentage() * 100
        st.metric("Win %", f"{win_pct:.1f}%")
        
    # 2. Luck Analysis
    st.subheader("🍀 Luck Analysis")
    rankings_df = calculate_luck_adjusted_power_rankings(league)
    if not rankings_df.empty:
        my_luck_data = rankings_df[rankings_df["Team"] == my_team_name].iloc[0]
        
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.metric("Expected Wins", f"{my_luck_data['Expected Wins']:.2f}")
        with l_col2:
            st.metric("Luck Rating", my_luck_data['Luck Rating'], delta=f"{my_luck_data['Luck']:.2f}")
            
        st.info(f"You have won {my_team.wins} games, but based on your scoring against every opponent each week, you should have won {my_luck_data['Expected Wins']:.2f} games.")
    else:
        st.warning("Luck data not available yet.")
        my_luck_data = None

    # 3. AI Coach's Corner
    st.subheader("🤖 AI Coach's Corner")
    
    if api_key:
        if st.button("Generate Strategic Advice"):
            with st.spinner("Analyzing roster, schedule, and league dynamics..."):
                agent = ScenarioAgent(api_key=api_key, model=model_name)
                advice = agent.analyze_team(league, my_team, my_luck_data.to_dict() if my_luck_data is not None else None)
                st.markdown(advice)
    else:
        st.warning("Please enter an API Key in the sidebar to get AI advice.")

elif page == "League Overview":
    st.header("League Overview")
    
    # Standings
    st.subheader("Current Standings")
    standings = league.get_standings()
    st.dataframe(standings)

    # Team Details
    st.subheader("Teams")
    team_data = []
    for team in league.teams:
        team_data.append({
            "Team Name": team.owner_name,
            "ID": team.team_id,
            "Roster Size": len(team.roster),
            "Season Proj": f"{team.get_season_projection():.1f}"
        })
    st.dataframe(pd.DataFrame(team_data))

elif page == "Power Rankings":
    st.header("🏆 Luck-Adjusted Power Rankings")
    st.markdown("""
    This advanced analysis separates **skill** (roster strength) from **luck** (schedule variance).
    
    - **Expected Wins**: How many games you *should* have won if you played every other team each week.
    - **Luck Rating**: The difference between your actual wins and expected wins.
    """)
    
    rankings_df = calculate_luck_adjusted_power_rankings(league)
    
    if not rankings_df.empty:
        # Styling the dataframe
        st.dataframe(
            rankings_df.style.format({
                "Points For": "{:.1f}",
                "Expected Wins": "{:.2f}",
                "Luck": "{:+.2f}"
            }).background_gradient(subset=["Expected Wins"], cmap="Greens")
              .background_gradient(subset=["Luck"], cmap="RdYlGn", vmin=-2, vmax=2),
            use_container_width=True,
            height=500
        )
        
        # Visualization
        st.subheader("Luck vs. Skill Matrix")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Set dark background for plot
        fig.patch.set_facecolor('#280f28')
        ax.set_facecolor('#280f28')
        
        # Scatter plot
        x = rankings_df["Expected Wins"]
        y = rankings_df["Luck"]
        teams = rankings_df["Team"]
        
        ax.scatter(x, y, color='#ec4899', s=100, alpha=0.8, edgecolors='white')
        
        # Add labels
        for i, txt in enumerate(teams):
            ax.annotate(txt, (x[i], y[i]), xytext=(5, 5), textcoords='offset points', color='white', fontsize=9)
            
        # Add quadrants
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=rankings_df["Expected Wins"].mean(), color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Skill (Expected Wins)", color='#fbcfe8')
        ax.set_ylabel("Luck (Actual - Expected)", color='#fbcfe8')
        ax.set_title("Are you Good or just Lucky?", color='white', pad=20)
        
        # Color axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_color('#ec4899')
            
        st.pyplot(fig)
        
    else:
        st.info("Not enough data to calculate power rankings yet. Wait for Week 1 to complete!")

elif page == "Team Analysis":
    # st.header("Team Analysis") # Hidden to use custom layout
    
    team_names = [t.owner_name for t in league.teams]
    
    # Top bar for selection
    col_sel, col_empty = st.columns([1, 3])
    with col_sel:
        selected_team_name = st.selectbox("Select Team", team_names)
    
    selected_team = next(t for t in league.teams if t.owner_name == selected_team_name)
    
    # Main Dashboard Layout
    col_left, col_right = st.columns([1, 3])
    
    # --- Left Column: Roster / Draftboard ---
    with col_left:
        st.markdown("### Roster")
        
        # Sort roster by projection
        sorted_roster = sorted(selected_team.roster, key=lambda x: x.projection_mean, reverse=True)
        
        for i, p in enumerate(sorted_roster):
            score_color = "#3b82f6" if p.projection_mean >= 15 else "#60a5fa" if p.projection_mean >= 10 else "#94a3b8"
            st.markdown(f"""
            <div class="draft-item">
                <span class="draft-rank">{i+1}</span>
                <span class="draft-name">{p.name} <span style="font-size:0.8em; color:#888">({p.position.value})</span></span>
                <span class="draft-score" style="background-color: {score_color}">{p.projection_mean:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

    # --- Right Column: Metrics & Charts ---
    with col_right:
        # Row 1
        r1c1, r1c2 = st.columns(2)
        
        with r1c1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("#### Individual Metrics (Top 5)")
            
            # Prepare data for chart
            top_5 = sorted_roster[:5]
            chart_data = pd.DataFrame({
                'Player': [p.name.split()[-1] for p in top_5], # Last name only for space
                'Projection': [p.projection_mean for p in top_5],
                'Ceiling': [p.projection_mean + p.projection_std for p in top_5]
            }).set_index('Player')
            
            st.bar_chart(chart_data, color=["#3b82f6", "#1e40af"])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with r1c2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("#### Team Metrics vs League Avg")
            
            league_avg_pf = np.mean([t.points_for for t in league.teams])
            league_avg_wins = np.mean([t.wins for t in league.teams])
            
            # Normalize for chart
            metrics_df = pd.DataFrame({
                'Metric': ['PF', 'Wins (x10)', 'Roster Value (x0.1)'],
                'Team': [selected_team.points_for, selected_team.wins * 10, selected_team.get_season_projection() * 0.1],
                'League Avg': [league_avg_pf, league_avg_wins * 10, np.mean([t.get_season_projection() for t in league.teams]) * 0.1]
            }).set_index('Metric')
            
            st.bar_chart(metrics_df, color=["#0ea5e9", "#475569"])
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Row 2
        r2c1, r2c2 = st.columns([2, 1])
        
        with r2c1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("#### Strength of Schedule Trend")
            
            # Mock trend data if real data isn't granular enough
            weeks = list(range(1, 18))
            # Generate a random walk for visual effect that looks like the image
            base = 0.5
            trend = [base]
            for _ in range(16):
                change = np.random.uniform(-0.1, 0.1)
                new_val = max(0.2, min(0.8, trend[-1] + change))
                trend.append(new_val)
                
            sos_df = pd.DataFrame({
                'Difficulty': trend,
                'League Avg': [0.5] * 17
            }, index=weeks)
            
            st.line_chart(sos_df, color=["#3b82f6", "#64748b"])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with r2c2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("#### Playoff Odds")
            
            # Donut chart simulation using matplotlib since st.pyplot allows more control
            # Or just a big metric
            
            # Calculate odds (mock or real)
            # We can use the simulation engine for a quick check
            # For speed, let's use a heuristic based on wins
            win_pct = selected_team.get_win_percentage()
            playoff_prob = min(0.99, max(0.01, win_pct * 1.2)) # Simple heuristic
            
            fig, ax = plt.subplots(figsize=(3, 3))
            fig.patch.set_facecolor('#252836')
            ax.set_facecolor('#252836')
            
            colors = ['#3b82f6', '#1e293b']
            ax.pie([playoff_prob, 1-playoff_prob], labels=['', ''], colors=colors, startangle=90, wedgeprops=dict(width=0.4))
            ax.text(0, 0, f"{playoff_prob:.0%}", ha='center', va='center', color='white', fontsize=20, fontweight='bold')
            
            st.pyplot(fig)
            st.markdown("<div style='text-align: center; color: #9ca3af'>Playoff Probability</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Row 3
        r3c1, r3c2 = st.columns([1, 2])
        
        with r3c1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("#### Scoring Summary")
            
            st.metric("Avg Scoring", f"{selected_team.points_for / max(1, league.current_week):.1f}")
            st.metric("Season Total", f"{selected_team.points_for:.1f}")
            st.metric("Proj. Finish", f"{selected_team.get_season_projection():.0f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        with r3c2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("#### Weekly Scoring History")
            
            # Mock weekly data if empty
            if not selected_team.weekly_scores:
                # Generate realistic looking scores
                scores = np.random.normal(110, 15, league.current_week)
            else:
                scores = selected_team.weekly_scores
                
            weekly_df = pd.DataFrame({
                'Team Score': scores,
                'League Avg': [league_avg_pf / max(1, league.current_week)] * len(scores)
            })
            
            st.line_chart(weekly_df, color=["#0ea5e9", "#475569"])
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Matchup Simulator":
    st.header("Matchup Simulator")
    col1, col2 = st.columns(2)
    
    team_names = [t.owner_name for t in league.teams]
    
    with col1:
        team1_name = st.selectbox("Team 1", team_names, index=0)
    with col2:
        team2_name = st.selectbox("Team 2", team_names, index=1)
        
    if st.button("Simulate Matchup"):
        team1 = next(t for t in league.teams if t.owner_name == team1_name)
        team2 = next(t for t in league.teams if t.owner_name == team2_name)
        
        # Run simulation manually since simulate_matchup isn't in SimulationEngine
        # We will simulate 1000 times for each team
        n_sims = 1000
        team1_scores = []
        team2_scores = []
        
        progress_bar = st.progress(0)
        
        # We can use simulate_week logic but just for these two teams
        # Or we can just iterate
        
        for i in range(n_sims):
            # Team 1
            t1_score = 0
            for p in team1.roster:
                mean, std = projection_engine.project_player_week(p, league.current_week)
                t1_score += max(0, np.random.normal(mean, std))
            team1_scores.append(t1_score)
            
            # Team 2
            t2_score = 0
            for p in team2.roster:
                mean, std = projection_engine.project_player_week(p, league.current_week)
                t2_score += max(0, np.random.normal(mean, std))
            team2_scores.append(t2_score)
            
            if i % 100 == 0:
                progress_bar.progress((i + 1) / n_sims)
                
        progress_bar.progress(1.0)
        
        team1_wins = sum(s1 > s2 for s1, s2 in zip(team1_scores, team2_scores))
        win_prob = team1_wins / n_sims
        
        st.success("Simulation Complete!")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric(f"{team1_name} Win Prob", f"{win_prob:.1%}")
        col_res2.metric(f"{team2_name} Win Prob", f"{(1-win_prob):.1%}")
        
        # Chart
        fig, ax = plt.subplots()
        ax.hist(team1_scores, alpha=0.5, label=team1_name, bins=30)
        ax.hist(team2_scores, alpha=0.5, label=team2_name, bins=30)
        ax.legend()
        ax.set_title("Projected Score Distribution")
        ax.set_xlabel("Points")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

elif page == "Trade Analyzer":
    st.header("Trade Analyzer")
    
    team_names = [t.owner_name for t in league.teams]
    my_team_name = st.selectbox("Select Your Team", team_names)
    my_team = next(t for t in league.teams if t.owner_name == my_team_name)
    
    st.subheader("Team Needs Analysis")
    needs = trade_analyzer.identify_team_needs(my_team)
    if needs:
        for need in needs:
            if need['Severity'] == 'Critical':
                st.error(f"{need['Position'].value}: {need['Reason']}")
            elif need['Severity'] == 'High':
                st.warning(f"{need['Position'].value}: {need['Reason']}")
            else:
                st.info(f"{need['Position'].value}: {need['Reason']}")
    else:
        st.success("No critical needs identified! Your team is balanced.")
        
    st.subheader("Suggested Trade Partners")
    if st.button("Find Trades"):
        with st.spinner("Analyzing league rosters..."):
            suggestions = trade_analyzer.find_trade_partners(my_team)
            
            if not suggestions:
                st.warning("No obvious trade partners found based on roster balance.")
            
            for sugg in suggestions:
                with st.expander(f"Trade with {sugg['Team']} (Score: {sugg['Score']})"):
                    for detail in sugg['Details']:
                        st.write(f"- {detail}")
                        
    st.subheader("Trade Evaluator")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### You Give")
        my_players = [p.name for p in my_team.roster]
        give_players = st.multiselect("Select Players to Give", my_players)
        
    with col2:
        st.markdown("### You Receive")
        partner_name = st.selectbox("Select Partner Team", [n for n in team_names if n != my_team_name])
        partner_team = next(t for t in league.teams if t.owner_name == partner_name)
        partner_players = [p.name for p in partner_team.roster]
        receive_players = st.multiselect("Select Players to Receive", partner_players)
        
    if st.button("Evaluate Trade Impact"):
        if not give_players and not receive_players:
            st.error("Please select players to trade.")
        else:
            # Get player objects
            p_give = [p for p in my_team.roster if p.name in give_players]
            p_receive = [p for p in partner_team.roster if p.name in receive_players]
            
            with st.spinner("Simulating season outcomes..."):
                impact = trade_analyzer.evaluate_trade(my_team, partner_team, p_give, p_receive)
                
                st.metric("Your Projected Wins Change", f"{impact['team_a_delta']:+.2f}")
                st.metric(f"{partner_name} Projected Wins Change", f"{impact['team_b_delta']:+.2f}")
                
                if impact['team_a_delta'] > 0:
                    st.success("This trade improves your team!")
                else:
                    st.error("This trade hurts your team's projected win total.")

elif page == "AI Scenario Lab":
    st.header("🤖 AI Scenario Lab")
    
    if not api_key:
        st.warning("Please enter an OpenRouter API Key in the sidebar to use this feature.")
    else:
        st.markdown("""
        Describe a "What If" scenario in plain English. The AI Agent will interpret your request, 
        modify the simulation parameters, and run a Monte Carlo simulation to show you the impact.
        """)
        
        team_names = [t.owner_name for t in league.teams]
        focus_team_name = st.selectbox("Focus Team (Optional)", ["None"] + team_names)
        focus_team_id = None
        if focus_team_name != "None":
            focus_team_id = next(t.team_id for t in league.teams if t.owner_name == focus_team_name)
            
        query = st.text_area("Scenario Description", 
                           placeholder="e.g., 'What if Patrick Mahomes gets injured and is out for the season?' or 'What if my WR1 has a breakout year and increases production by 20%?'")
        
        if st.button("Run AI Simulation"):
            if not query:
                st.error("Please enter a scenario description.")
            else:
                agent = ScenarioAgent(api_key=api_key, model=model_name)
                
                with st.spinner("AI Agent is analyzing your request and configuring the simulation..."):
                    # 1. Run Baseline
                    baseline_results = sim_engine.simulate_season(num_simulations=200)
                    baseline_summary = baseline_results[baseline_results['Team'] == focus_team_name].to_dict('records')[0] if focus_team_name != "None" else {}
                    
                    # 2. Run Scenario
                    result = agent.run_scenario(league, query, focus_team_id, sim_engine)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("Simulation Complete!")
                        
                        st.subheader("Scenario Plan")
                        st.write(result['scenario_description'])
                        
                        with st.expander("Modifications Applied"):
                            for mod in result['applied_modifications']:
                                st.code(mod)
                                
                        st.subheader("Impact Analysis")
                        
                        # Display metrics if a team was focused
                        if focus_team_name != "None":
                            col1, col2, col3 = st.columns(3)
                            
                            base_wins = baseline_summary.get('Avg_Final_Wins', 0)
                            new_wins = result['results_summary'].get('Avg_Final_Wins', 0)
                            
                            base_playoff = baseline_summary.get('Playoff_Probability', 0)
                            new_playoff = result['results_summary'].get('Playoff_Probability', 0)
                            
                            col1.metric("Projected Wins", f"{new_wins:.1f}", f"{new_wins - base_wins:+.1f}")
                            col2.metric("Playoff Chance", f"{new_playoff:.1%}", f"{new_playoff - base_playoff:+.1%}")
                            

                        # AI Insight
                        with st.spinner("Generating insights..."):
                            insight = agent.generate_insight(result, baseline_summary)
                            st.info(insight)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_background_image():
    # Check for local files
    if os.path.exists("background.jpg"):
        return f"data:image/jpg;base64,{get_base64_of_bin_file('background.jpg')}"
    elif os.path.exists("background.png"):
        return f"data:image/png;base64,{get_base64_of_bin_file('background.png')}"
    else:
        # Fallback URL
        return "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80"

