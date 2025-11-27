import os
import json
import copy
from typing import List, Dict, Any
from openai import OpenAI
from fantasy_football_simulation import League, Team, Player, SimulationEngine, ProjectionEngine, ScoringEngine

class ScenarioAgent:
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def analyze_team(self, league: League, team: Team, luck_data: Dict = None) -> str:
        """
        Generate strategic advice for a specific team.
        """
        roster_str = "\n".join([f"- {p.name} ({p.position.value}): Proj={p.projection_mean:.1f}" for p in team.roster])
        
        luck_context = ""
        if luck_data:
            luck_context = f"""
            Power Ranking Context:
            - Expected Wins: {luck_data.get('Expected Wins', 'N/A')}
            - Luck Rating: {luck_data.get('Luck Rating', 'N/A')}
            """

        prompt = f"""
        You are a Fantasy Football Head Coach and GM.
        Analyze this team and provide 3 specific, actionable recommendations.
        
        Team: {team.owner_name}
        Record: {team.wins}-{team.losses}
        Points For: {team.points_for}
        {luck_context}
        
        Roster:
        {roster_str}
        
        Provide the output in Markdown format with these sections:
        ### 📋 State of the Union
        (Brief summary of team health and luck)
        
        ### 🧠 Strategic Recommendations
        1. **[Action Item 1]**: Detail
        2. **[Action Item 2]**: Detail
        3. **[Action Item 3]**: Detail
        
        Keep it concise, encouraging but realistic.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert fantasy football coach."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating advice: {str(e)}"

    def parse_scenario(self, league: League, query: str, focus_team_id: str = None) -> Dict[str, Any]:
        """
        Ask the LLM to interpret a natural language scenario and determine
        how to modify player projections or statuses.
        """
        
        # Context building
        context = "Current League Context:\n"
        if focus_team_id:
            team = next((t for t in league.teams if t.team_id == focus_team_id), None)
            if team:
                context += f"Focus Team: {team.owner_name}\n"
                context += "Roster:\n"
                for p in team.roster:
                    context += f"- {p.name} ({p.position.value}): Proj={p.projection_mean}, Status={p.status.value}\n"
        
        prompt = f"""
        You are a Fantasy Football Simulation Expert.
        The user wants to run a "What If" scenario simulation.
        
        {context}
        
        User Query: "{query}"
        
        Based on this query, determine which players need to be modified.
        You can modify:
        - 'projection_mean': The projected points (float)
        - 'projection_std': The volatility/variance (float)
        - 'status': 'healthy', 'questionable', 'doubtful', 'out'
        
        Return a JSON object with this structure:
        {{
            "modifications": [
                {{
                    "player_name": "Exact Name",
                    "attribute": "projection_mean", 
                    "value": 25.5,
                    "reason": "Explanation"
                }}
            ],
            "description": "A brief description of the scenario being simulated."
        }}
        
        If the user mentions a player not in the list, try to infer or ignore if impossible.
        Only return the JSON.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            return {"error": str(e), "modifications": [], "description": "Failed to parse scenario."}

    def run_scenario(self, league: League, query: str, focus_team_id: str, simulation_engine: SimulationEngine) -> Dict[str, Any]:
        """
        1. Parse scenario
        2. Clone league
        3. Apply mods
        4. Run sim
        5. Return results
        """
        
        # 1. Parse
        scenario_plan = self.parse_scenario(league, query, focus_team_id)
        
        if "error" in scenario_plan:
            return scenario_plan

        # 2. Clone League (Deep copy to avoid affecting main state)
        # Deepcopying complex objects can be tricky, so we might need a lighter approach
        # For now, let's try deepcopy, if it fails we'll manually copy the relevant team
        try:
            sim_league = copy.deepcopy(league)
        except Exception:
            # Fallback: Just modify the live league and revert later? 
            # Or just copy the specific team we are analyzing?
            # Let's assume deepcopy works for our dataclasses
            return {"error": "Failed to clone league state for simulation."}

        # 3. Apply Modifications
        applied_mods = []
        
        # Helper to find player in new league
        def find_player(name):
            for t in sim_league.teams:
                for p in t.roster:
                    if p.name.lower() == name.lower():
                        return p
            return None

        for mod in scenario_plan.get("modifications", []):
            player = find_player(mod["player_name"])
            if player:
                attr = mod["attribute"]
                val = mod["value"]
                
                # Handle Enum conversion for status
                if attr == "status":
                    # Import here to avoid circular dependency issues if placed at top
                    from fantasy_football_simulation import PlayerStatus
                    try:
                        # Map string to enum
                        val = PlayerStatus(val.lower())
                    except ValueError:
                        pass # Keep original if invalid

                setattr(player, attr, val)
                applied_mods.append(f"{player.name}: {attr} -> {val}")

        # 4. Run Simulation
        # We need a new engine for the cloned league
        # Re-init engines with the new league
        # Note: SimulationEngine takes a league in __init__
        
        # We need to create a new ProjectionEngine? No, it's stateless mostly.
        # But SimulationEngine binds to a league.
        
        sim_engine_scenario = SimulationEngine(sim_league, simulation_engine.projection_engine)
        
        # Simulate season or week?
        # Usually scenarios are "What if I win this week?" or "What if X gets injured for the season?"
        # Let's do a season sim to see long term impact
        results = sim_engine_scenario.simulate_season(num_simulations=500)
        
        # Get specific results for the focus team
        team_result = results[results['Team'] == next(t.owner_name for t in league.teams if t.team_id == focus_team_id)]
        
        return {
            "scenario_description": scenario_plan.get("description"),
            "applied_modifications": applied_mods,
            "results_summary": team_result.to_dict('records')[0] if not team_result.empty else {},
            "full_results": results
        }

    def generate_insight(self, scenario_results: Dict[str, Any], baseline_results: Dict[str, Any]) -> str:
        """
        Compare scenario vs baseline and generate a narrative.
        """
        prompt = f"""
        Analyze these simulation results.
        
        Scenario: {scenario_results.get('scenario_description')}
        
        Baseline Results (Current State):
        {baseline_results}
        
        Scenario Results (What If):
        {scenario_results.get('results_summary')}
        
        Modifications Made:
        {scenario_results.get('applied_modifications')}
        
        Write a short, insightful paragraph explaining the impact of this scenario on the team's playoff chances and expected wins.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a fantasy football analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Could not generate insight: {str(e)}"
