"""Agent brain: LLM-driven decision making using Smolagents.

This module connects actors to their AI "brains" which can:
- Observe the world around them
- Reason about their situation
- Take actions via registered tools

Supports multiple LLM providers via LiteLLM (Ollama, OpenAI, Anthropic, etc.).
See AI-AGENT-INTEGRATION.md for setup instructions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smolagents import Tool, LiteLLMModel, CodeAgent

from src.models.schema import Actor, ActorRole, Map
from src.simulation import physics
from src.agents.prompts import get_system_prompt_for_role

if TYPE_CHECKING:
	from src.simulation.engine import SimulationEngine


class ObserveSurroundingsTool(Tool):
	"""Tool for an actor to observe their surroundings."""
	
	name = "observe_surroundings"
	description = (
		"Observe your surroundings and see what's around you. "
		"Returns information about visible tiles, nearby actors, and points of interest. "
		"Use this to understand your environment before taking action."
	)
	inputs = {
		"vision_range": {
			"type": "integer",
			"description": "How far to look (in tiles). Default is 10.",
			"default": 10,
		}
	}
	output_type = "string"
	
	def __init__(self, actor: Actor, world_map: Map, **kwargs):
		super().__init__(**kwargs)
		self.actor = actor
		self.world_map = world_map
	
	def forward(self, vision_range: int = 10) -> str:
		"""Execute the observation."""
		# Get field of vision
		viewport = physics.get_visible_tiles_and_actors(
			self.world_map, 
			self.actor, 
			vision_range=vision_range
		)
		
		# Build observation report
		report = f"You are at position ({self.actor.x}, {self.actor.y}).\n\n"
		
		# Visible actors
		if viewport.visible_actors:
			report += "Visible actors:\n"
			for other_actor in viewport.visible_actors:
				distance = physics.distance_chebyshev(
					self.actor.x, self.actor.y,
					other_actor.x, other_actor.y
				)
				report += f"- {other_actor.name} ({other_actor.role.value}) at ({other_actor.x}, {other_actor.y}), distance: {distance} tiles\n"
		else:
			report += "No other actors are visible.\n"
		
		report += f"\nYou can see {len(viewport.visible_tiles)} tiles around you.\n"
		
		# Check for shops nearby
		shops_nearby = []
		for x, y in viewport.visible_tiles:
			tile = self.world_map.tile_at(x, y)
			if tile and tile.interactable:
				shops_nearby.append((x, y))
		
		if shops_nearby:
			report += f"\nShops visible: {len(shops_nearby)} shop(s) within sight.\n"
		
		return report


class MoveTool(Tool):
	"""Tool for an actor to move in a direction."""
	
	name = "move"
	description = (
		"Move in a specified direction. Valid directions are: "
		"north, south, east, west, northeast, northwest, southeast, southwest. "
		"Returns success or failure message."
	)
	inputs = {
		"direction": {
			"type": "string",
			"description": "Direction to move (north, south, east, west, northeast, northwest, southeast, southwest)",
		}
	}
	output_type = "string"
	
	def __init__(self, actor: Actor, world_map: Map, engine: SimulationEngine, **kwargs):
		super().__init__(**kwargs)
		self.actor = actor
		self.world_map = world_map
		self.engine = engine
	
	def forward(self, direction: str) -> str:
		"""Execute the move."""
		direction = direction.lower().strip()
		
		if direction not in physics.DIRECTIONS_8:
			valid = ", ".join(physics.DIRECTIONS_8.keys())
			return f"Invalid direction. Valid directions are: {valid}"
		
		dx, dy = physics.DIRECTIONS_8[direction]
		new_x = self.actor.x + dx
		new_y = self.actor.y + dy
		
		# Check if move is valid
		if not physics.can_move_to(self.world_map, new_x, new_y, exclude_actor_id=self.actor.id):
			blocking = physics.get_blocking_actor(self.world_map, new_x, new_y)
			if blocking:
				return f"Cannot move {direction}: blocked by {blocking.name}"
			return f"Cannot move {direction}: tile is not walkable (wall or out of bounds)"
		
		# Execute move
		old_x, old_y = self.actor.x, self.actor.y
		self.actor.x = new_x
		self.actor.y = new_y
		
		# Log event
		self.engine._log_event(
			actor_id=self.actor.id,
			event_type="move",
			description=f"{self.actor.name} moved {direction} from ({old_x},{old_y}) to ({new_x},{new_y})",
			data={"from": (old_x, old_y), "to": (new_x, new_y), "direction": direction}
		)
		
		return f"Successfully moved {direction} to ({new_x}, {new_y})"


class WaitTool(Tool):
	"""Tool for an actor to wait and observe without moving."""
	
	name = "wait"
	description = (
		"Wait for one turn without taking action. "
		"Useful when you want to observe without moving or need to think about your next action."
	)
	inputs = {}
	output_type = "string"
	
	def __init__(self, actor: Actor, engine: SimulationEngine, **kwargs):
		super().__init__(**kwargs)
		self.actor = actor
		self.engine = engine
	
	def forward(self) -> str:
		"""Execute the wait."""
		self.engine._log_event(
			actor_id=self.actor.id,
			event_type="wait",
			description=f"{self.actor.name} is waiting and observing",
			data={}
		)
		return "You wait and observe your surroundings."


class AgentBrain:
	"""AI brain for an autonomous actor using LLM via LiteLLM.
	
	Supports multiple LLM providers: Ollama (local), OpenAI, Anthropic, etc.
	See AI-AGENT-INTEGRATION.md for provider setup instructions.
	"""
	
	def __init__(
		self,
		actor: Actor,
		world_map: Map,
		engine: SimulationEngine,
		model_name: str = "ollama/llama3.2",
		api_base: str = "http://localhost:11434",
	):
		"""Initialize an agent brain.
		
		Args:
		    actor: The Actor this brain controls.
		    world_map: The world Map.
		    engine: The SimulationEngine instance.
		    model_name: LLM model identifier (e.g., 'ollama/llama3.2', 'gpt-4o-mini').
		    api_base: API base URL (required for Ollama, optional for cloud providers).
		"""
		self.actor = actor
		self.world_map = world_map
		self.engine = engine
		
		# Initialize LiteLLM model (auto-detects provider from model_name)
		self.model = LiteLLMModel(
			model_id=model_name,
			api_base=api_base,
		)
		
		# Get role-specific system prompt
		system_prompt = get_system_prompt_for_role(actor.role)
		
		# Initialize tools
		self.tools = [
			ObserveSurroundingsTool(actor=actor, world_map=world_map),
			MoveTool(actor=actor, world_map=world_map, engine=engine),
			WaitTool(actor=actor, engine=engine),
		]
		
		# Create the agent
		self.agent = CodeAgent(
			tools=self.tools,
			model=self.model,
			system_prompt=system_prompt,
			max_steps=3,  # Limit reasoning steps per turn
		)
	
	def take_turn(self) -> str:
		"""Let the agent take one turn (observe, reason, act).
		
		Returns:
		    A summary of what the agent did.
		"""
		try:
			# Prompt the agent to take action
			prompt = (
				f"You are {self.actor.name}, a {self.actor.role.value}. "
				f"It's your turn to act. First observe your surroundings, then decide what to do. "
				f"You have {self.actor.hp} HP and {self.actor.gold} gold."
			)
			
			result = self.agent.run(prompt)
			return f"{self.actor.name}: {result}"
			
		except Exception as e:
			# If agent fails, default to waiting
			self.engine._log_event(
				actor_id=self.actor.id,
				event_type="error",
				description=f"{self.actor.name} encountered an error: {str(e)}",
				data={"error": str(e)}
			)
			return f"{self.actor.name} failed to act: {str(e)}"


def create_agent_for_actor(
	actor: Actor,
	world_map: Map,
	engine: SimulationEngine,
	model_name: str = "ollama/llama3.2",
	api_base: str = "http://localhost:11434",
) -> AgentBrain:
	"""Factory function to create an agent brain for an actor.
	
	Args:
	    actor: The Actor to create a brain for.
	    world_map: The world Map.
	    engine: The SimulationEngine instance.
	    model_name: LLM model identifier (default: local Ollama llama3.2).
	    api_base: API base URL (default: local Ollama server).
	    
	Returns:
	    An AgentBrain instance.
	"""
	return AgentBrain(
		actor=actor,
		world_map=world_map,
		engine=engine,
		model_name=model_name,
		api_base=api_base,
	)
