"""Agent brain: LLM-driven decision making using Smolagents.

This module connects actors to their AI "brains" which can:
- Observe the world around them (including visible inventories)
- Remember past interactions and relationships
- Propose and evaluate trades using LLM judgment
- Move and act in the world

Supports multiple LLM providers via LiteLLM (Ollama, OpenAI, Anthropic, etc.).
See AI-AGENT-INTEGRATION.md for setup instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from smolagents import Tool, LiteLLMModel, ToolCallingAgent

from src.models.schema import Actor, Item, Map, TradeProposal
from src.simulation import physics
from src.agents.prompts import TRADE_EVALUATION_PROMPT, get_system_prompt_for_role

if TYPE_CHECKING:
	from src.simulation.engine import SimulationEngine


# ---------------------------------------------------------------------------
# Relationship memory
# ---------------------------------------------------------------------------

@dataclass
class InteractionRecord:
	"""One remembered interaction between two actors."""
	tick: int
	actor_id: str
	event_type: str   # "trade_accepted", "trade_declined", "spoke", "observed", ...
	notes: str        # Free-text written by the LLM or system


class RelationshipMemory:
	"""Stores an actor's remembered interactions with every other actor.

	Designately open-ended: the LLM appends free-text notes so it can
	remember anything — not just structured outcomes.
	"""

	def __init__(self) -> None:
		# actor_id -> list of interaction records
		self._records: dict[str, list[InteractionRecord]] = {}

	def record(self, tick: int, actor_id: str, event_type: str, notes: str) -> None:
		self._records.setdefault(actor_id, []).append(
			InteractionRecord(tick=tick, actor_id=actor_id, event_type=event_type, notes=notes)
		)

	def history_for(self, actor_id: str, limit: int = 10) -> list[InteractionRecord]:
		return self._records.get(actor_id, [])[-limit:]

	def summary_for(self, actor_id: str, actor_name: str) -> str:
		"""Return a human-readable history string ready for prompt injection."""
		records = self.history_for(actor_id)
		if not records:
			return f"No prior history with {actor_name}."
		lines = [f"[tick {r.tick}] {r.event_type}: {r.notes}" for r in records]
		return "\n".join(lines)



# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ObserveSurroundingsTool(Tool):
	"""Tool for an actor to observe their surroundings, including nearby inventories."""
	
	name = "observe_surroundings"
	description = (
		"Observe your surroundings. Returns your position, visible actors and their "
		"inventories, nearby shops, and a summary of your own gold and inventory. "
		"Use this before deciding whether to propose a trade."
	)
	inputs = {
		"vision_range": {
			"type": "integer",
			"description": "How far to look (in tiles). Default is 10.",
			"default": 10,
			"nullable": True,
		}
	}
	output_type = "string"
	
	def __init__(self, actor: Actor, world_map: Map, memory: RelationshipMemory, **kwargs):
		super().__init__(**kwargs)
		self.actor = actor
		self.world_map = world_map
		self.memory = memory
	
	def forward(self, vision_range: int = 10) -> str:
		viewport = physics.get_visible_tiles_and_actors(
			self.world_map, self.actor, vision_range=vision_range
		)

		lines: list[str] = []
		lines.append(f"You are at ({self.actor.x}, {self.actor.y}).")
		lines.append(f"Your gold: {self.actor.gold}g")

		if self.actor.inventory:
			inv = ", ".join(f"{i.name} x{i.quantity}" for i in self.actor.inventory)
			lines.append(f"Your inventory: {inv}")
		else:
			lines.append("Your inventory: (empty)")

		if viewport.visible_actors:
			lines.append("\nVisible actors:")
			for other in viewport.visible_actors:
				dist = physics.distance_chebyshev(
					self.actor.x, self.actor.y, other.x, other.y
				)
				inv_str = (
					", ".join(f"{i.name} (base {i.base_price}g) x{i.quantity}" for i in other.inventory)
					if other.inventory else "nothing visible"
				)
				lines.append(
					f"  - {other.name} [{other.role.value}] at ({other.x},{other.y}), "
					f"dist {dist}, gold {other.gold}g | carrying: {inv_str}"
				)
				# Surface relationship history so the LLM can use it
				history = self.memory.summary_for(other.id, other.name)
				lines.append(f"    History: {history}")
		else:
			lines.append("\nNo other actors visible.")

		shop_coords = [
			(x, y) for x, y in viewport.visible_tiles
			if self.world_map.tile_at(x, y).interactable
		]
		if shop_coords:
			lines.append(f"\nShop tiles in view: {len(shop_coords)}")

		return "\n".join(lines)


class ProposeTradeTool(Tool):
	"""Tool to propose a trade to a nearby actor.

	The *proposer* specifies what they offer and what they want.
	The target actor's LLM brain then evaluates the deal autonomously
	based on their personality, item values, and relationship history.
	"""

	name = "propose_trade"
	description = (
		"Propose a trade to another actor who is nearby. "
		"Specify the target actor's id, what you are offering (item ids and/or gold), "
		"and what you are requesting from them (item ids and/or gold). "
		"The other actor will decide whether to accept or decline based on their own judgment. "
		"Returns the outcome and what they said."
	)
	inputs = {
		"target_actor_id": {
			"type": "string",
			"description": "The id of the actor you want to trade with.",
		},
		"offered_item_ids": {
			"type": "string",
			"description": (
				"Comma-separated item ids from YOUR inventory that you are offering. "
				"Leave empty or pass empty string if offering only gold."
			),
			"default": "",
			"nullable": True,
		},
		"offered_gold": {
			"type": "integer",
			"description": "Amount of gold you are offering. Use 0 if offering no gold.",
			"default": 0,
			"nullable": True,
		},
		"requested_item_ids": {
			"type": "string",
			"description": (
				"Comma-separated item ids from the TARGET'S inventory that you want. "
				"Leave empty or pass empty string if requesting only gold."
			),
			"default": "",
			"nullable": True,
		},
		"requested_gold": {
			"type": "integer",
			"description": "Amount of gold you are requesting. Use 0 if requesting no gold.",
			"default": 0,
			"nullable": True,
		},
	}
	output_type = "string"

	def __init__(
		self,
		actor: Actor,
		world_map: Map,
		engine: SimulationEngine,
		memory: RelationshipMemory,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.actor = actor
		self.world_map = world_map
		self.engine = engine
		self.memory = memory

	def _resolve_items(
		self, id_string: str, owner: Actor
	) -> tuple[list[Item], str | None]:
		"""Parse a comma-separated item id string against an actor's inventory.

		Returns (matched_items, error_message).  error_message is None on success.
		"""
		if not id_string or not id_string.strip():
			return [], None
		wanted = [s.strip() for s in id_string.split(",") if s.strip()]
		inv_by_id = {item.id: item for item in owner.inventory}
		matched: list[Item] = []
		for iid in wanted:
			if iid not in inv_by_id:
				return [], f"{owner.name} does not have item '{iid}' in their inventory."
			matched.append(inv_by_id[iid])
		return matched, None

	def forward(
		self,
		target_actor_id: str,
		offered_item_ids: str = "",
		offered_gold: int = 0,
		requested_item_ids: str = "",
		requested_gold: int = 0,
	) -> str:
		# --- Validate target exists and is nearby ---
		target = next(
			(a for a in self.world_map.actors if a.id == target_actor_id), None
		)
		if target is None:
			return f"No actor with id '{target_actor_id}' found."

		dist = physics.distance_chebyshev(
			self.actor.x, self.actor.y, target.x, target.y
		)
		if dist > 2:
			return (
				f"{target.name} is too far away ({dist} tiles). "
				"Move closer before proposing a trade."
			)

		# --- Validate gold ---
		if offered_gold > self.actor.gold:
			return (
				f"You only have {self.actor.gold}g but are trying to offer {offered_gold}g."
			)
		if requested_gold > target.gold:
			return (
				f"{target.name} only has {target.gold}g but you are requesting {requested_gold}g."
			)

		# --- Validate items exist in correct inventories ---
		offered_items, err = self._resolve_items(offered_item_ids, self.actor)
		if err:
			return err
		requested_items, err = self._resolve_items(requested_item_ids, target)
		if err:
			return err

		proposal = TradeProposal(
			proposer_id=self.actor.id,
			target_id=target.id,
			offered_items=offered_items,
			offered_gold=offered_gold,
			requested_items=requested_items,
			requested_gold=requested_gold,
		)

		# --- Route to the target's brain for LLM evaluation ---
		outcome = self.engine.evaluate_trade_proposal(
			proposal=proposal,
			proposer=self.actor,
			target=target,
		)
		return outcome


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


# ---------------------------------------------------------------------------
# Agent brain
# ---------------------------------------------------------------------------

class AgentBrain:
	"""AI brain for an autonomous actor.

	Holds:
	- A Smolagents ToolCallingAgent backed by any LiteLLM-compatible model
	- A RelationshipMemory so the LLM can recall past interactions
	- evaluate_trade_proposal(): called by the engine when *this* actor
	  is the target of a trade so the LLM decides accept/decline
	"""

	def __init__(
		self,
		actor: Actor,
		world_map: Map,
		engine: SimulationEngine,
		model_name: str = "ollama/llama3.2",
		api_base: str = "http://localhost:11434",
	):
		self.actor = actor
		self.world_map = world_map
		self.engine = engine
		self.memory = RelationshipMemory()

		self.model = LiteLLMModel(model_id=model_name, api_base=api_base)
		system_prompt = get_system_prompt_for_role(actor.role)

		self.tools = [
			ObserveSurroundingsTool(actor=actor, world_map=world_map, memory=self.memory),
			MoveTool(actor=actor, world_map=world_map, engine=engine),
			WaitTool(actor=actor, engine=engine),
			ProposeTradeTool(actor=actor, world_map=world_map, engine=engine, memory=self.memory),
		]

		self.agent = ToolCallingAgent(
			tools=self.tools,
			model=self.model,
			instructions=system_prompt,
			max_steps=3,
		)

	def take_turn(self) -> str:
		"""Let the agent take one turn (observe, reason, act)."""
		try:
			inv_summary = (
				", ".join(f"{i.name} x{i.quantity}" for i in self.actor.inventory)
				or "nothing"
			)
			prompt = (
				f"You are {self.actor.name}, a {self.actor.role.value}. "
				f"Tick {self.engine.tick_count}. "
				f"HP: {self.actor.hp} | Gold: {self.actor.gold}g | Carrying: {inv_summary}. "
				"Observe your surroundings, then decide what to do. "
				"If a good trade opportunity is nearby, use propose_trade."
			)
			result = self.agent.run(prompt)
			return f"{self.actor.name}: {result}"
		except Exception as e:
			self.engine._log_event(
				actor_id=self.actor.id,
				event_type="error",
				description=f"{self.actor.name} encountered an error: {e}",
				data={"error": str(e)},
			)
			return f"{self.actor.name} failed to act: {e}"

	def evaluate_trade_proposal(
		self,
		proposal: TradeProposal,
		proposer: Actor,
	) -> tuple[bool, str]:
		"""Ask this actor's LLM to evaluate an incoming trade proposal.

		The LLM receives a fully-rendered prompt containing:
		- The proposer's identity and role
		- Relationship history with the proposer
		- A plain-English summary of what is offered vs. requested

		It must respond with two lines:
		  DECISION: ACCEPT / DECLINE
		  RESPONSE: <in-character spoken reply>

		Returns (accepted: bool, spoken_response: str).
		"""
		# Build human-readable offer summaries
		offered_parts: list[str] = []
		if proposal.offered_gold:
			offered_parts.append(f"{proposal.offered_gold} gold")
		for item in proposal.offered_items:
			offered_parts.append(f"{item.name} x{item.quantity} (base value {item.base_price}g each)")
		offered_summary = ", ".join(offered_parts) if offered_parts else "nothing"

		requested_parts: list[str] = []
		if proposal.requested_gold:
			requested_parts.append(f"{proposal.requested_gold} gold")
		for item in proposal.requested_items:
			requested_parts.append(f"{item.name} x{item.quantity} (base value {item.base_price}g each)")
		requested_summary = ", ".join(requested_parts) if requested_parts else "nothing"

		relationship = self.memory.summary_for(proposer.id, proposer.name)

		prompt = TRADE_EVALUATION_PROMPT.format(
			actor_gold=self.actor.gold,
			proposer_name=proposer.name,
			proposer_role=proposer.role.value,
			relationship_history=relationship,
			offered_summary=offered_summary,
			requested_summary=requested_summary,
		)

		try:
			raw: str = self.model(prompt)  # Direct model call, no tool loop needed
		except Exception as e:
			return False, f"{self.actor.name} couldn't process the offer right now."

		# Parse two-line response
		lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
		accepted = False
		spoken = f"{self.actor.name} considers the offer silently."
		for line in lines:
			if line.upper().startswith("DECISION:"):
				accepted = "ACCEPT" in line.upper()
			elif line.upper().startswith("RESPONSE:"):
				spoken = line.split(":", 1)[1].strip()

		return accepted, spoken


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
