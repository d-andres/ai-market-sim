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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from smolagents import Tool, LiteLLMModel, ChatMessage, MessageRole

from src.models.schema import Actor, Map, PlannedAction, TradeProposal
from src.simulation import physics
from src.agents.prompts import TRADE_EVALUATION_PROMPT, get_system_prompt_for_role
from src.agents.response_parser import parse_plan, parse_trade_decision

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



# ---------------------------------------------------------------------------
# Agent brain
# ---------------------------------------------------------------------------

class AgentBrain:
	"""AI brain for an autonomous actor.

	Responsible for:
	- Observing the world (via ObserveSurroundingsTool)
	- Producing a multi-step plan as a list of PlannedAction objects (via create_plan())
	- Evaluating incoming trade proposals from other actors (via evaluate_trade_proposal())

	The engine calls create_plan() once whenever an actor has no outstanding plan
	or was interrupted.  Execution of each step happens entirely inside the engine
	so that no LLM call is required for step-by-step execution.
	"""

	# How many steps the LLM is asked to plan at once.
	PLAN_HORIZON: int = 8

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
		self.system_prompt = get_system_prompt_for_role(actor.role)
		self.obs_tool = ObserveSurroundingsTool(
			actor=actor, world_map=world_map, memory=self.memory
		)

	def create_plan(self, interrupt_reason: str = "") -> tuple[list[PlannedAction], str]:
		"""Call the LLM once to produce a list of planned actions and a thought summary.

		Returns (plan, summary) where summary is a brief in-character description
		of the actor's reasoning that gets written to the event log.
		"""
		observation = self.obs_tool.forward()
		inv_summary = (
			", ".join(f"{i.name} x{i.quantity}" for i in self.actor.inventory)
			or "nothing"
		)

		interrupt_ctx = ""
		if interrupt_reason:
			interrupt_ctx = (
				f"\n\n⚠️ INTERRUPT — your previous plan was cancelled because: {interrupt_reason}\n"
				"Reassess the situation and form a new plan."
			)

		user_msg = (
			f"Tick {self.engine.tick_count} | HP: {self.actor.hp} | "
			f"Gold: {self.actor.gold}g | Carrying: {inv_summary}\n\n"
			f"WORLD OBSERVATION:\n{observation}"
			f"{interrupt_ctx}\n\n"
			f"Create a plan of up to {self.PLAN_HORIZON} steps.\n"
			"Return ONLY a valid JSON object with exactly two keys:\n"
			'  "summary": a single sentence (15-25 words) written in third person describing '
			"your character's thoughts and intended actions in their own voice and personality.\n"
			'  "plan": a JSON array where each element is one of these shapes:\n'
			'    {"action_type": "move", "params": {"direction": "north"}, "reason": ""}\n'
			'    {"action_type": "wait", "params": {}, "reason": ""}\n'
			'    {"action_type": "propose_trade", "params": {"target_actor_id": "actor_id", '
			'"offered_gold": 50, "offered_item_ids": "", "requested_item_ids": "item_crown", '
			'"requested_gold": 0}, "reason": ""}\n'
			"Valid directions: north, south, east, west, northeast, northwest, southeast, southwest.\n"
			"No prose outside the JSON. No markdown. Only the JSON object."
		)

		messages = [
			ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}]),
			ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": user_msg}]),
		]

		_fallback_summary = f"{self.actor.name} pauses, unsure what to do next."
		try:
			response = self.model(messages)
			content: str = response.content if hasattr(response, "content") else str(response)
			return parse_plan(content, self.actor.name)
		except Exception as e:
			return [PlannedAction(action_type="wait", reason=f"Plan generation failed: {e}")], _fallback_summary

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
			response = self.model([ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])])
			raw: str = response.content if hasattr(response, "content") else str(response)
		except Exception as e:
			return False, f"{self.actor.name} couldn't process the offer right now."

		return parse_trade_decision(raw, self.actor.name)


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
