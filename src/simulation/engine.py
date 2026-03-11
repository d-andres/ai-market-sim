"""Simulation engine: the "Heartbeat" loop that drives the world forward.

This module implements the core tick-based simulation system that:
- Maintains world state (actors, their positions, inventories)
- Logs events for debugging and replay
- Provides a deterministic, reproducible simulation
- All actor behavior is driven by AI agents (Phase 3+)

Phase 3: AI agents control all actor behavior via LLM reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from src.models.schema import Actor, Map, TradeProposal

if TYPE_CHECKING:
	from src.agents.brain import AgentBrain


@dataclass
class SimulationEvent:
	"""A timestamped event that occurred during simulation."""

	tick: int
	timestamp: str
	actor_id: str
	event_type: str  # "move", "interact", "spawn", etc.
	description: str
	data: dict = field(default_factory=dict)


class SimulationEngine:
	"""Core simulation engine managing world state and tick progression."""

	def __init__(
		self,
		world_map: Map,
		tick_rate: float = 1.0,
		enable_ai: bool = True,
		ollama_model: str = "ollama/llama3.2",
		ollama_base_url: str = "http://localhost:11434",
	):
		"""Initialize the simulation engine.

		Args:
		    world_map: The Map object representing the world.
		    tick_rate: Seconds per tick (default 1.0 = 1 tick per second).
		    enable_ai: Whether to enable AI agents (default True).
		    ollama_model: LLM model identifier (format depends on provider).
		    ollama_base_url: API base URL (for providers that need it, like Ollama).
		"""
		self.world_map = world_map
		self.tick_rate = tick_rate
		self.tick_count = 0
		self.start_time = datetime.now()
		self.events: list[SimulationEvent] = []
		self.enable_ai = enable_ai
		
		# Initialize AI brains for each actor
		self.agent_brains: dict[str, AgentBrain] = {}
		if enable_ai:
			self._initialize_agent_brains(ollama_model, ollama_base_url)
	
	def _initialize_agent_brains(self, model_name: str, api_base: str) -> None:
		"""Initialize AI brains for all actors."""
		from src.agents.brain import create_agent_for_actor
		
		for actor in self.world_map.actors:
			try:
				brain = create_agent_for_actor(
					actor=actor,
					world_map=self.world_map,
					engine=self,
					model_name=model_name,
					api_base=api_base,
				)
				self.agent_brains[actor.id] = brain
				self._log_event(
					actor_id=actor.id,
					event_type="init",
					description=f"{actor.name} brain initialized",
					data={"role": actor.role.value}
				)
			except Exception as e:
				self._log_event(
					actor_id=actor.id,
					event_type="error",
					description=f"Failed to initialize brain for {actor.name}: {str(e)}",
					data={"error": str(e)}
				)

	def _log_event(
		self, actor_id: str, event_type: str, description: str, data: dict | None = None
	) -> None:
		"""Log a simulation event."""
		event = SimulationEvent(
			tick=self.tick_count,
			timestamp=str(datetime.now()),
			actor_id=actor_id,
			event_type=event_type,
			description=description,
			data=data or {},
		)
		self.events.append(event)

	def tick(self) -> dict:
		"""Execute one simulation tick.

		Returns a snapshot of changes during this tick.
		In Phase 3: Each actor's AI brain takes a turn to reason and act.
		"""
		updates = {
			"tick": self.tick_count,
			"events": [],
			"actor_actions": [],
		}
		
		# Give each AI agent a turn
		if self.enable_ai:
			for actor in self.world_map.actors:
				if actor.id in self.agent_brains:
					try:
						action_summary = self.agent_brains[actor.id].take_turn()
						updates["actor_actions"].append({
							"actor_id": actor.id,
							"actor_name": actor.name,
							"summary": action_summary,
						})
					except Exception as e:
						error_msg = f"{actor.name} failed to take turn: {str(e)}"
						self._log_event(
							actor_id=actor.id,
							event_type="error",
							description=error_msg,
							data={"error": str(e)}
						)
						updates["actor_actions"].append({
							"actor_id": actor.id,
							"actor_name": actor.name,
							"summary": error_msg,
						})

		# Collect events from this tick.
		updates["events"] = [
			{
				"tick": e.tick,
				"actor_id": e.actor_id,
				"event_type": e.event_type,
				"description": e.description,
			}
			for e in self.events
			if e.tick == self.tick_count
		]

		self.tick_count += 1
		return updates

	def get_elapsed_time(self) -> float:
		"""Get elapsed simulation time in seconds."""
		return self.tick_count * self.tick_rate

	def get_elapsed_time_formatted(self) -> str:
		"""Get elapsed simulation time as a formatted string."""
		elapsed = self.get_elapsed_time()
		minutes = int(elapsed // 60)
		seconds = int(elapsed % 60)
		return f"{minutes:02d}:{seconds:02d}"

	def get_state_snapshot(self) -> dict:
		"""Get the current simulation state as a snapshot."""
		return {
			"tick": self.tick_count,
			"elapsed_time": self.get_elapsed_time(),
			"elapsed_time_formatted": self.get_elapsed_time_formatted(),
			"actors": [
				{
					"id": actor.id,
					"name": actor.name,
					"role": actor.role.value,
					"x": actor.x,
					"y": actor.y,
					"gold": actor.gold,
					"hp": actor.hp,
				}
				for actor in self.world_map.actors
			],
			"recent_events": [
				{
					"tick": e.tick,
					"actor_id": e.actor_id,
					"event_type": e.event_type,
					"description": e.description,
				}
				for e in self.events[-20:]  # Last 20 events.
			],
		}

	def get_event_log(self, limit: int = 100) -> list[dict]:
		"""Get the event log (up to limit recent events)."""
		return [
			{
				"tick": e.tick,
				"timestamp": e.timestamp,
				"actor_id": e.actor_id,
				"event_type": e.event_type,
				"description": e.description,
			}
			for e in self.events[-limit:]
		]

	def evaluate_trade_proposal(
		self,
		proposal: TradeProposal,
		proposer: Actor,
		target: Actor,
	) -> str:
		"""Route a trade proposal to the target's brain for LLM evaluation,
		then execute the transfer if accepted and record memory on both sides.

		Returns a human-readable outcome string sent back to the proposer's tool.
		"""
		target_brain = self.agent_brains.get(target.id)

		if target_brain is None:
			# No AI brain — target always declines
			return f"{target.name} ignores your offer."

		accepted, spoken = target_brain.evaluate_trade_proposal(
			proposal=proposal, proposer=proposer
		)

		if accepted:
			result = self._resolve_trade(proposal, proposer, target)
			if result is not None:
				# Something became invalid between evaluation and execution
				accepted = False
				spoken = result

		event_type = "trade_accepted" if accepted else "trade_declined"
		description = (
			f"{target.name} {'accepted' if accepted else 'declined'} a trade with "
			f"{proposer.name}. {target.name} said: \"{spoken}\""
		)
		self._log_event(
			actor_id=target.id,
			event_type=event_type,
			description=description,
			data={
				"proposer_id": proposer.id,
				"accepted": accepted,
				"offered_gold": proposal.offered_gold,
				"offered_items": [i.id for i in proposal.offered_items],
				"requested_gold": proposal.requested_gold,
				"requested_items": [i.id for i in proposal.requested_items],
			},
		)

		# Write interaction memories on both brains so future turns recall this
		notes = (
			f"{'Accepted' if accepted else 'Declined'} trade — "
			f"offered {proposal.offered_gold}g + [{', '.join(i.name for i in proposal.offered_items)}], "
			f"requested {proposal.requested_gold}g + [{', '.join(i.name for i in proposal.requested_items)}]. "
			f"Said: \"{spoken}\""
		)
		target_brain.memory.record(
			tick=self.tick_count,
			actor_id=proposer.id,
			event_type=event_type,
			notes=notes,
		)
		proposer_brain = self.agent_brains.get(proposer.id)
		if proposer_brain:
			proposer_brain.memory.record(
				tick=self.tick_count,
				actor_id=target.id,
				event_type=event_type,
				notes=(
					f"{target.name} {'accepted' if accepted else 'declined'} our trade. "
					f"They said: \"{spoken}\""
				),
			)

		if accepted:
			return f"{target.name} accepted the trade. They said: \"{spoken}\""
		return f"{target.name} declined the trade. They said: \"{spoken}\""

	def _resolve_trade(
		self,
		proposal: TradeProposal,
		proposer: Actor,
		target: Actor,
	) -> str | None:
		"""Execute the actual item/gold transfer for an accepted proposal.

		Returns None on success, or an error string if the trade is no longer
		valid (e.g. gold or items changed between evaluation and execution).
		"""
		# Re-validate gold
		if proposal.offered_gold > proposer.gold:
			return f"{proposer.name} no longer has enough gold."
		if proposal.requested_gold > target.gold:
			return f"{target.name} no longer has enough gold."

		# Re-validate items still in inventory
		proposer_inv = {item.id: item for item in proposer.inventory}
		target_inv = {item.id: item for item in target.inventory}
		for item in proposal.offered_items:
			if item.id not in proposer_inv:
				return f"{proposer.name} no longer has {item.name}."
		for item in proposal.requested_items:
			if item.id not in target_inv:
				return f"{target.name} no longer has {item.name}."

		# Transfer gold
		proposer.gold -= proposal.offered_gold
		proposer.gold += proposal.requested_gold
		target.gold += proposal.offered_gold
		target.gold -= proposal.requested_gold

		# Transfer items: offered items go from proposer → target
		for item in proposal.offered_items:
			proposer.inventory = [i for i in proposer.inventory if i.id != item.id]
			target.inventory.append(item)

		# Requested items go from target → proposer
		for item in proposal.requested_items:
			target.inventory = [i for i in target.inventory if i.id != item.id]
			proposer.inventory.append(item)

		return None


# Global simulation instance (will be initialized in main.py).
ENGINE: SimulationEngine | None = None


def initialize_engine(
	world_map: Map,
	tick_rate: float = 1.0,
	enable_ai: bool = True,
	ollama_model: str = "ollama/llama3.2",
	ollama_base_url: str = "http://localhost:11434",
) -> SimulationEngine:
	"""Create and initialize the global simulation engine.
	
	Args:
	    world_map: The Map object to simulate.
	    tick_rate: Seconds per tick.
	    enable_ai: Whether to enable AI agents.
	    ollama_model: LLM model identifier (e.g., 'ollama/llama3.2', 'gpt-4o-mini', 'claude-3-5-sonnet').
	    ollama_base_url: API base URL (required for Ollama, ignored for cloud providers).
	"""
	global ENGINE
	ENGINE = SimulationEngine(
		world_map,
		tick_rate,
		enable_ai=enable_ai,
		ollama_model=ollama_model,
		ollama_base_url=ollama_base_url,
	)
	return ENGINE


def get_engine() -> SimulationEngine:
	"""Get the global simulation engine."""
	if ENGINE is None:
		raise RuntimeError(
			"Engine not initialized. Call initialize_engine() first."
		)
	return ENGINE


__all__ = [
	"SimulationEvent",
	"SimulationEngine",
	"initialize_engine",
	"get_engine",
]
