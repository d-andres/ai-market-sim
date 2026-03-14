"""Simulation engine: the "Heartbeat" loop that drives the world forward.

This module implements the core tick-based simulation system that:
- Maintains world state (actors, their positions, inventories)
- Logs events for debugging and replay
- Provides a deterministic, reproducible simulation
- All actor behavior is driven by AI agents (Phase 3+)

Phase 3: AI agents control all actor behavior via LLM reasoning.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from src.models.schema import Actor, Map, PlannedAction, TradeProposal
from src.simulation import physics

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
		self.llm_call_count: int = 0  # total LLM planning calls made
		self.llm_pending_actors: list[str] = []  # actor names currently waiting on LLM
		
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

		For each actor:
		- If they have no plan or were interrupted → LLM planning calls fire in parallel.
		- Otherwise → pop the next planned action and execute it instantly (no LLM).
		"""
		updates = {
			"tick": self.tick_count,
			"events": [],
			"actor_actions": [],
		}

		if self.enable_ai:
			# Split actors into those needing a new plan vs those executing existing steps
			needs_plan: list = []  # (actor, brain)
			has_steps: list = []   # actor

			for actor in self.world_map.actors:
				brain = self.agent_brains.get(actor.id)
				if not brain:
					continue
				if actor.needs_replan or not actor.action_queue:
					needs_plan.append((actor, brain))
				else:
					has_steps.append(actor)

			# —— Fire all planning calls in parallel ——
			if needs_plan:
				self.llm_call_count += len(needs_plan)
				self.llm_pending_actors = [a.name for a, _ in needs_plan]

				def _plan_actor(actor_brain_pair):
					actor, brain = actor_brain_pair
					return actor, brain.create_plan(interrupt_reason=actor.interrupt_reason)

				with ThreadPoolExecutor(max_workers=len(needs_plan)) as pool:
					futures = {pool.submit(_plan_actor, pair): pair[0] for pair in needs_plan}
					for future in as_completed(futures):
						actor = futures[future]
						try:
							actor, (plan, thought_summary) = future.result()
							actor.action_queue = plan
							actor.needs_replan = False
							actor.interrupt_reason = ""
							self._log_event(
								actor_id=actor.id,
								event_type="plan",
								description=thought_summary,
								data={"steps": len(plan)},
							)
							updates["actor_actions"].append({
								"actor_id": actor.id,
								"actor_name": actor.name,
								"summary": thought_summary,
							})
						except Exception as e:
							self._log_event(
								actor_id=actor.id,
								event_type="error",
								description=f"{actor.name} failed to plan: {e}",
								data={"error": str(e)},
							)
							actor.action_queue = []

				self.llm_pending_actors = []

			# —— Execute one step for actors with existing plans (sequential, mutates state) ——
			for actor in has_steps:
				action = actor.action_queue.pop(0)
				result = self._execute_action(actor, action)
				updates["actor_actions"].append({
					"actor_id": actor.id,
					"actor_name": actor.name,
					"summary": result,
				})

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

	def interrupt_actor(self, actor_id: str, reason: str) -> None:
		"""Mark an actor as needing to replan immediately.

		Called when something unexpected happens (blocked, attacked, etc.).
		Clears the actor's current plan so the LLM is called next tick.
		"""
		actor = next((a for a in self.world_map.actors if a.id == actor_id), None)
		if actor:
			actor.needs_replan = True
			actor.interrupt_reason = reason
			actor.action_queue = []
			self._log_event(
				actor_id=actor_id,
				event_type="interrupt",
				description=f"{actor.name} interrupted: {reason}",
				data={"reason": reason},
			)

	def _execute_action(self, actor: Actor, action: PlannedAction) -> str:
		"""Execute one planned action step. Returns a description of what happened.

		If the action cannot be completed (e.g. path blocked), interrupt_actor()
		is called so the LLM replans on the next tick.
		"""
		if action.action_type == "move":
			direction = action.params.get("direction", "").lower().strip()
			if direction not in physics.DIRECTIONS_8:
				reason = f"Invalid direction in plan: '{direction}'"
				self.interrupt_actor(actor.id, reason)
				return reason
			dx, dy = physics.DIRECTIONS_8[direction]
			new_x, new_y = actor.x + dx, actor.y + dy
			if not physics.can_move_to(self.world_map, new_x, new_y, exclude_actor_id=actor.id):
				blocking = physics.get_blocking_actor(self.world_map, new_x, new_y)
				reason = (
					f"Path blocked by {blocking.name} at ({new_x},{new_y})"
					if blocking else
					f"Tile ({new_x},{new_y}) is not walkable"
				)
				self.interrupt_actor(actor.id, reason)
				# Also interrupt the blocker so they react
				if blocking:
					self.interrupt_actor(blocking.id, f"{actor.name} tried to move into your tile")
				return reason
			old_x, old_y = actor.x, actor.y
			actor.x, actor.y = new_x, new_y
			desc = f"{actor.name} moved {direction} from ({old_x},{old_y}) to ({new_x},{new_y})"
			self._log_event(actor.id, "move", desc, {"from": [old_x, old_y], "to": [new_x, new_y]})
			return desc

		elif action.action_type == "wait":
			desc = f"{actor.name} is waiting and observing"
			self._log_event(actor.id, "wait", desc)
			return desc

		elif action.action_type == "propose_trade":
			return self._execute_trade_action(actor, action.params)

		else:
			return f"Unknown action type: '{action.action_type}'"

	def _execute_trade_action(self, actor: Actor, params: dict) -> str:
		"""Execute a propose_trade action step."""
		target_id = params.get("target_actor_id", "")
		target = next((a for a in self.world_map.actors if a.id == target_id), None)
		if target is None:
			reason = f"Trade target '{target_id}' not found"
			self.interrupt_actor(actor.id, reason)
			return reason

		dist = physics.distance_chebyshev(actor.x, actor.y, target.x, target.y)
		if dist > 2:
			reason = f"{target.name} is {dist} tiles away — need to move closer first"
			self.interrupt_actor(actor.id, reason)
			return reason

		offered_gold = int(params.get("offered_gold", 0))
		requested_gold = int(params.get("requested_gold", 0))

		def _resolve_items(id_str: str, owner: Actor) -> tuple[list, str | None]:
			if not id_str or not id_str.strip():
				return [], None
			inv = {i.id: i for i in owner.inventory}
			matched = []
			for iid in [s.strip() for s in id_str.split(",") if s.strip()]:
				if iid not in inv:
					return [], f"{owner.name} does not have item '{iid}'"
				matched.append(inv[iid])
			return matched, None

		offered_items, err = _resolve_items(params.get("offered_item_ids", ""), actor)
		if err:
			self.interrupt_actor(actor.id, err)
			return err
		requested_items, err = _resolve_items(params.get("requested_item_ids", ""), target)
		if err:
			self.interrupt_actor(actor.id, err)
			return err

		proposal = TradeProposal(
			proposer_id=actor.id,
			target_id=target.id,
			offered_items=offered_items,
			offered_gold=offered_gold,
			requested_items=requested_items,
			requested_gold=requested_gold,
		)
		return self.evaluate_trade_proposal(proposal=proposal, proposer=actor, target=target)

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
