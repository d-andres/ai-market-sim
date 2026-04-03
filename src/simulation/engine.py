"""Simulation engine: the "Heartbeat" loop that drives the world forward.

This module implements the core tick-based simulation system that:
- Maintains world state (actors, their positions, inventories)
- Logs events for debugging and replay
- Provides a deterministic, reproducible simulation
- All actor behavior is driven by AI agents (Phase 3+)

Phase 3: AI agents control all actor behavior via LLM reasoning.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import TYPE_CHECKING

from src.models.schema import (
	Actor,
	ActorRole,
	AttackParams,
	ConverseParams,
	ConversationPacket,
	EquipParams,
	UnequipParams,
	UseItemParams,
	WorldItem,
	Item,
	Map,
	MoveParams,
	PickUpParams,
	PlaceParams,
	PlannedAction,
	ProposeTradeParams,
	TradeProposal,
)
from src.simulation import physics

if TYPE_CHECKING:
	from src.agents.brain import AgentBrain


# ---------------------------------------------------------------------------
# Planning cadence and load-shedding
# ---------------------------------------------------------------------------

# Minimum ticks after a plan completes before an actor re-queues for exploration.
# Only critical-interrupt replans (interrupt_reason set) bypass this.
# At the default 2s/tick this equals 46 seconds.
REPLAN_CADENCE_TICKS: int = 23

# Minimum ticks to wait before retrying after a planning failure (≈ 30 seconds).
PLAN_FAILURE_COOLDOWN_TICKS: int = 15

# Hard cap on simultaneous in-flight LLM planning requests.
MAX_CONCURRENT_PLAN_JOBS: int = 10

# Social/crime mechanics
WITNESS_VISION_RANGE: int = 10
SELF_DEFENSE_WINDOW_TICKS: int = 12

# Dialogue-driven social inference
POSITIVE_DIALOGUE_WORDS = (
	"thanks", "thank", "please", "sorry", "friend", "help", "appreciate", "grateful"
)
NEGATIVE_DIALOGUE_WORDS = (
	"idiot", "stupid", "hate", "liar", "scum", "trash", "fool"
)
THREAT_DIALOGUE_WORDS = (
	"kill", "hurt", "attack", "destroy", "die"
)
ACCUSATION_DIALOGUE_WORDS = (
	"stole", "steal", "thief", "murder", "murdered", "crime", "criminal", "attacked", "assault"
)


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
		self._plan_executor = ThreadPoolExecutor(max_workers=max(1, len(self.world_map.actors)))
		self._pending_plan_futures: dict[str, Future] = {}
		self._pending_plan_started_tick: dict[str, int] = {}
		self._plan_generation: dict[str, int] = {a.id: 0 for a in self.world_map.actors}
		self._pending_plan_generation: dict[str, int] = {}
		self.stale_plan_discard_count: int = 0
		self.applied_plan_count: int = 0
		self._plan_last_completed_tick: dict[str, int] = {}
		self._plan_failure_cooldown_until: dict[str, int] = {}
		self._conversation_packets: dict[str, ConversationPacket] = {}
		self._last_reflex_event: dict[str, tuple[str, int]] = {}
		# Separate executor for conversation packet generation so it never blocks tick().
		self._conversation_executor = ThreadPoolExecutor(max_workers=2)
		# Keyed by listener (target) actor id → in-flight Future.
		self._pending_packet_futures: dict[str, Future] = {}
		# Combat and social tracking for crime/self-defense logic.
		self._last_attack_tick: dict[tuple[str, str], int] = {}  # (attacker_id, victim_id) -> tick
		self._suspicion: dict[tuple[str, str], int] = {}  # (accuser_id, suspect_id) -> score

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
				self._plan_generation.setdefault(actor.id, 0)
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

	def _actor_by_id(self, actor_id: str) -> Actor | None:
		"""Find an actor by id from the current world state."""
		return next((a for a in self.world_map.actors if a.id == actor_id), None)

	def _adjust_likeness(self, observer: Actor, subject_id: str, delta: int) -> None:
		"""Adjust observer's affinity/trust toward another actor, clamped to [-100, 100]."""
		current = observer.likeness.get(subject_id, 0)
		observer.likeness[subject_id] = max(-100, min(100, current + delta))

	def suspicion_score(self, observer_id: str, suspect_id: str) -> int:
		"""Get observer's current suspicion score about a suspect."""
		return self._suspicion.get((observer_id, suspect_id), 0)

	def _increase_suspicion(self, observer_id: str, suspect_id: str, delta: int) -> int:
		"""Increase suspicion score and return the updated value."""
		key = (observer_id, suspect_id)
		self._suspicion[key] = self._suspicion.get(key, 0) + max(0, delta)
		return self._suspicion[key]

	def _adjust_suspicion(self, observer_id: str, suspect_id: str, delta: int) -> int:
		"""Adjust suspicion score (positive or negative), clamped at 0 lower bound."""
		key = (observer_id, suspect_id)
		cur = self._suspicion.get(key, 0)
		next_val = max(0, cur + delta)
		self._suspicion[key] = next_val
		return next_val

	def _recent_events_for_dialogue(self, actor_ids: set[str], limit: int = 8) -> list[str]:
		"""Recent event snippets involving relevant actors, for dialogue credibility context."""
		out: list[str] = []
		for e in reversed(self.events):
			if e.actor_id in actor_ids or e.data.get("target_id") in actor_ids or e.data.get("offender_id") in actor_ids:
				out.append(f"[tick {e.tick}] {e.event_type}: {e.description}")
			if len(out) >= limit:
				break
		out.reverse()
		return out

	def _mentioned_actors(self, text: str, exclude_ids: set[str] | None = None) -> list[Actor]:
		"""Find actor mentions by id or whole-word name tokens in free-form dialogue."""
		exclude = exclude_ids or set()
		lower = text.lower()
		mentioned: list[Actor] = []

		for candidate in self.world_map.actors:
			if candidate.id in exclude:
				continue
			if candidate.id.lower() in lower:
				mentioned.append(candidate)
				continue
			name_hit = False
			for token in candidate.name.lower().split():
				if len(token) < 4:
					continue
				if re.search(rf"\b{re.escape(token)}\b", lower):
					name_hit = True
					break
			if name_hit:
				mentioned.append(candidate)

		return mentioned

	def _apply_dialogue_social_effects(self, speaker: Actor, listener: Actor, line: str) -> None:
		"""Adjust likeness/suspicion using simple lexical cues from free-form dialogue.

		This intentionally lightweight parser gives LLM-generated conversation immediate
		social consequences without needing a separate structured action.
		"""
		text = (line or "").lower().strip()
		if not text:
			return

		candidate_actors = [a for a in self.world_map.actors if a.id != listener.id]
		susp_snapshot = {a.id: self.suspicion_score(listener.id, a.id) for a in candidate_actors}
		recent_events = self._recent_events_for_dialogue({speaker.id, listener.id, *[a.id for a in candidate_actors]})

		impact: dict | None = None
		listener_brain = self.agent_brains.get(listener.id)
		if listener_brain is not None:
			impact = listener_brain.assess_dialogue_social_impact(
				speaker=speaker,
				line=line,
				candidate_actors=candidate_actors,
				suspicion_snapshot=susp_snapshot,
				recent_events=recent_events,
			)

		if impact is not None:
			likeness_delta = int(impact.get("likeness_delta", 0))
			if likeness_delta:
				self._adjust_likeness(listener, speaker.id, likeness_delta)

			action_bias = str(impact.get("action_bias", "watch"))
			confidence = float(impact.get("confidence", 0.0))
			rationale = str(impact.get("rationale", ""))
			trust_to_speaker = listener.likeness.get(speaker.id, 0)

			for suspect_id, raw_delta in (impact.get("suspicion_deltas", {}) or {}).items():
				suspect = self._actor_by_id(suspect_id)
				if suspect is None or suspect.id == listener.id:
					continue

				# Credibility gate: accusations from neutral/low-trust speakers shouldn't be
				# strongly believed unless trust is earned.
				delta = int(raw_delta)
				if delta > 0:
					if trust_to_speaker < 20:
						delta = 0
					else:
						trust_factor = min(1.0, max(0.0, (trust_to_speaker - 20) / 60.0))
						delta = max(1, int(round(delta * trust_factor)))
					if confidence < 0.35:
						delta = max(0, delta - 1)

				updated = self._adjust_suspicion(listener.id, suspect.id, delta)
				if delta != 0:
					self._log_event(
						listener.id,
						"dialogue_suspicion",
						f"{listener.name} updates suspicion of {suspect.name} after hearing {speaker.name}.",
						{
							"speaker_id": speaker.id,
							"suspect_id": suspect.id,
							"delta": delta,
							"suspicion": updated,
							"trust_to_speaker": trust_to_speaker,
							"confidence": confidence,
							"action_bias": action_bias,
							"rationale": rationale,
						},
					)

				if listener.role == ActorRole.GUARD and action_bias in {"question", "escalate"} and updated >= 8:
					severity = min(12, max(3, updated // 2))
					self._guard_confront_offender(listener, suspect, severity, "reported_crime")
			return

		# Heuristic fallback path (AI unavailable): still allows social updates in
		# no-LLM tests or degraded runtime.
		pos_hits = sum(1 for w in POSITIVE_DIALOGUE_WORDS if w in text)
		neg_hits = sum(1 for w in NEGATIVE_DIALOGUE_WORDS if w in text)
		threat_hits = sum(1 for w in THREAT_DIALOGUE_WORDS if w in text)
		accusation_hits = sum(1 for w in ACCUSATION_DIALOGUE_WORDS if w in text)

		if pos_hits:
			self._adjust_likeness(listener, speaker.id, pos_hits)
		if neg_hits:
			self._adjust_likeness(listener, speaker.id, -2 * neg_hits)
		if threat_hits:
			self._adjust_likeness(listener, speaker.id, -3 * threat_hits)

		if accusation_hits > 0:
			trust_to_speaker = listener.likeness.get(speaker.id, 0)
			mentioned = self._mentioned_actors(text, exclude_ids={speaker.id, listener.id})
			for suspect in mentioned:
				if trust_to_speaker < 20:
					delta = 0
				else:
					delta = max(1, (1 + accusation_hits) // 2)
				updated = self._adjust_suspicion(listener.id, suspect.id, delta)
				if delta != 0:
					self._log_event(
						listener.id,
						"dialogue_suspicion",
						f"{listener.name} grows suspicious of {suspect.name} after hearing {speaker.name}.",
						{
							"speaker_id": speaker.id,
							"suspect_id": suspect.id,
							"delta": delta,
							"suspicion": updated,
							"trust_to_speaker": trust_to_speaker,
						},
					)
					if listener.role == ActorRole.GUARD and updated >= 8:
						severity = min(12, max(3, updated // 2))
						self._guard_confront_offender(listener, suspect, severity, "reported_crime")

	def _witnesses_at(
		self,
		x: int,
		y: int,
		exclude_ids: set[str] | None = None,
	) -> list[Actor]:
		"""Actors who can see a location (line-of-sight + range)."""
		exclude = exclude_ids or set()
		witnesses: list[Actor] = []
		for observer in self.world_map.actors:
			if observer.id in exclude:
				continue
			dist = physics.distance_chebyshev(observer.x, observer.y, x, y)
			if dist > WITNESS_VISION_RANGE:
				continue
			if physics.can_see(self.world_map, observer.x, observer.y, x, y):
				witnesses.append(observer)
		return witnesses

	def _theft_severity(self, item: Item) -> int:
		"""Translate stolen item value to crime severity (1..25)."""
		return max(1, min(25, (item.base_price // 20) + 1))

	def _assault_severity(self, damage: int) -> int:
		"""Translate attack damage into severity for witnessed assault."""
		return max(2, min(20, damage // 2 + 2))

	def _is_self_defense(self, attacker: Actor, target: Actor) -> bool:
		"""Return True when target recently attacked attacker within defense window."""
		last = self._last_attack_tick.get((target.id, attacker.id))
		if last is None:
			return False
		return (self.tick_count - last) <= SELF_DEFENSE_WINDOW_TICKS

	def _closest_guard(self, x: int, y: int) -> Actor | None:
		guards = [a for a in self.world_map.actors if a.role == ActorRole.GUARD and a.hp > 0]
		if not guards:
			return None
		return min(guards, key=lambda g: physics.distance_chebyshev(g.x, g.y, x, y))

	def _guard_confront_offender(
		self,
		guard: Actor,
		offender: Actor,
		severity: int,
		crime_type: str,
		victim: Actor | None = None,
	) -> None:
		"""Immediate guard response based on offender infamy and crime severity."""
		if guard.id == offender.id:
			return
		if offender not in self.world_map.actors or guard not in self.world_map.actors:
			return

		score = offender.infamy + (severity * 2)
		if score < 20:
			desc = (
				f"{guard.name} confronts {offender.name} for {crime_type} but issues only a warning."
			)
			self._log_event(guard.id, "guard_warning", desc, {"offender_id": offender.id, "crime": crime_type})
			self.interrupt_actor(offender.id, f"{guard.name} warned you: {crime_type} is criminal.")
			self._adjust_likeness(guard, offender.id, -max(1, severity // 2))
			return

		if score < 45:
			fine = min(offender.gold, max(5, severity * 3))
			offender.gold -= fine
			guard.gold += fine
			desc = f"{guard.name} fines {offender.name} {fine}g for {crime_type}."
			self._log_event(
				guard.id,
				"guard_fine",
				desc,
				{"offender_id": offender.id, "crime": crime_type, "fine": fine},
			)
			offender.infamy = max(0, offender.infamy - max(1, severity // 3))
			self._adjust_likeness(guard, offender.id, -severity)
			return

		if score < 80:
			damage = max(5, severity * 2)
			if offender.hp > 1:
				damage = min(damage, offender.hp - 1)
			offender.hp = max(0, offender.hp - damage)
			desc = f"{guard.name} beats {offender.name} for {damage} damage as punishment for {crime_type}."
			self._log_event(
				guard.id,
				"guard_assault",
				desc,
				{"offender_id": offender.id, "crime": crime_type, "damage": damage, "hp": offender.hp},
			)
			self._log_event(
				guard.id,
				"combat",
				desc,
				{"target_id": offender.id, "damage": damage, "hp": offender.hp, "source": "guard_assault"},
			)
			offender.infamy = max(0, offender.infamy - max(1, severity // 2))
			self._adjust_likeness(guard, offender.id, -severity * 2)
			return

		# Very high notoriety/severity => lethal force.
		damage, is_crit = physics.calc_damage(guard, offender)
		offender.hp = max(0, offender.hp - damage)
		desc = (
			f"{guard.name} attempts lethal force on {offender.name} for {crime_type} "
			f"({damage} damage{' CRIT' if is_crit else ''})."
		)
		self._log_event(
			guard.id,
			"guard_lethal",
			desc,
			{"offender_id": offender.id, "crime": crime_type, "damage": damage, "hp": offender.hp},
		)
		self._log_event(
			guard.id,
			"combat",
			desc,
			{"target_id": offender.id, "damage": damage, "hp": offender.hp, "source": "guard_lethal"},
		)
		self._adjust_likeness(guard, offender.id, -severity * 3)
		if offender.hp == 0:
			self._handle_actor_death(offender, killer=guard)

	def _register_crime(
		self,
		offender: Actor,
		crime_type: str,
		severity: int,
		x: int,
		y: int,
		victim: Actor | None = None,
		details: dict | None = None,
	) -> None:
		"""Record a crime, update social state, and trigger witness/guard reactions."""
		witnesses = self._witnesses_at(x, y, exclude_ids={offender.id})
		witness_ids = [w.id for w in witnesses]

		if witnesses:
			offender.infamy += severity
			offender.fame = max(0, offender.fame - max(0, severity // 3))

		for w in witnesses:
			self._adjust_likeness(w, offender.id, -severity)
			brain = self.agent_brains.get(w.id)
			if brain:
				brain.memory.record(
					tick=self.tick_count,
					actor_id=offender.id,
					event_type=f"witnessed_{crime_type}",
					notes=(
						f"I witnessed {offender.name} commit {crime_type} at ({x},{y}) "
						f"(severity {severity})."
					),
				)
			if w.role == ActorRole.GUARD:
				self._guard_confront_offender(w, offender, severity, crime_type, victim=victim)

		# Optional delayed-report flavor for shop thefts not directly witnessed.
		if crime_type == "theft" and victim is not None and victim.id not in witness_ids:
			self._suspicion[(victim.id, offender.id)] = self._suspicion.get((victim.id, offender.id), 0) + severity
			guard = self._closest_guard(victim.x, victim.y)
			if guard is not None:
				self._log_event(
					victim.id,
					"report_theft",
					f"{victim.name} suspects {offender.name} and reports theft to {guard.name}.",
					{"suspect_id": offender.id, "severity": severity},
				)
				# Guard only punishes immediately if suspect still carries the stolen item.
				stolen_item_id = (details or {}).get("item_id")
				if stolen_item_id and any(i.id == stolen_item_id for i in offender.inventory):
					self._guard_confront_offender(guard, offender, severity, "suspected_theft", victim=victim)
				else:
					self._adjust_likeness(victim, offender.id, -max(1, severity // 2))

		payload = {
			"crime_type": crime_type,
			"severity": severity,
			"offender_id": offender.id,
			"victim_id": victim.id if victim else None,
			"witnesses": witness_ids,
		}
		if details:
			payload.update(details)

		self._log_event(
			offender.id,
			"crime",
			f"{offender.name} committed {crime_type} (severity {severity}). "
			+ (f"Witnessed by {len(witnesses)} actor(s)." if witnesses else "No direct witnesses."),
			payload,
		)

	def _submit_plan_request(self, actor: Actor, brain: AgentBrain) -> None:
		"""Submit a non-blocking LLM planning request for one actor."""
		if actor.id in self._pending_plan_futures:
			return
		interrupt_reason = actor.interrupt_reason
		generation = self._plan_generation.get(actor.id, 0)
		future = self._plan_executor.submit(brain.create_plan, interrupt_reason)
		self._pending_plan_futures[actor.id] = future
		self._pending_plan_started_tick[actor.id] = self.tick_count
		self._pending_plan_generation[actor.id] = generation
		self.llm_call_count += 1
		self._log_event(
			actor_id=actor.id,
			event_type="plan_submitted",
			description=f"{actor.name} requested a new plan",
			data={"interrupt_reason": interrupt_reason, "generation": generation},
		)

	def _collect_completed_plan_requests(self, updates: dict) -> None:
		"""Apply completed planning jobs without blocking the current tick."""
		for actor_id, future in list(self._pending_plan_futures.items()):
			if not future.done():
				continue

			actor = self._actor_by_id(actor_id)
			if actor is None:
				self._pending_plan_futures.pop(actor_id, None)
				self._pending_plan_started_tick.pop(actor_id, None)
				self._pending_plan_generation.pop(actor_id, None)
				continue

			submitted_generation = self._pending_plan_generation.get(actor_id, 0)
			current_generation = self._plan_generation.get(actor_id, 0)
			if submitted_generation != current_generation:
				# Actor state changed while planning; discard stale result safely.
				self.stale_plan_discard_count += 1
				self._log_event(
					actor_id=actor.id,
					event_type="plan_discarded_stale",
					description=f"Discarded stale plan for {actor.name}",
					data={
						"submitted_generation": submitted_generation,
						"current_generation": current_generation,
					},
				)
				self._pending_plan_futures.pop(actor_id, None)
				self._pending_plan_started_tick.pop(actor_id, None)
				self._pending_plan_generation.pop(actor_id, None)
				continue

			try:
				plan, thought_summary = future.result()
				actor.action_queue = plan
				actor.needs_replan = False
				actor.interrupt_reason = ""
				self.applied_plan_count += 1
				self._plan_last_completed_tick[actor_id] = self.tick_count
				self._log_event(
					actor_id=actor.id,
					event_type="plan",
					description=thought_summary,
					data={"steps": len(plan), "generation": current_generation},
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
				actor.needs_replan = True
				self._plan_failure_cooldown_until[actor_id] = self.tick_count + PLAN_FAILURE_COOLDOWN_TICKS

			self._pending_plan_futures.pop(actor_id, None)
			self._pending_plan_started_tick.pop(actor_id, None)
			self._pending_plan_generation.pop(actor_id, None)

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
			# Apply any plan jobs that completed since previous tick(s), without blocking.
			self._collect_completed_plan_requests(updates)

			# Submit new plan requests in priority order, respecting cadence/cooldown/cap.
			# Priority 0 = critical interrupt (needs_replan + interrupt_reason) → always immediate
			# Priority 1 = voluntary replan or soft replan  → subject to REPLAN_CADENCE_TICKS
			# Priority 2 = queue exhausted                  → subject to REPLAN_CADENCE_TICKS
			candidates = [
				a for a in self.world_map.actors
				if (a.needs_replan or not a.action_queue)
				and a.id not in self._pending_plan_futures
				and self.agent_brains.get(a.id)
				and self._plan_failure_cooldown_until.get(a.id, 0) <= self.tick_count
			]
			candidates.sort(key=lambda a: (
				0 if (a.needs_replan and a.interrupt_reason) else
				1 if a.needs_replan else
				2
			))
			for actor in candidates:
				if len(self._pending_plan_futures) >= MAX_CONCURRENT_PLAN_JOBS:
					break
				# Critical interrupts (interrupt_reason set) bypass cadence; everything else waits.
				is_critical = actor.needs_replan and bool(actor.interrupt_reason)
				if not is_critical:
					last = self._plan_last_completed_tick.get(actor.id, 0)
					if self.tick_count - last < REPLAN_CADENCE_TICKS:
						continue
				self._submit_plan_request(actor, self.agent_brains[actor.id])

			# Keep UI status in sync with currently pending planning jobs.
			self.llm_pending_actors = [
				a.name
				for a in self.world_map.actors
				if a.id in self._pending_plan_futures
			]

			# Execute one step for actors with existing plans (sequential, mutates state).
			for actor in self.world_map.actors:
				if not actor.action_queue:
					# Run a local reflex action while the actor waits on a pending LLM plan.
					if actor.id in self._pending_plan_futures:
						result = self._reflex_action(actor)
						updates["actor_actions"].append({
							"actor_id": actor.id,
							"actor_name": actor.name,
							"summary": result,
						})
					continue
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

	def _effective_inventory(self, actor: Actor) -> list[Item]:
		"""Carried items plus any shelf items owned by this actor."""
		shelf = [gi.item for gi in self.world_map.world_items if gi.owner_id == actor.id]
		return list(actor.inventory) + shelf

	def interrupt_actor(self, actor_id: str, reason: str) -> None:
		"""Mark an actor as needing to replan immediately.

		Called when something unexpected happens (blocked, attacked, etc.).
		Clears the actor's current plan so the LLM is called next tick.
		"""
		actor = next((a for a in self.world_map.actors if a.id == actor_id), None)
		if actor:
			self._plan_generation[actor_id] = self._plan_generation.get(actor_id, 0) + 1
			pending = self._pending_plan_futures.pop(actor_id, None)
			if pending is not None and not pending.done():
				pending.cancel()
			self._pending_plan_started_tick.pop(actor_id, None)
			self._pending_plan_generation.pop(actor_id, None)
			actor.needs_replan = True
			actor.interrupt_reason = reason
			actor.action_queue = []
			self._log_event(
				actor_id=actor_id,
				event_type="interrupt",
				description=f"{actor.name} interrupted: {reason}",
				data={"reason": reason},
			)

	def _reflex_action(self, actor: Actor) -> str:
		"""Execute a local fallback action when an actor has no queued plan but is waiting on LLM.

		Keeps actors visibly active every tick without requiring LLM input.
		Priority: trade-range adjacency > conversation-range awareness > idle observe.
		"""
		from src.agents.prompts import CONVERSATION_RANGE

		def _log_reflex_once(description: str, data: dict) -> str:
			"""Suppress identical reflex logs for a few ticks to avoid event spam."""
			last = self._last_reflex_event.get(actor.id)
			if last is not None:
				last_desc, last_tick = last
				if last_desc == description and (self.tick_count - last_tick) < 5:
					return description
			self._log_event(actor.id, "reflex_action", description, data)
			self._last_reflex_event[actor.id] = (description, self.tick_count)
			return description

		# Prefer trade-range adjacency (≤2 tiles) — actor is in position and holding
		for other in self.world_map.actors:
			if other.id == actor.id:
				continue
			dist = physics.distance_chebyshev(actor.x, actor.y, other.x, other.y)
			if dist <= 2:
				desc = f"{actor.name} stands ready near {other.name}, waiting for a plan"
				return _log_reflex_once(
					desc,
					{"nearby_actor": other.id, "dist": dist, "context": "trade_range"},
				)

		# Acknowledge nearby actors within conversation range (≤8 tiles)
		for other in self.world_map.actors:
			if other.id == actor.id:
				continue
			dist = physics.distance_chebyshev(actor.x, actor.y, other.x, other.y)
			if dist <= CONVERSATION_RANGE:
				desc = f"{actor.name} watches {other.name} nearby while gathering thoughts"
				return _log_reflex_once(
					desc,
					{"nearby_actor": other.id, "dist": dist, "context": "conversation_range"},
				)

		# Default: hold position and observe surroundings
		desc = f"{actor.name} pauses and observes their surroundings"
		return _log_reflex_once(desc, {"context": "idle"})

	def _execute_action(self, actor: Actor, action: PlannedAction) -> str:
		"""Execute one planned action step. Returns a description of what happened.

		If the action cannot be completed (e.g. path blocked), interrupt_actor()
		is called so the LLM replans on the next tick.
		"""
		if action.action_type == "move":
			try:
				p = MoveParams.model_validate(action.params)
			except Exception as e:
				reason = f"Invalid move params: {e}"
				self.interrupt_actor(actor.id, reason)
				return reason
			direction = p.direction.lower().strip()
			if direction not in physics.DIRECTIONS_8:
				reason = f"Invalid direction in plan: '{direction}'"
				self.interrupt_actor(actor.id, reason)
				return reason
			dx, dy = physics.DIRECTIONS_8[direction]
			new_x, new_y = actor.x + dx, actor.y + dy
			if not physics.can_move_to(self.world_map, new_x, new_y, exclude_actor_id=actor.id):
				blocking = physics.get_blocking_actor(self.world_map, new_x, new_y)
				if blocking:
					# Already adjacent — skip the move step, don't wipe plans
					current_dist = physics.distance_chebyshev(actor.x, actor.y, blocking.x, blocking.y)
					if current_dist <= 2:
						desc = f"{actor.name} is adjacent to {blocking.name} and ready to trade"
						self._log_event(actor.id, "wait", desc)
						return desc
					# Far away — real block, interrupt both
					reason = f"Path blocked by {blocking.name} at ({new_x},{new_y})"
					self.interrupt_actor(actor.id, reason)
					self.interrupt_actor(blocking.id, f"{actor.name} tried to move into your tile")
					return reason
				reason = f"Tile ({new_x},{new_y}) is not walkable"
				self.interrupt_actor(actor.id, reason)
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

		elif action.action_type == "pick_up":
			return self._execute_pickup_action(actor, action)

		elif action.action_type == "place":
			return self._execute_place_action(actor, action)

		elif action.action_type == "converse":
			return self._execute_converse_action(actor, action)

		elif action.action_type == "attack":
			return self._execute_attack_action(actor, action)

		elif action.action_type == "equip":
			return self._execute_equip_action(actor, action)

		elif action.action_type == "unequip":
			return self._execute_unequip_action(actor, action)

		elif action.action_type == "use_item":
			return self._execute_use_item_action(actor, action)

		else:
			return f"Unknown action type: '{action.action_type}'"

	def _execute_pickup_action(self, actor: Actor, action: PlannedAction) -> str:
		"""Pick up an unowned item from the current or an adjacent tile."""
		try:
			p = PickUpParams.model_validate(action.params)
		except Exception as e:
			reason = f"Invalid pick_up params: {e}"
			self.interrupt_actor(actor.id, reason)
			return reason

		# Search for the item within adjacency range (Chebyshev ≤ 1)
		target_gi: WorldItem | None = None
		for gi in self.world_map.world_items:
			if gi.item.id == p.item_id:
				if physics.distance_chebyshev(actor.x, actor.y, gi.x, gi.y) <= 1:
					target_gi = gi
					break

		if target_gi is None:
			reason = f"No item '{p.item_id}' within reach to pick up"
			self.interrupt_actor(actor.id, reason)
			return reason

		owner = None
		is_theft = target_gi.owner_id is not None and target_gi.owner_id != actor.id
		if is_theft:
			owner = self._actor_by_id(target_gi.owner_id)

		# Carry-capacity check
		if self._carry_count(actor) >= actor.max_carry:
			reason = (
				f"{actor.name} is carrying too many items ({actor.max_carry}/{actor.max_carry}) "
				"— drop or place something first"
			)
			self.interrupt_actor(actor.id, reason)
			return reason

		# Transfer: remove from ground, add to inventory
		self.world_map.world_items = [
			gi for gi in self.world_map.world_items if gi is not target_gi
		]
		actor.inventory.append(target_gi.item)

		if is_theft:
			severity = self._theft_severity(target_gi.item)
			self._register_crime(
				offender=actor,
				crime_type="theft",
				severity=severity,
				x=target_gi.x,
				y=target_gi.y,
				victim=owner,
				details={"item_id": target_gi.item.id, "item_value": target_gi.item.base_price},
			)
			owner_name = owner.name if owner else str(target_gi.owner_id)
			desc = (
				f"WARNING: {actor.name} steals {target_gi.item.name} from {owner_name}! "
				f"(value {target_gi.item.base_price}g)"
			)
			self._log_event(
				actor.id,
				"steal",
				desc,
				{"item_id": target_gi.item.id, "owner_id": target_gi.owner_id, "x": target_gi.x, "y": target_gi.y},
			)
		else:
			desc = f"{actor.name} picked up {target_gi.item.name} x{target_gi.item.quantity}"
			self._log_event(actor.id, "pick_up", desc, {"item_id": target_gi.item.id, "x": target_gi.x, "y": target_gi.y})
		return desc

	def _execute_place_action(self, actor: Actor, action: PlannedAction) -> str:
		"""Place an item from inventory onto a nearby tile."""
		try:
			p = PlaceParams.model_validate(action.params)
		except Exception as e:
			reason = f"Invalid place params: {e}"
			self.interrupt_actor(actor.id, reason)
			return reason

		if physics.distance_chebyshev(actor.x, actor.y, p.x, p.y) > 1:
			reason = f"Tile ({p.x},{p.y}) is too far away to place on"
			self.interrupt_actor(actor.id, reason)
			return reason

		if not self.world_map.in_bounds(p.x, p.y):
			reason = f"Tile ({p.x},{p.y}) is out of map bounds"
			self.interrupt_actor(actor.id, reason)
			return reason

		item = next((i for i in actor.inventory if i.id == p.item_id), None)
		if item is None:
			reason = f"{actor.name} does not have item '{p.item_id}' in inventory"
			self.interrupt_actor(actor.id, reason)
			return reason

		actor.inventory = [i for i in actor.inventory if i.id != p.item_id]
		tile = self.world_map.tile_at(p.x, p.y)
		# Items placed on a shop tile are owned by the placing actor (their shelf)
		owner_id = actor.id if tile.tile_type.value == "shop" else None
		self.world_map.world_items.append(WorldItem(item=item, x=p.x, y=p.y, owner_id=owner_id))
		loc = "shelf" if tile.tile_type.value == "shop" else "ground"
		desc = f"{actor.name} placed {item.name} on the {loc} at ({p.x},{p.y})"
		self._log_event(actor.id, "place", desc, {"item_id": item.id, "x": p.x, "y": p.y, "location": loc})
		return desc

	def _resolve_actor(self, target_id: str) -> Actor | None:
		"""Find an actor by id, falling back to case-insensitive name match."""
		by_id = next((a for a in self.world_map.actors if a.id == target_id), None)
		if by_id:
			return by_id
		lower = target_id.lower()
		return next((a for a in self.world_map.actors if a.name.lower() == lower), None)

	def _execute_converse_action(self, actor: Actor, action: PlannedAction) -> str:
		"""Execute a converse action: initiator speaks, listener replies, both record memory.

		Conversation packet generation is non-blocking: the LLM call is submitted to a
		dedicated executor and the action is re-queued until the reply is ready.
		"""
		from src.agents.prompts import CONVERSATION_RANGE
		try:
			p = ConverseParams.model_validate(action.params)
		except Exception as e:
			reason = f"Invalid converse params: {e}"
			self.interrupt_actor(actor.id, reason)
			return reason
		target = self._resolve_actor(p.target_actor_id)
		if target is None:
			return f"{actor.name} tries to speak but finds no one called '{p.target_actor_id}'."

		dist = physics.distance_chebyshev(actor.x, actor.y, target.x, target.y)
		if dist > CONVERSATION_RANGE:
			reason = f"{target.name} is {dist} tiles away — too far to converse"
			self.interrupt_actor(actor.id, reason)
			return reason

		opening_line = p.opening_line.strip() or "Hello."

		listener_brain = self.agent_brains.get(target.id)
		if listener_brain:
			# ── Non-blocking packet resolution ───────────────────────────────
			packet_future = self._pending_packet_futures.get(target.id)

			if packet_future is not None:
				if not packet_future.done():
					# LLM still generating — retry this action next tick
					actor.action_queue.insert(0, action)
					return f"{actor.name} waits for {target.name} to formulate a reply..."
				# Future finished — collect result
				try:
					c_reply, c_follow_ups, c_tone, c_impression = packet_future.result()
				except Exception:
					c_reply, c_follow_ups, c_tone, c_impression = "*nods but says nothing*", [], "", "No impression formed."
				self._pending_packet_futures.pop(target.id, None)
				reply = c_reply
				listener_impression = c_impression
				if c_follow_ups:
					self._conversation_packets[target.id] = ConversationPacket(
						initiator_id=actor.id,
						lines=c_follow_ups,
						created_tick=self.tick_count,
						tone=c_tone,
					)
			else:
				# Check cached packet first
				packet = self._conversation_packets.get(target.id)
				packet_valid = (
					packet is not None
					and packet.initiator_id == actor.id
					and (self.tick_count - packet.created_tick) < packet.packet_ttl_ticks
					and bool(packet.lines)
				)
				if packet_valid:
					reply = packet.lines.pop(0)
					listener_impression = f"(continuing, tone: {packet.tone})" if packet.tone else "(continuing)"
					if not packet.lines:
						del self._conversation_packets[target.id]
				else:
					# No cache and no pending future — submit non-blocking LLM call
					self._pending_packet_futures[target.id] = self._conversation_executor.submit(
						listener_brain.conduct_conversation_packet, actor, opening_line
					)
					self.llm_call_count += 1
					actor.action_queue.insert(0, action)
					return f"{actor.name} says to {target.name}: '{opening_line}' (waiting for reply...)"

		else:
			reply = "*nods but says nothing*"
			listener_impression = "No impression formed."

		# Log two separate events so they appear as distinct lines in the event log
		self._log_event(
			actor.id, "converse",
			f'{actor.name} says to {target.name}: "{opening_line}"',
			{"target_id": p.target_actor_id, "line": opening_line},
		)
		self._log_event(
			target.id, "converse",
			f'{target.name} replies to {actor.name}: "{reply}"',
			{"target_id": actor.id, "line": reply},
		)

		# Free-form dialogue can shape trust/suspicion.
		self._apply_dialogue_social_effects(speaker=actor, listener=target, line=opening_line)
		self._apply_dialogue_social_effects(speaker=target, listener=actor, line=reply)

		# Both actors record the exchange in their relationship memory
		initiator_brain = self.agent_brains.get(actor.id)
		if initiator_brain:
			initiator_brain.memory.record(
				tick=self.tick_count,
				actor_id=target.id,
				event_type="converse",
				notes=f"I said: \"{opening_line}\" — they replied: \"{reply}\"",
			)
		if listener_brain:
			listener_brain.memory.record(
				tick=self.tick_count,
				actor_id=actor.id,
				event_type="converse",
				notes=f"{actor.name} said: \"{opening_line}\" — I replied: \"{reply}\". My impression: {listener_impression}",
			)
			# The listener has already recorded the conversation in memory; they will
			# naturally incorporate it when their current plan exhausts and cadence allows.
			# No interrupt needed — conversations are non-critical events.

		return f'{actor.name} → {target.name}: "{opening_line}" | {target.name}: "{reply}"'

	def _execute_attack_action(self, actor: Actor, action: PlannedAction) -> str:
		"""Execute a melee attack against a target actor."""
		try:
			p = AttackParams.model_validate(action.params)
		except Exception as e:
			reason = f"Invalid attack params: {e}"
			self.interrupt_actor(actor.id, reason)
			return reason

		target = self._resolve_actor(p.target_actor_id)
		if target is None:
			reason = f"Attack target '{p.target_actor_id}' not found"
			self.interrupt_actor(actor.id, reason)
			return reason

		dist = physics.distance_chebyshev(actor.x, actor.y, target.x, target.y)
		if dist > physics.MELEE_RANGE:
			reason = f"{target.name} is {dist} tiles away — move closer to attack"
			self.interrupt_actor(actor.id, reason)
			return reason

		damage, is_crit = physics.calc_damage(actor, target)
		target.hp = max(0, target.hp - damage)
		self._last_attack_tick[(actor.id, target.id)] = self.tick_count

		# Unprovoked witnessed assault is a crime; retaliation in recent self-defense window is exempt.
		if not self._is_self_defense(actor, target):
			severity = self._assault_severity(damage)
			self._register_crime(
				offender=actor,
				crime_type="assault",
				severity=severity,
				x=actor.x,
				y=actor.y,
				victim=target,
				details={"target_id": target.id, "damage": damage},
			)

		crit_str = " (CRITICAL HIT!)" if is_crit else ""
		desc = (
			f"{actor.name} attacks {target.name} for {damage} damage{crit_str}! "
			f"{target.name} HP: {target.hp}/{target.max_hp}"
		)
		self._log_event(actor.id, "attack", desc, {
			"target_id": target.id,
			"damage": damage,
			"is_crit": is_crit,
			"target_hp": target.hp,
		})
		self._log_event(actor.id, "combat", desc, {
			"target_id": target.id,
			"damage": damage,
			"is_crit": is_crit,
			"target_hp": target.hp,
			"source": "attack",
		})

		if target.hp == 0:
			self._handle_actor_death(target, killer=actor)

		return desc

	def _handle_actor_death(self, actor: Actor, killer: Actor | None = None) -> None:
		"""Remove a dead actor: drop their inventory + equipped items and a corpse WorldItem.

		- All personal inventory items land as unowned floor WorldItems at the actor's position.
		- Equipped items are also dropped.
		- A 'corpse' WorldItem (unique id, non-tradeable relic) is placed at the same tile.
		- The actor is removed from world_map.actors.
		- Other actors whose plans reference this actor will get an interrupt naturally
		  the next time they try to target them.
		"""
		killer_name = killer.name if killer else "unknown causes"
		death_desc = f"{actor.name} has been slain by {killer_name}!"
		self._log_event(
			actor.id, "death", death_desc,
			{"killer_id": killer.id if killer else None, "x": actor.x, "y": actor.y},
		)

		# Drop all personal inventory items
		for item in actor.inventory:
			self.world_map.world_items.append(
				WorldItem(item=item, x=actor.x, y=actor.y, owner_id=None)
			)
		actor.inventory.clear()

		# Drop all equipped items
		for slot, item in actor.equipped.items():
			if item is not None:
				self.world_map.world_items.append(
					WorldItem(item=item, x=actor.x, y=actor.y, owner_id=None)
				)
				actor.equipped[slot] = None

		# Drop a corpse marker
		corpse_item = Item(
			id=f"corpse_{actor.id}",
			name=f"Corpse of {actor.name}",
			description=(
				f"The lifeless body of {actor.name}, slain by {killer_name}. "
				"Grim testament to the dangers of this market."
			),
			base_price=0,
			quantity=1,
			metadata={"category": "corpse", "type": "body", "origin_id": actor.id},
		)
		self.world_map.world_items.append(
			WorldItem(item=corpse_item, x=actor.x, y=actor.y, owner_id=None)
		)

		# Remove from simulation
		self.world_map.actors = [a for a in self.world_map.actors if a.id != actor.id]
		self.agent_brains.pop(actor.id, None)

	def _carry_count(self, actor: Actor) -> int:
		"""Number of item stacks the actor is carrying in personal inventory.

		Shelf items (WorldItems with owner_id == actor.id) are NOT counted.
		"""
		return len(actor.inventory)

	def _execute_equip_action(self, actor: Actor, action: PlannedAction) -> str:
		"""Move an item from inventory into its equipment slot."""
		try:
			p = EquipParams.model_validate(action.params)
		except Exception as e:
			reason = f"Invalid equip params: {e}"
			self.interrupt_actor(actor.id, reason)
			return reason

		# Find item in inventory
		item = next((i for i in actor.inventory if i.id == p.item_id), None)
		if item is None:
			reason = f"{actor.name} does not have '{p.item_id}' in inventory"
			self.interrupt_actor(actor.id, reason)
			return reason

		raw_slot = item.metadata.get("slot", "")
		if not raw_slot:
			reason = f"'{item.name}' has no equipment slot defined"
			self.interrupt_actor(actor.id, reason)
			return reason

		# two_hand weapons occupy the 'hand' slot and clear 'offhand'
		if raw_slot == "two_hand":
			equip_slot = "hand"
			# Return any existing offhand to inventory
			old_offhand = actor.equipped.get("offhand")
			if old_offhand is not None:
				actor.inventory.append(old_offhand)
				actor.equipped["offhand"] = None
		else:
			equip_slot = raw_slot

		if equip_slot not in actor.equipped:
			reason = f"Equipment slot '{equip_slot}' is not valid"
			self.interrupt_actor(actor.id, reason)
			return reason

		# Block equipping an offhand while a two-hand weapon is in hand
		if equip_slot == "offhand":
			hand_item = actor.equipped.get("hand")
			if hand_item is not None and hand_item.metadata.get("slot") == "two_hand":
				reason = f"Cannot equip offhand while wielding two-handed {hand_item.name}"
				self.interrupt_actor(actor.id, reason)
				return reason

		# Return any currently equipped item in that slot to inventory
		old_item = actor.equipped.get(equip_slot)
		if old_item is not None:
			actor.inventory.append(old_item)

		actor.inventory.remove(item)
		actor.equipped[equip_slot] = item

		desc = f"{actor.name} equipped {item.name} to {equip_slot} slot"
		self._log_event(actor.id, "equip", desc, {"item_id": item.id, "slot": equip_slot})
		return desc

	def _execute_unequip_action(self, actor: Actor, action: PlannedAction) -> str:
		"""Move an equipped item back to inventory."""
		try:
			p = UnequipParams.model_validate(action.params)
		except Exception as e:
			reason = f"Invalid unequip params: {e}"
			self.interrupt_actor(actor.id, reason)
			return reason

		slot = p.slot
		if slot not in actor.equipped:
			reason = f"'{slot}' is not a valid equipment slot"
			self.interrupt_actor(actor.id, reason)
			return reason

		item = actor.equipped.get(slot)
		if item is None:
			desc = f"{actor.name} has nothing in the {slot} slot"
			return desc

		actor.equipped[slot] = None
		actor.inventory.append(item)

		desc = f"{actor.name} unequipped {item.name} from {slot} slot"
		self._log_event(actor.id, "unequip", desc, {"item_id": item.id, "slot": slot})
		return desc

	def _execute_use_item_action(self, actor: Actor, action: PlannedAction) -> str:
		"""Consume a consumable item from inventory, applying its effects."""
		try:
			p = UseItemParams.model_validate(action.params)
		except Exception as e:
			reason = f"Invalid use_item params: {e}"
			self.interrupt_actor(actor.id, reason)
			return reason

		item = next((i for i in actor.inventory if i.id == p.item_id), None)
		if item is None:
			reason = f"{actor.name} does not have '{p.item_id}' in inventory"
			self.interrupt_actor(actor.id, reason)
			return reason

		category = item.metadata.get("category", "")
		if category != "consumable":
			reason = f"'{item.name}' is not a consumable item"
			self.interrupt_actor(actor.id, reason)
			return reason

		effects = []

		if item.hp_delta != 0:
			old_hp = actor.hp
			actor.hp = max(0, min(actor.max_hp, actor.hp + item.hp_delta))
			actual = actor.hp - old_hp
			if actual >= 0:
				effects.append(f"+{actual} HP ({actor.hp}/{actor.max_hp})")
			else:
				effects.append(f"{actual} HP ({actor.hp}/{actor.max_hp})")

		if item.attack_delta != 0:
			actor.base_attack = max(0, actor.base_attack + item.attack_delta)
			effects.append(f"attack {'+'if item.attack_delta>0 else ''}{item.attack_delta} → {actor.base_attack}")

		if item.defense_delta != 0:
			actor.base_defense = max(0, actor.base_defense + item.defense_delta)
			effects.append(f"defense {'+'if item.defense_delta>0 else ''}{item.defense_delta} → {actor.base_defense}")

		if item.crit_delta != 0.0:
			actor.base_crit = max(0.0, min(1.0, actor.base_crit + item.crit_delta))
			effects.append(f"crit {'+'if item.crit_delta>0 else ''}{item.crit_delta:.2f} → {actor.base_crit:.2f}")

		actor.inventory.remove(item)

		effects_str = ", ".join(effects) if effects else "no effect"
		desc = f"{actor.name} used {item.name}: {effects_str}"
		self._log_event(actor.id, "use_item", desc, {"item_id": item.id, "effects": effects})
		return desc

	def _execute_trade_action(self, actor: Actor, params: dict) -> str:
		"""Execute a propose_trade action step."""
		try:
			p = ProposeTradeParams.model_validate(params)
		except Exception as e:
			reason = f"Invalid propose_trade params: {e}"
			self.interrupt_actor(actor.id, reason)
			return reason
		target = self._resolve_actor(p.target_actor_id)
		if target is None:
			reason = f"Trade target '{p.target_actor_id}' not found"
			self.interrupt_actor(actor.id, reason)
			return reason

		dist = physics.distance_chebyshev(actor.x, actor.y, target.x, target.y)
		if dist > 2:
			reason = f"{target.name} is {dist} tiles away — need to move closer first"
			self.interrupt_actor(actor.id, reason)
			return reason

		def _resolve_items(id_str: str, owner: Actor) -> tuple[list, str | None]:
			if not id_str or not id_str.strip():
				return [], None
			# Effective inventory = carried items + owned shelf items
			eff_inv = self._effective_inventory(owner)
			inv = {i.id: i for i in eff_inv}
			matched = []
			for iid in [s.strip() for s in id_str.split(",") if s.strip()]:
				if iid not in inv:
					return [], f"{owner.name} does not have item '{iid}'"
				matched.append(inv[iid])
			return matched, None

		offered_items, err = _resolve_items(p.offered_item_ids, actor)
		if err:
			self.interrupt_actor(actor.id, err)
			return err
		requested_items, err = _resolve_items(p.requested_item_ids, target)
		if err:
			self.interrupt_actor(actor.id, err)
			return err

		proposal = TradeProposal(
			proposer_id=actor.id,
			target_id=target.id,
			offered_items=offered_items,
			offered_gold=p.offered_gold,
			requested_items=requested_items,
			requested_gold=p.requested_gold,
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
					"fame": actor.fame,
					"infamy": actor.infamy,
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
		Handles shelf items (owned WorldItems) on the target's side.
		"""
		# Re-validate gold
		if proposal.offered_gold > proposer.gold:
			return f"{proposer.name} no longer has enough gold."
		if proposal.requested_gold > target.gold:
			return f"{target.name} no longer has enough gold."

		# Re-validate items (proposer must carry offered items)
		proposer_carried = {item.id: item for item in proposer.inventory}
		for item in proposal.offered_items:
			if item.id not in proposer_carried:
				return f"{proposer.name} no longer has {item.name}."

		# Carry-capacity checks before committing the transfer
		items_proposer_gains = len(proposal.requested_items)
		items_target_gains = len(proposal.offered_items)
		proposer_free = proposer.max_carry - self._carry_count(proposer)
		target_free = target.max_carry - self._carry_count(target)
		# Adjust for items leaving each inventory
		proposer_net = items_proposer_gains - len(proposal.offered_items)
		target_net = items_target_gains - len(proposal.requested_items)
		if proposer_net > proposer_free:
			return f"{proposer.name} doesn't have enough carry space for this trade."
		if target_net > target_free:
			return f"{target.name} doesn't have enough carry space for this trade."

		# Requested items may be in target's carried inventory OR on their shelf
		target_carried = {item.id: item for item in target.inventory}
		target_shelf: dict[str, WorldItem] = {
			gi.item.id: gi
			for gi in self.world_map.world_items
			if gi.owner_id == target.id
		}
		for item in proposal.requested_items:
			if item.id not in target_carried and item.id not in target_shelf:
				return f"{target.name} no longer has {item.name}."

		# Transfer gold
		proposer.gold -= proposal.offered_gold
		proposer.gold += proposal.requested_gold
		target.gold += proposal.offered_gold
		target.gold -= proposal.requested_gold

		# Transfer offered items: proposer carried → target inventory
		for item in proposal.offered_items:
			proposer.inventory = [i for i in proposer.inventory if i.id != item.id]
			target.inventory.append(item)

		# Transfer requested items: target (carried or shelf) → proposer inventory
		for item in proposal.requested_items:
			if item.id in target_shelf:
				# Remove from the world items list
				self.world_map.world_items = [
					gi for gi in self.world_map.world_items if gi.item.id != item.id
				]
			else:
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
