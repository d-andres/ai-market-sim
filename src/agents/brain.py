"""Agent brain: LLM-driven decision making via Ollama.

This module connects actors to their AI "brains" which can:
- Observe the world around them (including visible inventories)
- Remember past interactions and relationships
- Propose and evaluate trades using LLM judgment
- Move and act in the world

Uses the local Ollama server for all LLM inference.
See AI-AGENT-INTEGRATION.md for setup instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import TYPE_CHECKING

from src.agents.ollama_client import OllamaClient, ChatMessage, MessageRole
from src.models.schema import Actor, ActorRole, Map, PlannedAction, TradeProposal
from src.simulation import physics
from src.agents.prompts import (
	TRADE_EVALUATION_PROMPT,
	CONVERSATION_RANGE,
	DIALOGUE_SOCIAL_IMPACT_PROMPT,
	ADAPTIVE_FALLBACK_POLICY_PROMPT,
	ADAPTIVE_FALLBACK_PLAN_PROMPT,
	get_system_prompt_for_role,
)
from src.agents.response_parser import clean, parse_plan, parse_trade_decision

if TYPE_CHECKING:
	from src.simulation.engine import SimulationEngine


def _extract_json_object(raw: str) -> dict | None:
	"""Extract and parse the first JSON object from model output."""
	text = clean(raw)
	match = re.search(r"\{[\s\S]*\}", text)
	if not match:
		return None
	try:
		parsed = json.loads(match.group(0))
		return parsed if isinstance(parsed, dict) else None
	except Exception:
		return None


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

class ObserveSurroundingsTool:
	"""Builds a text description of an actor's surroundings for LLM prompts."""
	
	def __init__(
		self,
		actor: Actor,
		world_map: Map,
		memory: RelationshipMemory,
		engine: SimulationEngine | None = None,
	):
		self.actor = actor
		self.world_map = world_map
		self.memory = memory
		self.engine = engine
	
	def forward(self, vision_range: int = 10) -> str:
		viewport = physics.get_visible_tiles_and_actors(
			self.world_map, self.actor, vision_range=vision_range
		)

		lines: list[str] = []
		lines.append(f"You are at ({self.actor.x}, {self.actor.y}).")
		lines.append(f"Your gold: {self.actor.gold}g")
		lines.append(f"Your social standing: fame {self.actor.fame}, infamy {self.actor.infamy}")
		lines.append(
			f"Your combat stats: HP {self.actor.hp}/{self.actor.max_hp}, "
			f"ATK {self.actor.base_attack}, DEF {self.actor.base_defense}, "
			f"CRIT {self.actor.base_crit:.0%}"
		)

		if self.actor.inventory:
			inv = ", ".join(f"{i.name} x{i.quantity}" for i in self.actor.inventory)
			lines.append(f"Your inventory ({len(self.actor.inventory)}/{self.actor.max_carry}): {inv}")
		else:
			lines.append(f"Your inventory (0/{self.actor.max_carry}): (empty)")

		equipped_strs = [
			f"{slot}: {item.name}" for slot, item in self.actor.equipped.items() if item is not None
		]
		if equipped_strs:
			lines.append(f"Your equipped: {', '.join(equipped_strs)}")
		else:
			lines.append("Your equipped: (nothing)")

		if viewport.visible_actors:
			lines.append("\nVisible actors:")
			for other in viewport.visible_actors:
				suspicion = self.engine.suspicion_score(self.actor.id, other.id) if self.engine else 0
				dist = physics.distance_chebyshev(
					self.actor.x, self.actor.y, other.x, other.y
				)
				# Effective inventory = carried + shelf items owned by this actor
				# Only visible when within conversation/trade range
				if dist <= CONVERSATION_RANGE:
					eff_items = list(other.inventory) + [
						gi.item for gi in self.world_map.world_items if gi.owner_id == other.id
					]
					inv_str = (
						", ".join(f"{i.name} (id:{i.id}, base {i.base_price}g) x{i.quantity}" for i in eff_items)
						if eff_items else "nothing visible"
					)
				else:
					inv_str = "(too far to see details)"
				# Compute explicit move direction so the LLM never has to guess
				dx = other.x - self.actor.x
				dy = other.y - self.actor.y
				v = ("south" if dy > 0 else "north" if dy < 0 else "")
				h = ("east"  if dx > 0 else "west"  if dx < 0 else "")
				direction_hint = (f"{v}{h}" if v and h else v or h or "here")
				lines.append(
					f"  - {other.name} (id: {other.id}) [{other.role.value}] at ({other.x},{other.y}), "
					f"dist {dist}, direction: {direction_hint}"
					+ (" [TRADE RANGE — use propose_trade]" if dist <= 2 else "")
					+ (" [CONVERSATION RANGE — can use converse]" if dist <= CONVERSATION_RANGE else "")
					+ (" [ATTACK RANGE — can use attack]" if dist <= 1 else "")
					+ f", gold {other.gold}g, HP {other.hp}/{other.max_hp}, fame {other.fame}, infamy {other.infamy}"
					+ f", your_likeness {self.actor.likeness.get(other.id, 0)}"
					+ f", your_suspicion {suspicion}"
					+ f" | carrying/selling: {inv_str}"
				)
				history = self.memory.summary_for(other.id, other.name)
				lines.append(f"    History: {history}")
		else:
			lines.append("\nNo other actors visible.")

		# Ground/shelf items in view (owned shelf items can be stolen)
		# Items at shop tiles owned by other actors are hidden until the actor
		# is adjacent (dist <= 2) — actors must explore to discover shop inventory.
		DISCOVERY_RANGE = 2
		visible_set = set(viewport.visible_tiles)
		visible_gis = [
			gi for gi in self.world_map.world_items
			if (gi.x, gi.y) in visible_set
		]
		carry_full = len(self.actor.inventory) >= self.actor.max_carry
		if visible_gis:
			header = "\nItems in view (pick_up allowed in range; owned items count as theft):"
			if carry_full:
				header += " [INVENTORY FULL — drop/place something first]"
			lines.append(header)
			by_pos: dict[tuple[int, int], list] = {}
			for gi in visible_gis:
				by_pos.setdefault((gi.x, gi.y), []).append(gi)
			for (gx, gy), gis in sorted(by_pos.items()):
				dist = physics.distance_chebyshev(self.actor.x, self.actor.y, gx, gy)
				tile = self.world_map.tile_at(gx, gy)
				is_shop_tile = tile.tile_type.value == "shop"
				# Hide shop item details if the actor is too far away to inspect
				if is_shop_tile and dist > DISCOVERY_RANGE:
					lines.append(f"  - ({gx},{gy}) dist {dist}: [Shop shelf — move closer to inspect items]")
					continue
				item_strs = []
				for gi in gis:
					is_corpse = gi.item.metadata.get("category") == "corpse"
					if is_corpse:
						label = f"{gi.item.name} (id:{gi.item.id}) [CORPSE]"
					elif gi.owner_id is None or gi.owner_id == self.actor.id:
						label = f"{gi.item.name} (id:{gi.item.id}) x{gi.item.quantity} [FREE PICKUP]"
					else:
						label = (
							f"{gi.item.name} (id:{gi.item.id}) x{gi.item.quantity} "
							f"[OWNED by {gi.owner_id} — PICKUP IS THEFT, severity by value {gi.item.base_price}g]"
						)
					item_strs.append(label)
				pickup_hint = " [PICK UP RANGE]" if dist <= 1 and not carry_full else ""
				lines.append(f"  - ({gx},{gy}) dist {dist}{pickup_hint}: {', '.join(item_strs)}")

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
	# Set high enough so actors have actions to execute between replan intervals.
	PLAN_HORIZON: int = 20
	# AI-authored fallback profile refresh cadence and fallback planning budget.
	FALLBACK_POLICY_TTL_TICKS: int = 40
	FALLBACK_PLAN_HORIZON: int = 10
	FALLBACK_MAX_ATTEMPTS: int = 1

	def __init__(
		self,
		actor: Actor,
		world_map: Map,
		engine: SimulationEngine,
		model_name: str = "qwen3:4b",
		api_base: str = "http://localhost:11434",
	):
		self.actor = actor
		self.world_map = world_map
		self.engine = engine
		self.memory = RelationshipMemory()

		# Thinking mode is enabled so the model separates its reasoning (hidden
		# in the ``thinking`` field) from the structured JSON output (in
		# ``content``).  With ``think=False`` qwen3 dumps verbose reasoning
		# directly into the content, making JSON extraction unreliable.
		# max_tokens must be large enough for both thinking + content tokens.
		# A generous timeout (300s) accommodates concurrent requests queuing
		# behind each other on the Ollama server.
		self.model = OllamaClient(
			model_id=model_name,
			api_base=api_base,
			timeout=300,
			max_tokens=8192,
			num_ctx=4096,
			think=True,
		)
		self.system_prompt = get_system_prompt_for_role(actor.role)
		self.obs_tool = ObserveSurroundingsTool(
			actor=actor, world_map=world_map, memory=self.memory, engine=engine
		)
		self._adaptive_fallback_policy: dict | None = None
		self._adaptive_fallback_policy_tick: int = -10_000
		# Metrics: last request/response for the engine to read after completion.
		self.last_request_prompt: str = ""
		self.last_response_raw: str = ""

		# Per-actor conversation log: captures every LLM exchange with thinking.
		# Each entry: {tick, call_type, prompt, thinking, response, duration}
		self.conversation_log: list[dict] = []
		self.CONVERSATION_LOG_MAX: int = 50

		# Separate lightweight model for health checks (short timeout).
		# Thinking is disabled so all tokens go to the content response.
		self._health_check_model = OllamaClient(
			model_id=model_name,
			api_base=api_base,
			timeout=30,
			max_tokens=128,
			num_ctx=8192,
			think=False,
		)

	def _log_llm_call(self, call_type: str, prompt: str, thinking: str, response: str, duration: float) -> None:
		"""Append an LLM exchange to the per-actor conversation log."""
		self.conversation_log.append({
			"tick": self.engine.tick_count if self.engine else 0,
			"call_type": call_type,
			"prompt": prompt,
			"thinking": thinking,
			"response": response,
			"duration": round(duration, 2),
		})
		# Keep bounded.
		if len(self.conversation_log) > self.CONVERSATION_LOG_MAX:
			self.conversation_log = self.conversation_log[-self.CONVERSATION_LOG_MAX:]

	def health_check(self) -> tuple[bool, str]:
		"""Send a trivial prompt to verify the LLM is responding.

		Returns (alive, detail) where alive is True if a response was received.
		Uses a dedicated model instance with a short 30-second timeout.
		"""
		messages = [
			ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Respond with exactly: OK"}]),
		]
		try:
			response = self._health_check_model(messages)
			content = response.content if hasattr(response, "content") else str(response)
			thinking = response.thinking if hasattr(response, "thinking") else ""
			self._log_llm_call("health_check", "Respond with exactly: OK", thinking, content.strip()[:200], response.duration if hasattr(response, "duration") else 0)
			return True, content.strip()[:200]
		except Exception as e:
			self._log_llm_call("health_check", "Respond with exactly: OK", "", f"[ERROR] {e}", 0)
			return False, str(e)[:200]

	def _role_goal_directive(self) -> str:
		"""Return the high-level role objective for strategic planning prompts."""
		if self.actor.role == ActorRole.GUARD:
			return "Maintain order, investigate suspicious behavior, and keep visible protective presence."
		if self.actor.role == ActorRole.SHOPKEEPER:
			return "Protect stock, engage customers, and optimize profitable trades without inviting danger."
		return "Acquire high-value items and adapt tactics to risk, personality, and social consequences."

	def _traits_text(self) -> str:
		traits = getattr(self.actor, "personality_traits", None) or []
		if not traits:
			return "role-default"
		return ", ".join(str(t).strip() for t in traits if str(t).strip())

	def _is_low_quality_plan(self, plan: list[PlannedAction]) -> bool:
		"""Treat empty or all-wait plans as low quality."""
		if not plan:
			return True
		non_wait = [a for a in plan if a.action_type != "wait"]
		if not non_wait:
			return True
		return False

	def _normalize_fallback_policy(self, raw_policy: dict | None, refresh_reason: str) -> dict:
		"""Sanitize policy payload from model into a stable JSON structure."""
		raw_policy = raw_policy or {}
		identity_summary = str(raw_policy.get("identity_summary") or "").strip()
		if not identity_summary:
			identity_summary = f"{self.actor.name} acts according to role obligations and personal temperament."

		goals = [
			str(g).strip()
			for g in (raw_policy.get("goals") or [])
			if str(g).strip()
		][:5]
		if not goals:
			goals = [self._role_goal_directive()]

		preferred_tactics = [
			str(t).strip()
			for t in (raw_policy.get("preferred_tactics") or [])
			if str(t).strip()
		][:6]

		revision_triggers = [
			str(t).strip()
			for t in (raw_policy.get("revision_triggers") or [])
			if str(t).strip()
		][:6]

		risk_posture = str(raw_policy.get("risk_posture") or "medium").strip().lower()
		if risk_posture not in {"low", "medium", "high"}:
			risk_posture = "medium"

		return {
			"identity_summary": identity_summary,
			"goals": goals,
			"preferred_tactics": preferred_tactics,
			"risk_posture": risk_posture,
			"revision_triggers": revision_triggers,
			"refresh_reason": refresh_reason,
		}

	def _refresh_adaptive_fallback_policy(self, observation: str, refresh_reason: str) -> None:
		"""Ask the LLM to author/adjust a compact fallback strategy profile."""
		recent_events = self.engine.get_event_log(limit=8)
		recent_lines = [f"[tick {e['tick']}] {e['description']}" for e in recent_events]
		prompt = ADAPTIVE_FALLBACK_POLICY_PROMPT.format(
			actor_name=self.actor.name,
			actor_role=self.actor.role.value,
			traits=self._traits_text(),
			role_goal=self._role_goal_directive(),
			tick=self.engine.tick_count,
			hp=self.actor.hp,
			max_hp=self.actor.max_hp,
			gold=self.actor.gold,
			fame=self.actor.fame,
			infamy=self.actor.infamy,
			refresh_reason=refresh_reason,
			recent_events="\n".join(recent_lines) if recent_lines else "(none)",
			observation=observation,
		)

		messages = [
			ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}]),
			ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}]),
		]

		try:
			response = self.model(messages)
			raw = response.content if hasattr(response, "content") else str(response)
			thinking = response.thinking if hasattr(response, "thinking") else ""
			self._log_llm_call("fallback_policy", prompt, thinking, raw, response.duration if hasattr(response, "duration") else 0)
			parsed = _extract_json_object(raw)
			self._adaptive_fallback_policy = self._normalize_fallback_policy(parsed, refresh_reason)
			self._adaptive_fallback_policy_tick = self.engine.tick_count
		except Exception:
			if self._adaptive_fallback_policy is None:
				self._adaptive_fallback_policy = self._normalize_fallback_policy(None, refresh_reason)
				self._adaptive_fallback_policy_tick = self.engine.tick_count

	def _ensure_adaptive_fallback_policy(
		self,
		observation: str,
		refresh_reason: str,
		force_refresh: bool = False,
	) -> None:
		stale = (self.engine.tick_count - self._adaptive_fallback_policy_tick) >= self.FALLBACK_POLICY_TTL_TICKS
		if force_refresh or self._adaptive_fallback_policy is None or stale:
			self._refresh_adaptive_fallback_policy(observation, refresh_reason)

	def _adaptive_fallback_plan(
		self,
		observation: str,
		interrupt_reason: str,
		failure_reason: str,
	) -> tuple[list[PlannedAction], str]:
		"""Generate a replacement plan using an AI-authored fallback policy."""
		self._ensure_adaptive_fallback_policy(
			observation,
			failure_reason,
			force_refresh=bool(interrupt_reason),
		)

		for attempt in range(self.FALLBACK_MAX_ATTEMPTS):
			policy = self._adaptive_fallback_policy or self._normalize_fallback_policy(None, failure_reason)
			prompt = ADAPTIVE_FALLBACK_PLAN_PROMPT.format(
				actor_name=self.actor.name,
				actor_role=self.actor.role.value,
				traits=self._traits_text(),
				tick=self.engine.tick_count,
				horizon=self.FALLBACK_PLAN_HORIZON,
				failure_reason=failure_reason,
				interrupt_reason=interrupt_reason or "(none)",
				policy_json=json.dumps(policy, ensure_ascii=True),
				observation=observation,
			)

			messages = [
				ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}]),
				ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}]),
			]

			try:
				response = self.model(messages)
				content = response.content if hasattr(response, "content") else str(response)
				thinking = response.thinking if hasattr(response, "thinking") else ""
				self._log_llm_call("fallback_plan", prompt, thinking, content, response.duration if hasattr(response, "duration") else 0)
				plan, summary = parse_plan(content, self.actor.name)
				if not self._is_low_quality_plan(plan):
					return plan, f"{summary} (adaptive fallback)"
			except Exception:
				pass

			if attempt + 1 < self.FALLBACK_MAX_ATTEMPTS:
				self._ensure_adaptive_fallback_policy(
					observation,
					refresh_reason=f"retry_after_{failure_reason}",
					force_refresh=True,
				)

		return [PlannedAction(action_type="wait", params={}, reason="adaptive fallback exhausted")], (
			f"{self.actor.name} pauses briefly to reassess strategy."
		)

	def create_plan(
		self,
		interrupt_reason: str = "",
		allow_adaptive_fallback: bool = True,
	) -> tuple[list[PlannedAction], str]:
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
		role_goal = self._role_goal_directive()

		user_msg = (
			f"Tick {self.engine.tick_count} | HP: {self.actor.hp}/{self.actor.max_hp} | "
			f"Gold: {self.actor.gold}g | Fame: {self.actor.fame} | Infamy: {self.actor.infamy} | Carrying: {inv_summary}\n\n"
			f"{role_goal}\n\n"
			f"WORLD OBSERVATION:\n{observation}"
			f"{interrupt_ctx}\n\n"
			f"Create a plan of {self.PLAN_HORIZON} steps.\n"
			"Return ONLY a JSON object: {\"summary\": \"<one sentence>\", \"plan\": [...]}\n"
			"Each step is ONE of:\n"
			"  {\"action_type\":\"move\",\"params\":{\"direction\":\"north|south|east|west|ne|nw|se|sw\"},\"reason\":\"\"}\n"
			"  {\"action_type\":\"wait\",\"params\":{},\"reason\":\"\"}\n"
			"  {\"action_type\":\"propose_trade\",\"params\":{\"target_actor_id\":\"\",\"offered_gold\":0,\"offered_item_ids\":\"\",\"requested_item_ids\":\"\",\"requested_gold\":0},\"reason\":\"\"}\n"
			"  {\"action_type\":\"converse\",\"params\":{\"target_actor_id\":\"\",\"opening_line\":\"\"},\"reason\":\"\"}\n"
			"  {\"action_type\":\"pick_up\",\"params\":{\"item_id\":\"\"},\"reason\":\"\"}\n"
			"  {\"action_type\":\"attack\",\"params\":{\"target_actor_id\":\"\"},\"reason\":\"\"}\n"
			"  {\"action_type\":\"equip\",\"params\":{\"item_id\":\"\"},\"reason\":\"\"}\n"
			"No prose. No markdown. Only the JSON object."
		)

		messages = [
			ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}]),
			ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": user_msg}]),
		]

		self.last_request_prompt = user_msg
		self.last_response_raw = ""

		try:
			response = self.model(messages)
			content: str = response.content if hasattr(response, "content") else str(response)
			thinking: str = response.thinking if hasattr(response, "thinking") else ""
			self.last_response_raw = content
			self._log_llm_call("plan", user_msg, thinking, content, response.duration if hasattr(response, "duration") else 0)
			plan, summary = parse_plan(content, self.actor.name)
			if self._is_low_quality_plan(plan):
				if not allow_adaptive_fallback:
					return [PlannedAction(action_type="wait", params={}, reason="initial warm-up fallback")], (
						f"{self.actor.name} could not form a reliable initial plan and will adapt after simulation starts."
					)
				return self._adaptive_fallback_plan(
					observation=observation,
					interrupt_reason=interrupt_reason,
					failure_reason="low_quality_or_idle_primary_plan",
				)
			return plan, summary
		except Exception as e:
			self.last_response_raw = f"[ERROR] {e}"
			self._log_llm_call("plan", user_msg, "", f"[ERROR] {e}", 0)
			if not allow_adaptive_fallback:
				return [PlannedAction(action_type="wait", params={}, reason=f"initial warm-up failed: {e}")], (
					f"{self.actor.name} failed initial planning and will recover after startup."
				)
			return self._adaptive_fallback_plan(
				observation=observation,
				interrupt_reason=interrupt_reason,
				failure_reason=f"primary_plan_generation_failed: {e}",
			)

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
			thinking: str = response.thinking if hasattr(response, "thinking") else ""
			self._log_llm_call("trade_eval", prompt, thinking, raw, response.duration if hasattr(response, "duration") else 0)
		except Exception as e:
			self._log_llm_call("trade_eval", prompt, "", f"[ERROR] {e}", 0)
			return False, f"{self.actor.name} couldn't process the offer right now."

		return parse_trade_decision(raw, self.actor.name)

	def conduct_conversation_packet(
		self,
		initiator: Actor,
		opening_line: str,
	) -> tuple[str, list[str], str, str]:
		"""Respond in-character and return pre-planned follow-up lines in one LLM call.

		Returns (reply, follow_ups, tone, impression).
		- reply: immediate spoken response
		- follow_ups: 0-2 pre-planned continuation lines for subsequent turns
		- tone: single-word emotional tone descriptor
		- impression: private assessment recorded in relationship memory
		"""
		from src.agents.prompts import CONVERSATION_PACKET_PROMPT
		relationship = self.memory.summary_for(initiator.id, initiator.name)
		prompt = CONVERSATION_PACKET_PROMPT.format(
			speaker_name=self.actor.name,
			speaker_role=self.actor.role.value,
			listener_name=initiator.name,
			listener_role=initiator.role.value,
			relationship_history=relationship,
			opening_line=opening_line,
		)
		try:
			response = self.model([
				ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}]),
				ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}]),
			])
			raw: str = response.content if hasattr(response, "content") else str(response)
			thinking: str = response.thinking if hasattr(response, "thinking") else ""
			self._log_llm_call("conversation", prompt, thinking, raw, response.duration if hasattr(response, "duration") else 0)
		except Exception:
			self._log_llm_call("conversation", prompt, "", "[ERROR]", 0)
			return f"{self.actor.name} says nothing.", [], "", "No impression formed."
		return _parse_conversation_packet_response(raw, self.actor.name)

	def assess_dialogue_social_impact(
		self,
		speaker: Actor,
		line: str,
		candidate_actors: list[Actor],
		suspicion_snapshot: dict[str, int],
		recent_events: list[str],
	) -> dict:
		"""Ask the listener LLM how free-form dialogue should alter trust/suspicion.

		This is intentionally advisory: the engine clamps and applies outputs conservatively.
		"""
		relationship = self.memory.summary_for(speaker.id, speaker.name)
		candidates = [
			{
				"id": a.id,
				"name": a.name,
				"role": a.role.value,
				"fame": a.fame,
				"infamy": a.infamy,
				"likeness": self.actor.likeness.get(a.id, 0),
				"suspicion": suspicion_snapshot.get(a.id, 0),
			}
			for a in candidate_actors
			if a.id != self.actor.id
		]

		prompt = DIALOGUE_SOCIAL_IMPACT_PROMPT.format(
			listener_name=self.actor.name,
			listener_role=self.actor.role.value,
			listener_fame=self.actor.fame,
			listener_infamy=self.actor.infamy,
			listener_hp=self.actor.hp,
			listener_max_hp=self.actor.max_hp,
			speaker_name=speaker.name,
			speaker_role=speaker.role.value,
			speaker_fame=speaker.fame,
			speaker_infamy=speaker.infamy,
			speaker_hp=speaker.hp,
			speaker_max_hp=speaker.max_hp,
			likeness_to_speaker=self.actor.likeness.get(speaker.id, 0),
			relationship_history=relationship,
			recent_events="\n".join(recent_events) if recent_events else "(none)",
			suspicion_snapshot=json.dumps(suspicion_snapshot, ensure_ascii=True),
			candidate_actors=json.dumps(candidates, ensure_ascii=True),
			line=line,
		)

		try:
			response = self.model([
				ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}]),
				ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}]),
			])
			raw = response.content if hasattr(response, "content") else str(response)
			thinking = response.thinking if hasattr(response, "thinking") else ""
			self._log_llm_call("social_impact", prompt, thinking, raw, response.duration if hasattr(response, "duration") else 0)
		except Exception:
			self._log_llm_call("social_impact", prompt, "", "[ERROR]", 0)
			return {
				"likeness_delta": 0,
				"suspicion_deltas": {},
				"action_bias": "watch",
				"confidence": 0.0,
				"rationale": "llm_unavailable",
			}

		return _parse_dialogue_social_impact_response(raw, candidates)

def _parse_conversation_packet_response(raw: str, actor_name: str) -> tuple[str, list[str], str, str]:
	"""Extract REPLY / FOLLOW_UP_n / IMPRESSION / TONE lines from a packet prompt response."""
	from src.agents.response_parser import clean
	content = clean(raw)
	reply = f"{actor_name} nods silently."
	follow_ups: list[str] = []
	impression = "No strong impression."
	tone = ""
	for line in content.splitlines():
		line = line.strip()
		upper = line.upper()
		if upper.startswith("REPLY:"):
			reply = line.split(":", 1)[1].strip()
		elif upper.startswith("FOLLOW_UP_1:"):
			val = line.split(":", 1)[1].strip()
			if val:
				follow_ups.append(val)
		elif upper.startswith("FOLLOW_UP_2:"):
			val = line.split(":", 1)[1].strip()
			if val:
				follow_ups.append(val)
		elif upper.startswith("IMPRESSION:"):
			impression = line.split(":", 1)[1].strip()
		elif upper.startswith("TONE:"):
			tone = line.split(":", 1)[1].strip()
	return reply, follow_ups, tone, impression


def _parse_dialogue_social_impact_response(raw: str, candidates: list[dict]) -> dict:
	"""Parse strict JSON from dialogue impact response and clamp values."""
	text = clean(raw)
	match = re.search(r"\{.*\}", text, flags=re.DOTALL)
	if match:
		text = match.group(0)

	try:
		obj = json.loads(text)
	except Exception:
		return {
			"likeness_delta": 0,
			"suspicion_deltas": {},
			"action_bias": "watch",
			"confidence": 0.0,
			"rationale": "parse_failed",
		}

	allowed_ids = {c["id"] for c in candidates}
	likeness_delta = int(obj.get("likeness_delta", 0))
	likeness_delta = max(-6, min(6, likeness_delta))

	susp_raw = obj.get("suspicion_deltas", {}) or {}
	susp_out: dict[str, int] = {}
	if isinstance(susp_raw, dict):
		for k, v in susp_raw.items():
			if k not in allowed_ids:
				continue
			try:
				dv = int(v)
			except Exception:
				continue
			susp_out[k] = max(-4, min(6, dv))

	action_bias = str(obj.get("action_bias", "watch")).strip().lower()
	if action_bias not in {"ignore", "watch", "question", "escalate"}:
		action_bias = "watch"

	try:
		confidence = float(obj.get("confidence", 0.0))
	except Exception:
		confidence = 0.0
	confidence = max(0.0, min(1.0, confidence))

	rationale = str(obj.get("rationale", ""))

	return {
		"likeness_delta": likeness_delta,
		"suspicion_deltas": susp_out,
		"action_bias": action_bias,
		"confidence": confidence,
		"rationale": rationale,
	}


def create_agent_for_actor(
	actor: Actor,
	world_map: Map,
	engine: SimulationEngine,
	model_name: str = "qwen3:4b",
	api_base: str = "http://localhost:11434",
) -> AgentBrain:
	"""Factory function to create an agent brain for an actor.
	
	Args:
	    actor: The Actor to create a brain for.
	    world_map: The world Map.
	    engine: The SimulationEngine instance.
	    model_name: Ollama model name (default: qwen3:4b).
	    api_base: Ollama server URL (default: http://localhost:11434).
	    
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
