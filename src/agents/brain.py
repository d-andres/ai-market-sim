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
import json
import re
from typing import TYPE_CHECKING

from smolagents import Tool, LiteLLMModel, ChatMessage, MessageRole

from src.models.schema import Actor, Map, PlannedAction, TradeProposal
from src.simulation import physics
from src.agents.prompts import (
	TRADE_EVALUATION_PROMPT,
	CONVERSATION_RANGE,
	DIALOGUE_SOCIAL_IMPACT_PROMPT,
	get_system_prompt_for_role,
)
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
	
	def __init__(
		self,
		actor: Actor,
		world_map: Map,
		memory: RelationshipMemory,
		engine: SimulationEngine | None = None,
		**kwargs,
	):
		super().__init__(**kwargs)
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
				eff_items = list(other.inventory) + [
					gi.item for gi in self.world_map.world_items if gi.owner_id == other.id
				]
				inv_str = (
					", ".join(f"{i.name} (id:{i.id}, base {i.base_price}g) x{i.quantity}" for i in eff_items)
					if eff_items else "nothing visible"
				)
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
	PLAN_HORIZON: int = 30

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
			actor=actor, world_map=world_map, memory=self.memory, engine=engine
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
			f"Tick {self.engine.tick_count} | HP: {self.actor.hp}/{self.actor.max_hp} | "
			f"ATK: {self.actor.base_attack} | DEF: {self.actor.base_defense} | "
			f"CRIT: {self.actor.base_crit:.0%} | "
			f"Gold: {self.actor.gold}g | Fame: {self.actor.fame} | Infamy: {self.actor.infamy} | Carrying: {inv_summary}\n\n"
			f"WORLD OBSERVATION:\n{observation}"
			f"{interrupt_ctx}\n\n"
			f"Create a plan of {self.PLAN_HORIZON} steps (fill all {self.PLAN_HORIZON} — do not stop early). "
			f"Chain multiple goals: move somewhere, do something, then move somewhere else. "
			f"Use wait steps only when genuinely idle between goals.\n"
			"MOVEMENT REMINDER: y increases southward. To move toward a higher y, go south. To move toward a lower y, go north. To move toward a higher x, go east. To move toward a lower x, go west.\n"
			"TRADE REMINDER: if an actor is marked [TRADE RANGE], do NOT move — use propose_trade immediately.\n"
			"CONVERSE REMINDER: if an actor is marked [CONVERSATION RANGE], you may use converse to talk with them — useful for building relationships, gathering information, or roleplay.\n"
			"ATTACK REMINDER: if an actor is marked [ATTACK RANGE], you may attack them with attack. Only attack if it makes sense for your character.\n"
			"CRIME REMINDER: stealing owned shelf items or initiating unprovoked violence can increase infamy and trigger guard punishment if witnessed.\n"
			"EQUIP REMINDER: use equip to put an item from your inventory into its slot. The item's slot is defined in items.json (hand, chest, head, legs, offhand). Two-handed weapons go in hand and block offhand.\n"
			"Return ONLY a valid JSON object with exactly two keys:\n"
			'  "summary": a single sentence (15-25 words) written in third person describing '
			"your character's thoughts and intended actions in their own voice and personality.\n"
			'  "plan": a JSON array where each element is one of these shapes:\n'
			'    {"action_type": "move", "params": {"direction": "north"}, "reason": ""}\n'
			'    {"action_type": "wait", "params": {}, "reason": ""}\n'
			'    {"action_type": "propose_trade", "params": {"target_actor_id": "<use the id field from observation>", '
			'"offered_gold": 50, "offered_item_ids": "", "requested_item_ids": "<item id from their carrying/selling>", '
			'"requested_gold": 0}, "reason": ""}\n'
			'    {"action_type": "converse", "params": {"target_actor_id": "<use the id field from observation>", "opening_line": "What do you sell here?"}, "reason": ""}\n'
			'    {"action_type": "pick_up", "params": {"item_id": "<id from items on ground list>"}, "reason": ""}\n'
			'    {"action_type": "place", "params": {"item_id": "<id from your inventory>", "x": 5, "y": 3}, "reason": ""}\n'
			'    {"action_type": "attack", "params": {"target_actor_id": "<id from observation>"}, "reason": ""}\n'
			'    {"action_type": "equip", "params": {"item_id": "<id from your inventory>"}, "reason": ""}\n'
			'    {"action_type": "unequip", "params": {"slot": "hand"}, "reason": ""}\n'
			'    {"action_type": "use_item", "params": {"item_id": "<id of consumable in your inventory>"}, "reason": ""}\n'
			"Valid directions: north, south, east, west, northeast, northwest, southeast, southwest.\n"
			"TRADE NOTE: to buy a shopkeeper's shelf item, use propose_trade with the item id shown in their carrying/selling list.\n"
			"PICK UP NOTE: use pick_up only for unowned floor items marked [PICK UP RANGE].\n"
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
		except Exception:
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
		except Exception:
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
	text = raw.strip()
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
