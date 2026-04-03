"""NiceGUI user interface components and page registration."""

from collections.abc import Callable
import html
import json

from fastapi import Request
from nicegui import ui

from .map_view import render_map_view, update_map_view

_PAGE_CSS = """
<link href="https://fonts.googleapis.com/css2?family=VT323&display=swap" rel="stylesheet">
<style>
  body, .nicegui-content, .q-page { background:#080808 !important; }
  .sim-title {
    font-family:'VT323',monospace;
    font-size:2.6rem;
    letter-spacing:0.15em;
    background: linear-gradient(90deg, #ff6ec7, #ff9d00, #ffe600, #39ff14, #00e5ff, #bf5fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 8px rgba(180,100,255,0.25));
  }
  .sim-section-title {
    font-family:'VT323',monospace;
    font-size:1.4rem;
    letter-spacing:0.1em;
    background: linear-gradient(90deg, #ff9d00, #ffe600, #39ff14, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .sim-page-title {
    font-family:'VT323',monospace;
    font-size:1.6rem;
    letter-spacing:0.1em;
    background: linear-gradient(90deg, #ff6ec7, #ff9d00, #ffe600, #39ff14);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .q-card { background:#111111 !important; border:1px solid #252525 !important; box-shadow:none !important; }
  .q-expansion-item__container, .q-expansion-item .q-item { background:#111111 !important; color:#7a7a7a !important; }
  .q-expansion-item__content { background:#0c0c0c !important; }
  .q-item__label { font-family:'VT323',monospace !important; font-size:1.1rem !important; letter-spacing:0.05em; }
  .q-field__control { background:#161616 !important; }
  .q-field__native, .q-field__input { color:#555 !important; font-family:'VT323',monospace !important; font-size:0.95rem; }
  .q-btn { font-family:'VT323',monospace !important; letter-spacing:0.05em; }
  * { scrollbar-width:thin; scrollbar-color:#252525 #0c0c0c; }
  *::-webkit-scrollbar { width:5px; }
  *::-webkit-scrollbar-track { background:#0c0c0c; }
  *::-webkit-scrollbar-thumb { background:#252525; border-radius:2px; }
  .nicegui-log { background:#080808 !important; color:#4caf50 !important; }
  .nicegui-log * { color:#4caf50 !important; }
  .nicegui-log > pre { font-family:'VT323',monospace !important; font-size:1.1rem !important; color:#4caf50 !important; line-height:1.4 !important; }
  .nicegui-log pre, .nicegui-log span, .nicegui-log div { color:#4caf50 !important; font-family:'VT323',monospace !important; }
</style>
"""


def register_pages(
	get_world_snapshot: Callable[[], dict],
	is_paused: Callable[[], bool] | None = None,
	set_paused: Callable[[bool], None] | None = None,
	is_world_ready: Callable[[], bool] | None = None,
	start_generation: Callable[[], None] | None = None,
	get_generation_status: Callable[[], dict] | None = None,
) -> None:
	"""Register NiceGUI pages for the application."""

	@ui.page("/", title="AI Market Sim", favicon="🏪")
	def dashboard(request: Request) -> None:
		ui.add_head_html(_PAGE_CSS)

		if is_world_ready is not None and not is_world_ready():
			status = get_generation_status() if get_generation_status is not None else {
				"ready": False,
				"generating": False,
				"error": "",
				"awaiting_responses": 0,
				"block_on_initial_plans": True,
			}

			ui.label("AI MARKET SIM").classes("sim-title")
			with ui.card().classes("w-full max-w-2xl"):
				ui.label("WORLD INITIALIZATION").classes("sim-page-title")
				status_label = ui.label("").style(
					"font-family:'VT323',monospace;font-size:1.1rem;color:#7a7a7a;"
				)
				progress_label = ui.label("").style(
					"font-family:'VT323',monospace;font-size:1.05rem;color:#4caf50;"
				)
				error_label = ui.label("").style(
					"font-family:'VT323',monospace;font-size:1rem;color:#ff6666;"
				)
				generate_button = {"el": None}

				def _start_generate() -> None:
					if generate_button["el"] is not None:
						generate_button["el"].props("disable")
					if start_generation is not None:
						start_generation()

				if not status.get("generating") and not status.get("ready") and start_generation is not None:
					generate_button["el"] = ui.button("GENERATE MAP", on_click=_start_generate).style(
						"font-family:'VT323',monospace;font-size:1.2rem;background:#0d200d;color:#4caf50;border:1px solid #2d6a2d;"
					)
					if status.get("generating"):
						generate_button["el"].props("disable")

				def _refresh_generation_status() -> None:
					latest = get_generation_status() if get_generation_status is not None else status
					if latest.get("ready"):
						ui.run_javascript("window.location.reload()")
						return
					if latest.get("generating"):
						if latest.get("block_on_initial_plans"):
							status_label.set_text("Loading... generating map, populating actors, and requesting initial plans.")
							progress_label.set_text(f"Awaiting {latest.get('awaiting_responses', 0)} AI responses")
						else:
							status_label.set_text("Loading... generating map and populating actors.")
							progress_label.set_text("Initial AI planning will continue during live ticks.")
						if generate_button["el"] is not None:
							generate_button["el"].props("disable")
					else:
						status_label.set_text("World not initialized. Click GENERATE MAP to begin.")
						progress_label.set_text("")
					error_label.set_text(latest.get("error") or "")

				_refresh_generation_status()
				ui.timer(1.0, _refresh_generation_status)

			return

		ui.label("AI MARKET SIM").classes("sim-title")
		ui.label("Autonomous agent marketplace simulation").style(
			"font-family:'VT323',monospace;font-size:1.05rem;color:#444;letter-spacing:0.06em;"
		)

		scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
		host = request.headers.get(
			"x-forwarded-host", request.headers.get("host", request.url.hostname or "")
		)
		base_url = f"{scheme}://{host}"
		world_url = f"{base_url}/world"
		health_url = f"{base_url}/health"

		with ui.card().classes("w-full py-1 px-3"):
			with ui.row().classes("w-full items-center gap-3 flex-wrap"):
				ui.label("API").style("font-family:'VT323',monospace;color:#555;font-size:1rem;letter-spacing:0.05em")
				ui.input(value=world_url).props("readonly dense").classes("flex-1 text-xs")
				ui.input(value=health_url).props("readonly dense").classes("flex-1 text-xs")

		world_state = get_world_snapshot()

		# ── Play / Pause toggle ────────────────────────────────────────────────
		_play_btn: dict = {"el": None}

		def _toggle_pause() -> None:
			if is_paused is None or set_paused is None:
				return
			new_paused = not is_paused()
			set_paused(new_paused)
			btn = _play_btn["el"]
			if btn is not None:
				btn.set_text("▶ PLAY" if new_paused else "⏸ PAUSE")

		# ── Controls + status bar (above the map) ──────────────────────────────
		with ui.card().classes("w-full"):
			with ui.row().classes("w-full items-center gap-4 flex-wrap"):
				tick_label = ui.label(
					f"Tick: {world_state['tick']} | Time: {world_state['elapsed_time_formatted']}"
				).style("font-family:'VT323',monospace;font-size:1.5rem;color:#d4a017;flex:1;letter-spacing:0.05em")
				llm_label = ui.label(
					f"LLM calls: {world_state.get('llm_calls', 0)}"
				).style("font-family:'VT323',monospace;font-size:1rem;color:#f0a500;background:#1a1200;padding:2px 10px;border-radius:3px;border:1px solid #3a2a00")
				thinking_label = ui.label("").style(
					"font-family:'VT323',monospace;font-size:1rem;color:#00bcd4;background:#001a1f;padding:2px 10px;border-radius:3px;border:1px solid #004d5a"
				)
				status_label = ui.label("PAUSED").style(
					"font-family:'VT323',monospace;font-size:1rem;color:#4caf50;letter-spacing:0.05em"
				)
				if is_paused is not None and set_paused is not None:
					_initial_paused = is_paused()
					_play_btn["el"] = ui.button(
						"▶ PLAY" if _initial_paused else "⏸ PAUSE",
						on_click=_toggle_pause,
					).style(
						"font-family:'VT323',monospace;font-size:1.1rem;background:#0d200d;color:#4caf50;border:1px solid #2d6a2d;"
					)

		# ── Map ────────────────────────────────────────────────────────────────
		map_label = render_map_view(
			ascii_map=world_state["ascii"],
			width=world_state["width"],
			height=world_state["height"],
		)

		# ── Event log ─────────────────────────────────────────────────────────
		with ui.card().classes("w-full mt-3"):
			ui.label("EVENT LOG").classes("sim-section-title")
			event_log_id = "event-log"
			event_log = ui.column().props(f"id={event_log_id}").classes("w-full gap-0").style(
				"height:300px;overflow:auto;background:#080808;border:1px solid #1d1d1d;padding:6px;"
			)

			def _event_color(event_type: str) -> str:
				et = event_type.lower()
				if et in {"attack", "combat", "guard_assault", "guard_lethal", "death"}:
					return "#ff6b6b"
				if et in {"crime", "steal", "report_theft", "guard_warning", "guard_fine"}:
					return "#ffa94d"
				if et in {"trade_accepted", "propose_trade"}:
					return "#4ecdc4"
				if et in {"trade_declined", "dialogue_suspicion"}:
					return "#f3c969"
				if et in {"converse"}:
					return "#8ecae6"
				if et in {"pick_up", "place", "equip", "unequip", "use_item"}:
					return "#a0e8af"
				if et in {"error", "interrupt", "plan_discarded_stale"}:
					return "#ff8787"
				if et in {"plan", "plan_submitted", "init"}:
					return "#b197fc"
				if et in {"move", "wait", "reflex_action"}:
					return "#74c0fc"
				return "#4caf50"

			def _compress_events(events: list[dict]) -> list[dict]:
				"""Suppress repeated wait/reflex lines when an actor repeats the same action next tick."""
				dedupe_types = {"wait", "reflex_action"}
				last_by_actor: dict[str, dict] = {}
				out: list[dict] = []
				for e in events:
					actor_id = str(e.get("actor_id", ""))
					etype = str(e.get("event_type", ""))
					desc = str(e.get("description", ""))
					tick = int(e.get("tick", 0))

					last = last_by_actor.get(actor_id)
					if (
						etype in dedupe_types
						and last is not None
						and last.get("event_type") == etype
						and last.get("description") == desc
						and tick == int(last.get("tick", -9999)) + 1
					):
						last_by_actor[actor_id] = {
							"event_type": etype,
							"description": desc,
							"tick": tick,
						}
						continue

					out.append(e)
					last_by_actor[actor_id] = {
						"event_type": etype,
						"description": desc,
						"tick": tick,
					}
				return out

			def _render_event_log(events: list[dict]) -> None:
				event_log.clear()
				filtered = _compress_events(events)[-500:]
				with event_log:
					for e in filtered:
						etype = str(e.get("event_type", ""))
						color = _event_color(etype)
						tick = int(e.get("tick", 0))
						actor_id = html.escape(str(e.get("actor_id", "?")))
						desc = html.escape(str(e.get("description", "")))
						etype_label = html.escape(etype.upper())
						ui.html(
							f'<div style="font-family:\'VT323\',monospace;font-size:1.1rem;line-height:1.3;white-space:pre-wrap;">'
							f'<span style="color:#6a6a6a;">[{tick:03d}]</span> '
							f'<span style="color:#8a8a8a;">{actor_id}</span> '
							f'<span style="color:{color};">[{etype_label}]</span> '
							f'<span style="color:{color};">{desc}</span>'
							"</div>"
						)
				ui.run_javascript(
					f"""
					(() => {{
						const el = document.getElementById('{event_log_id}');
						if (el) el.scrollTop = el.scrollHeight;
					}})();
					"""
				)

			_render_event_log(world_state.get("recent_events", []))

		def refresh_map() -> None:
			snapshot = get_world_snapshot()
			update_map_view(map_label, snapshot["ascii"])
			json_label.set_text(json.dumps(snapshot, indent=2))
			tick_label.set_text(f"Tick: {snapshot['tick']} | Time: {snapshot['elapsed_time_formatted']}")
			llm_label.set_text(f"LLM calls: {snapshot.get('llm_calls', 0)}")
			pending = snapshot.get("llm_pending_actors", [])
			if pending:
				names = ", ".join(pending)
				thinking_label.set_text(f"🧠 {names}...")
				thinking_label.set_visibility(True)
				status_label.set_text("⏳ THINKING...")
			else:
				thinking_label.set_text("")
				thinking_label.set_visibility(False)
				paused = is_paused() if is_paused is not None else True
				status_label.set_text("PAUSED" if paused else "✓ UPDATED")
			# Keep play/pause button label in sync
			btn = _play_btn["el"]
			if btn is not None and is_paused is not None:
				btn.set_text("▶ PLAY" if is_paused() else "⏸ PAUSE")
			# Re-render filtered/colorized events each refresh.
			_render_event_log(snapshot.get("recent_events", []))

		# ── State inspector ────────────────────────────────────────────────────
		with ui.card().classes("w-full mt-3"):
			with ui.expansion("STATE INSPECTOR", value=False).classes("w-full"):
				json_label = ui.label(json.dumps(world_state, indent=2)).classes(
					"font-mono text-xs whitespace-pre overflow-auto max-h-72"
				).style("color:#555")

		# ── Physics debug ──────────────────────────────────────────────────────
		with ui.card().classes("w-full mt-3"):
			ui.label("PHYSICS DEBUG").classes("sim-section-title")
			ui.label("Actor positions, FOV, and pathfinding results.").style(
				"font-family:'VT323',monospace;font-size:1rem;color:#444"
			)
			for actor in world_state.get("actors", []):
				with ui.expansion(f"{actor['name']}  [{actor['role'].upper()}]").classes("w-full"):
					ui.label(f"Position: ({actor['x']}, {actor['y']})").style("font-family:'VT323',monospace;font-size:1.05rem;color:#7a7a7a")
					ui.label(f"Health: {actor['hp']}  |  Gold: {actor['gold']}g").style("font-family:'VT323',monospace;font-size:1.05rem;color:#7a7a7a")
					visible_actors = actor.get("visible_actors", [])
					if visible_actors:
						ui.label(f"Sees {len(visible_actors)} actor(s):").style("font-family:'VT323',monospace;font-size:1.05rem;color:#4caf50")
						for v_actor in visible_actors:
							ui.label(f"  ► {v_actor['name']} at ({v_actor['x']}, {v_actor['y']})").style("font-family:'VT323',monospace;font-size:1.05rem;color:#555")
					else:
						ui.label("No actors in sight.").style("font-family:'VT323',monospace;font-size:1.05rem;color:#333")
					visible_tiles_count = len(actor.get("visible_tiles", []))
					ui.label(f"Visible tiles: {visible_tiles_count}").style("font-family:'VT323',monospace;font-size:1.05rem;color:#555")
					path = actor.get("path")
					if path:
						ui.label(f"Path ({len(path)} steps): {path[:5]}{'...' if len(path) > 5 else ''}").style("font-family:'VT323',monospace;font-size:1.05rem;color:#2d6a2d")
					else:
						ui.label("No path computed.").style("font-family:'VT323',monospace;font-size:1.05rem;color:#333")

		# ── Tick controls ──────────────────────────────────────────────────────
		ui.timer(2.0, refresh_map)


__all__ = ["render_map_view", "update_map_view", "register_pages"]
