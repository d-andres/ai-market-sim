"""NiceGUI user interface components and page registration."""

import asyncio
from collections.abc import Callable
import json

from fastapi import Request
from nicegui import ui

from .map_view import render_map_view, update_map_view

_PAGE_CSS = """
<link href="https://fonts.googleapis.com/css2?family=VT323&display=swap" rel="stylesheet">
<style>
  body, .nicegui-content, .q-page { background:#080808 !important; }
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
	advance_tick: Callable[[], None] | None = None,
	manual_tick: bool = False,
	is_world_ready: Callable[[], bool] | None = None,
	start_generation: Callable[[], None] | None = None,
	get_generation_status: Callable[[], dict] | None = None,
) -> None:
	"""Register NiceGUI pages for the application."""

	@ui.page("/")
	def dashboard(request: Request) -> None:
		ui.add_head_html(_PAGE_CSS)

		if is_world_ready is not None and not is_world_ready():
			status = get_generation_status() if get_generation_status is not None else {
				"ready": False,
				"generating": False,
				"error": "",
				"awaiting_responses": 0,
			}

			ui.label("⚔  AI MARKET SIM  ⚔").style(
				"font-family:'VT323',monospace;font-size:2.6rem;color:#d4a017;letter-spacing:0.15em;"
			)
			with ui.card().classes("w-full max-w-2xl"):
				ui.label("WORLD INITIALIZATION").style(
					"font-family:'VT323',monospace;font-size:1.6rem;color:#d4a017;letter-spacing:0.1em"
				)
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
						status_label.set_text("Loading... generating map, populating actors, and requesting initial plans.")
						progress_label.set_text(f"Awaiting {latest.get('awaiting_responses', 0)} AI responses")
						if generate_button["el"] is not None:
							generate_button["el"].props("disable")
					else:
						status_label.set_text("World not initialized. Click GENERATE MAP to begin.")
						progress_label.set_text("")
					error_label.set_text(latest.get("error") or "")

				_refresh_generation_status()
				ui.timer(1.0, _refresh_generation_status)

			return

		ui.label("⚔  AI MARKET SIM  ⚔").style(
			"font-family:'VT323',monospace;font-size:2.6rem;color:#d4a017;letter-spacing:0.15em;"
		)
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

		# ── Tick control coroutine (defined early so button lambda can reference it) ──
		tick_running = {"value": False}

		if manual_tick and advance_tick is not None:
			async def next_tick() -> None:
				if tick_running["value"]:
					return
				tick_running["value"] = True
				status_label.set_text("⏳ Agents thinking...")
				try:
					loop = asyncio.get_event_loop()
					await loop.run_in_executor(None, advance_tick)
					status_label.set_text("✓ Tick complete")
				except Exception as exc:
					status_label.set_text(f"✗ Error: {exc}")
				finally:
					tick_running["value"] = False

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
				status_label = ui.label("READY").style(
					"font-family:'VT323',monospace;font-size:1rem;color:#4caf50;letter-spacing:0.05em"
				)
				if manual_tick and advance_tick is not None:
					ui.button("▶ NEXT TICK", on_click=next_tick).style(
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
			ui.label("EVENT LOG").style(
				"font-family:'VT323',monospace;font-size:1.4rem;color:#d4a017;letter-spacing:0.1em"
			)
			event_log = ui.log(max_lines=500).classes("w-full").style(
				"height:300px;background:#080808 !important;color:#4caf50 !important;"
				"font-family:'VT323',monospace !important;font-size:1.1rem !important;"
			)
			for e in world_state.get("recent_events", []):
				event_log.push(f"[{e['tick']:03d}] {e['actor_id']}: {e['description']}")
			last_event_count = {"n": len(world_state.get("recent_events", []))}

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
				if not tick_running.get("value"):
					status_label.set_text("⏳ THINKING...")
			else:
				thinking_label.set_text("")
				thinking_label.set_visibility(False)
				if not tick_running.get("value"):
					status_label.set_text("✓ UPDATED")
			# Push only new events — ui.log auto-scrolls to the bottom
			events = snapshot.get("recent_events", [])
			for e in events[last_event_count["n"]:]:
				event_log.push(f"[{e['tick']:03d}] {e['actor_id']}: {e['description']}")
			last_event_count["n"] = len(events)

		# ── State inspector ────────────────────────────────────────────────────
		with ui.card().classes("w-full mt-3"):
			with ui.expansion("STATE INSPECTOR", value=False).classes("w-full"):
				json_label = ui.label(json.dumps(world_state, indent=2)).classes(
					"font-mono text-xs whitespace-pre overflow-auto max-h-72"
				).style("color:#555")

		# ── Physics debug ──────────────────────────────────────────────────────
		with ui.card().classes("w-full mt-3"):
			ui.label("PHYSICS DEBUG").style(
				"font-family:'VT323',monospace;font-size:1.4rem;color:#d4a017;letter-spacing:0.1em"
			)
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
