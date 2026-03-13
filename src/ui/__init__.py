"""NiceGUI user interface components and page registration."""

import asyncio
from collections.abc import Callable
import json

from fastapi import Request
from nicegui import ui

from .map_view import render_map_view, update_map_view


def register_pages(get_world_snapshot: Callable[[], dict], advance_tick: Callable[[], None] | None = None, manual_tick: bool = False) -> None:
	"""Register NiceGUI pages for the application."""

	@ui.page("/")
	def dashboard(request: Request) -> None:
		# Load VT323 roguelike font from Google Fonts
		ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=VT323&display=swap" rel="stylesheet">')

		ui.label("ai-market-sim").classes("text-3xl font-bold")
		ui.label("Basic world dashboard for iterative development.").classes(
			"text-sm text-gray-700"
		)

		scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
		host = request.headers.get(
			"x-forwarded-host", request.headers.get("host", request.url.hostname or "")
		)
		base_url = f"{scheme}://{host}"
		world_url = f"{base_url}/world"
		health_url = f"{base_url}/health"

		with ui.card().classes("w-full"):
			ui.label("Direct API URLs").classes("text-base font-bold")
			ui.label("Copy these if Spaces wrapper blocks link navigation.").classes(
				"text-xs text-gray-700"
			)
			ui.input("/world", value=world_url).props("readonly").classes("w-full")
			ui.input("/health", value=health_url).props("readonly").classes("w-full")

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
				).classes("text-lg font-mono flex-1")
				status_label = ui.label("Ready").classes("text-sm font-bold")
				if manual_tick and advance_tick is not None:
					ui.button("▶ Next Tick", on_click=next_tick).classes("bg-blue-600 text-white")

		# ── Map ────────────────────────────────────────────────────────────────
		map_label = render_map_view(
			ascii_map=world_state["ascii"],
			width=world_state["width"],
			height=world_state["height"],
		)

		# ── Recent events ──────────────────────────────────────────────────────
		with ui.card().classes("w-full mt-3"):
			ui.label("Recent Events").classes("text-xl font-bold")
			ui.label("Last 10 events from the simulation.").classes("text-sm text-gray-700")
			event_log_label = ui.label("\n".join(
				[f"[{e['tick']:03d}] {e['actor_id']}: {e['description']}"
				 for e in world_state.get("recent_events", [])]
			) or "[system] No events yet").classes("font-mono text-sm whitespace-pre overflow-auto max-h-64")

		def refresh_map() -> None:
			snapshot = get_world_snapshot()
			update_map_view(map_label, snapshot["ascii"])
			json_label.set_text(json.dumps(snapshot, indent=2))
			tick_label.set_text(f"Tick: {snapshot['tick']} | Time: {snapshot['elapsed_time_formatted']}")
			status_label.set_text("✓ Updated")
			event_lines = [
				f"[{e['tick']:03d}] {e['actor_id']}: {e['description']}"
				for e in snapshot.get("recent_events", [])
			]
			event_log_label.set_text("\n".join(event_lines) if event_lines else "[system] No events yet")

		# ── State inspector ────────────────────────────────────────────────────
		with ui.card().classes("w-full mt-3"):
			ui.label("State Inspector").classes("text-xl font-bold")
			ui.label("Live JSON snapshot.").classes("text-sm text-gray-700")
			json_label = ui.label(json.dumps(world_state, indent=2)).classes(
				"font-mono text-xs whitespace-pre overflow-auto max-h-72"
			)

		# ── Physics debug ──────────────────────────────────────────────────────
		with ui.card().classes("w-full mt-3"):
			ui.label("Physics Debug").classes("text-xl font-bold")
			ui.label("Actor positions, FOV, and pathfinding results.").classes(
				"text-sm text-gray-700"
			)
			for actor in world_state.get("actors", []):
				with ui.expansion(f"{actor['name']} ({actor['role']})").classes("w-full"):
					ui.label(f"Position: ({actor['x']}, {actor['y']})").classes("text-sm")
					ui.label(f"Health: {actor['hp']}, Gold: {actor['gold']}").classes("text-sm")
					visible_actors = actor.get("visible_actors", [])
					if visible_actors:
						ui.label(f"Sees {len(visible_actors)} actor(s):").classes("text-sm font-bold text-blue-700")
						for v_actor in visible_actors:
							ui.label(f"  • {v_actor['name']} at ({v_actor['x']}, {v_actor['y']})").classes("text-sm")
					else:
						ui.label("No actors in sight.").classes("text-sm text-gray-600")
					visible_tiles_count = len(actor.get("visible_tiles", []))
					ui.label(f"Visible tiles: {visible_tiles_count}").classes("text-sm")
					path = actor.get("path")
					if path:
						ui.label(f"Path to target ({len(path)} steps): {path[:5]}{'...' if len(path) > 5 else ''}").classes("text-sm text-green-700")
					else:
						ui.label("No path computed.").classes("text-sm text-gray-600")

		# ── Tick controls ──────────────────────────────────────────────────────
		ui.timer(2.0, refresh_map)


__all__ = ["render_map_view", "update_map_view", "register_pages"]
