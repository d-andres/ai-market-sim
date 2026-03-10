"""NiceGUI user interface components and page registration."""

from collections.abc import Callable
import json

from fastapi import Request
from nicegui import ui

from .log_feed import render_log_feed
from .map_view import render_map_view


def register_pages(get_world_snapshot: Callable[[], dict]) -> None:
	"""Register NiceGUI pages for the application."""

	@ui.page("/")
	def dashboard(request: Request) -> None:
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
		map_label = render_map_view(
			ascii_map=world_state["ascii"],
			width=world_state["width"],
			height=world_state["height"],
		)
		render_log_feed(
			[
				f"[system] Loaded map: {world_state['width']}x{world_state['height']}",
				f"[system] {len(world_state.get('actors', []))} actors spawned",
				"[hint] Use /world for JSON state and /health for liveness",
			]
		)

		def refresh_map() -> None:
			snapshot = get_world_snapshot()
			map_label.set_text(snapshot["ascii"])
			json_label.set_text(json.dumps(snapshot, indent=2))
			liveness_label.set_text('{"status": "ok"}')

		with ui.card().classes("w-full mt-3"):
			ui.label("State Inspector").classes("text-xl font-bold")
			ui.label("Live JSON snapshot and liveness payload.").classes(
				"text-sm text-gray-700"
			)
			json_label = ui.label(json.dumps(world_state, indent=2)).classes(
				"font-mono text-xs whitespace-pre overflow-auto max-h-72"
			)
			liveness_label = ui.label('{"status": "ok"}').classes(
				"font-mono text-xs text-green-700"
			)

		# Physics debug panel.
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

		ui.button("Refresh Map", on_click=refresh_map).classes("mt-3")


__all__ = ["render_map_view", "render_log_feed", "register_pages"]
