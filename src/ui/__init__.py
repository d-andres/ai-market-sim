"""NiceGUI user interface components and page registration."""

from collections.abc import Callable

from nicegui import ui

from .log_feed import render_log_feed
from .map_view import render_map_view


def register_pages(get_world_snapshot: Callable[[], dict]) -> None:
	"""Register NiceGUI pages for the application."""

	@ui.page("/")
	def dashboard() -> None:
		ui.label("ai-market-sim").classes("text-3xl font-bold")
		ui.label("Basic world dashboard for iterative development.").classes(
			"text-sm text-gray-700"
		)

		world_state = get_world_snapshot()
		map_label = render_map_view(
			ascii_map=world_state["ascii"],
			width=world_state["width"],
			height=world_state["height"],
		)
		render_log_feed(
			[
				"[system] Loaded map from current world state",
				"[hint] Use /world for JSON state and /health for liveness",
			]
		)

		def refresh_map() -> None:
			snapshot = get_world_snapshot()
			map_label.set_text(snapshot["ascii"])

		ui.button("Refresh Map", on_click=refresh_map).classes("mt-3")


__all__ = ["render_map_view", "render_log_feed", "register_pages"]
