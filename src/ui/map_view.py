"""Map visualization components for the simulation UI."""

from nicegui import ui


def render_map_view(*, ascii_map: str, width: int, height: int):
    """Render the current world map and return the map label for refreshes."""
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("World Map").classes("text-xl font-bold")
            ui.label(f"{width} x {height}").classes("text-sm text-gray-600")

        map_label = ui.label(ascii_map).classes("font-mono text-sm whitespace-pre")

        with ui.row().classes("gap-4 text-sm"):
            ui.label("Legend:").classes("font-bold")
            ui.label("# = Wall")
            ui.label(". = Floor")
            ui.label("S = Shop")
            ui.label("E = Entrance")

    return map_label
