"""Simple event log components for the simulation UI."""

from collections.abc import Sequence

from nicegui import ui


def render_log_feed(logs: Sequence[str] | None = None):
    """Render a basic log feed card and return the log container column."""
    entries = list(logs or [])
    if not entries:
        entries = [
            "[system] UI initialized",
            "[system] Waiting for simulation events...",
        ]

    with ui.card().classes("w-full"):
        ui.label("Action Log").classes("text-xl font-bold")
        log_container = ui.column().classes("w-full max-h-72 overflow-y-auto gap-1")

        with log_container:
            for entry in entries:
                ui.label(entry).classes("text-sm font-mono")

    return log_container
