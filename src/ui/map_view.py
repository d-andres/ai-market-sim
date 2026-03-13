"""Map visualization components for the simulation UI."""

from nicegui import ui

# Colour palette per tile/actor symbol
_COLOURS: dict[str, str] = {
    "#": "#6c6c6c",   # wall  — dim grey
    ".": "#1e1e1e",   # floor — near-black (subtle grid)
    "S": "#ffd700",   # shop  — gold
    "E": "#00e5ff",   # entrance — cyan
    "G": "#ff4444",   # guard — red
    "K": "#44ff88",   # shopkeeper — green
    "@": "#ffffff",   # player — bright white
}
_DEFAULT_COLOUR = "#888888"


def _map_to_html(ascii_map: str) -> str:
    """Convert an ASCII map string to an HTML string with per-symbol colouring."""
    lines: list[str] = []
    for row in ascii_map.splitlines():
        parts: list[str] = []
        for ch in row:
            colour = _COLOURS.get(ch, _DEFAULT_COLOUR)
            escaped = ch.replace("&", "&amp;").replace("<", "&lt;")
            parts.append(f'<span style="color:{colour}">{escaped}</span>')
        lines.append("".join(parts))
    inner = "<br>".join(lines)
    return (
        '<pre style="'
        "background:#000;margin:0;padding:1rem 1.5rem;"
        "font-family:'VT323',monospace;font-size:1.25rem;line-height:1.2;"
        'letter-spacing:0.05em;display:inline-block;">'
        f"{inner}</pre>"
    )


def render_map_view(*, ascii_map: str, width: int, height: int):
    """Render the world map and return the html element for live refreshes."""
    with ui.card().classes("w-full").style("background:#000;border:1px solid #333;"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("World Map").classes("text-xl font-bold").style("color:#ffd700")
            ui.label(f"{width} x {height}").style("color:#555;font-family:'VT323',monospace;font-size:1rem")

        with ui.element("div").classes("w-full flex justify-center"):
            map_html = ui.html(_map_to_html(ascii_map))

        # Colour-coded legend
        with ui.row().classes("gap-3 flex-wrap").style("font-family:'VT323',monospace;font-size:1rem"):
            for symbol, colour, label in [
                ("#", "#6c6c6c", "Wall"),
                (".", "#555",    "Floor"),
                ("S", "#ffd700", "Shop"),
                ("E", "#00e5ff", "Entrance"),
                ("G", "#ff4444", "Guard"),
                ("K", "#44ff88", "Shopkeeper"),
                ("@", "#ffffff", "Player"),
            ]:
                ui.label(f"{symbol} {label}").style(f"color:{colour}")

    return map_html


def update_map_view(map_html, ascii_map: str) -> None:
    """Update an existing map html element with new ASCII data."""
    map_html.set_content(_map_to_html(ascii_map))
