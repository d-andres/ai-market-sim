"""Map visualization components for the simulation UI."""

from nicegui import ui

# Colour palette per tile/actor symbol
_COLOURS: dict[str, str] = {
    "#": "#555555",   # wall  — mid grey
    ".": "#3a3a3a",   # floor — visible dark grey
    "S": "#ffd700",   # shop  — gold
    "E": "#00e5ff",   # entrance — cyan
    "G": "#ff4444",   # guard — red
    "K": "#44ff88",   # shopkeeper — green
    "@": "#ffffff",   # player — bright white
    "s": "#ffaa00",   # stocked shelf item — amber
    "i": "#bb88ff",   # floor item — lavender
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
    inner = "\n".join(lines)
    return (
        '<pre style="'
        "background:#000;margin:0;padding:1rem 1.5rem;"
        "font-family:'VT323',monospace;font-size:1.25rem;line-height:1.0;"
        'letter-spacing:0.15em;display:inline-block;">'
        f"{inner}</pre>"
    )


def render_map_view(*, ascii_map: str, width: int, height: int):
    """Render the world map and return the html element for live refreshes."""
    with ui.card().classes("w-full").style("background:#000;border:1px solid #222;"):
        with ui.row().classes("items-center justify-end w-full"):
            ui.label(f"{width} x {height}").style("color:#333;font-family:'VT323',monospace;font-size:1rem")

        with ui.element("div").classes("w-full flex justify-center"):
            map_html = ui.html(_map_to_html(ascii_map))

        # Colour-coded legend
        with ui.row().classes("gap-3 flex-wrap").style("font-family:'VT323',monospace;font-size:1rem"):
            for symbol, colour, label in [
                ("#", "#555555", "Wall"),
                (".", "#3a3a3a",  "Floor"),
                ("S", "#ffd700", "Shop"),
                ("E", "#00e5ff", "Entrance"),
                ("G", "#ff4444", "Guard"),
                ("K", "#44ff88", "Shopkeeper"),
                ("@", "#ffffff", "Player"),
                ("s", "#ffaa00", "Shelf item"),
                ("i", "#bb88ff", "Floor item"),
            ]:
                ui.label(f"{symbol} {label}").style(f"color:{colour}")

    return map_html


def update_map_view(map_html, ascii_map: str) -> None:
    """Update an existing map html element with new ASCII data."""
    map_html.set_content(_map_to_html(ascii_map))
