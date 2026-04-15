"""Map visualization components for the simulation UI."""

from urllib.parse import quote
from uuid import uuid4

from nicegui import ui

# Maps each ASCII char to (character to display, hex color).
_RENDER: dict[str, tuple[str, str]] = {
    "#": ("#",  "#c8c8c8"),
    ".": (".",  "#6a6a6a"),
    "S": ("S",  "#4a3800"),   # empty shop tile
    "s": ("S",  "#ffd700"),   # stocked shop tile
    "E": ("E",  "#00e5ff"),
    "G": ("G",  "#ff5555"),
    "K": ("K",  "#44ff88"),
    "@": ("@",  "#ffffff"),
    "i": ("*",  "#cc99ff"),   # floor item
}
_DEFAULT_COLOR = "#888888"

# Legend: (display char, hex color, label)
_LEGEND = [
    ("#", "#c8c8c8", "Wall"),
    (".", "#6a6a6a", "Floor"),
    ("S", "#ffd700", "Shop (stocked)"),
    ("S", "#4a3800", "Shop (empty)"),
    ("E", "#00e5ff", "Entrance"),
    ("G", "#ff5555", "Guard"),
    ("K", "#44ff88", "Shopkeeper"),
    ("@", "#ffffff", "Player"),
    ("*", "#cc99ff", "Item"),
]


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _map_to_svg_data_uri(ascii_map: str) -> tuple[str, int, int]:
    """Render the ASCII map to an SVG data URI with per-character colors.

    Returns:
        (data_uri, svg_width_px, svg_height_px)
    """
    rows = ascii_map.splitlines() or [""]
    height_rows = len(rows)
    width_cols = max((len(r) for r in rows), default=0)

    # Compact metrics so the map fits the main dashboard without oversized glyphs.
    pad_x = 10
    pad_y = 10
    cell_w = 8
    cell_h = 13
    font_size = 14

    svg_w = max(1, pad_x * 2 + width_cols * cell_w)
    svg_h = max(1, pad_y * 2 + height_rows * cell_h)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#000000"/>',
        (
            f'<g font-family="VT323, monospace" font-size="{font_size}" '
            'font-weight="400" dominant-baseline="alphabetic">'
        ),
    ]

    for y, row in enumerate(rows):
        baseline = pad_y + (y + 1) * cell_h
        for x, ch in enumerate(row):
            display_ch, color = _RENDER.get(ch, (ch, _DEFAULT_COLOR))
            x_pos = pad_x + x * cell_w
            parts.append(
                f'<text x="{x_pos}" y="{baseline}" fill="{color}">{_escape_xml(display_ch)}</text>'
            )

    parts.append("</g></svg>")
    svg = "".join(parts)
    return "data:image/svg+xml;utf8," + quote(svg), svg_w, svg_h


def render_map_view(*, ascii_map: str, width: int, height: int):
    """Render the world map and return the image element for live refreshes."""
    svg_uri, svg_w, svg_h = _map_to_svg_data_uri(ascii_map)
    viewport_id = f"map_viewport_{uuid4().hex}"
    image_id = f"map_image_{uuid4().hex}"

    with ui.element("div").classes("w-full").style(
        "background:#000000;border:1px solid #333;border-radius:4px;overflow:hidden;"
    ):
        with ui.row().classes("items-center justify-between w-full").style("padding:4px 12px 0"):
            ui.label(f"{width} x {height}").style("color:#444;font-family:'VT323',monospace;font-size:1rem")
            with ui.row().classes("items-center gap-2"):
                ui.button("-", on_click=lambda: ui.run_javascript(
                    f"window.__mapControls && window.__mapControls['{viewport_id}'] && window.__mapControls['{viewport_id}'].zoomOut()"
                )).props("dense flat").style(
                    "font-family:'VT323',monospace;background:#151515;color:#bbbbbb;border:1px solid #303030;min-width:24px;"
                )
                ui.button("+", on_click=lambda: ui.run_javascript(
                    f"window.__mapControls && window.__mapControls['{viewport_id}'] && window.__mapControls['{viewport_id}'].zoomIn()"
                )).props("dense flat").style(
                    "font-family:'VT323',monospace;background:#151515;color:#bbbbbb;border:1px solid #303030;min-width:24px;"
                )
                ui.button("Reset", on_click=lambda: ui.run_javascript(
                    f"window.__mapControls && window.__mapControls['{viewport_id}'] && window.__mapControls['{viewport_id}'].reset()"
                )).props("dense flat").style(
                    "font-family:'VT323',monospace;background:#151515;color:#999;border:1px solid #303030;"
                )

        with ui.element("div").props(f'id={viewport_id}').classes("w-full").style(
            "overflow:auto;min-height:500px;max-height:78vh;cursor:grab;"
            "background:#000000;border-top:1px solid #181818;border-bottom:1px solid #181818;"
        ):
            map_image = ui.image(svg_uri).props(f'id={image_id}').classes("block").style(
                f"display:block;width:{svg_w}px;height:{svg_h}px;"
                "background:#000000;transform-origin:top left;"
            )

        ui.run_javascript(
            f"""
            (() => {{
                const vp = document.getElementById('{viewport_id}');
                const img = document.getElementById('{image_id}');
                if (!vp || !img || vp.dataset.panzoomReady === '1') return;

                vp.dataset.panzoomReady = '1';
                let scale = 1;
                let isDragging = false;
                let startX = 0;
                let startY = 0;
                let startLeft = 0;
                let startTop = 0;

                const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
                const applyScale = (next) => {{
                    scale = clamp(next, 0.4, 4.0);
                    img.style.transform = `scale(${{scale}})`;
                }};

                const zoomBy = (factor) => applyScale(scale * factor);

                vp.addEventListener('wheel', (e) => {{
                    if (!e.ctrlKey && !e.metaKey) return;
                    e.preventDefault();
                    zoomBy(e.deltaY < 0 ? 1.1 : 0.9);
                }}, {{ passive: false }});

                vp.addEventListener('mousedown', (e) => {{
                    if (e.button !== 0) return;
                    isDragging = true;
                    startX = e.clientX;
                    startY = e.clientY;
                    startLeft = vp.scrollLeft;
                    startTop = vp.scrollTop;
                    vp.style.cursor = 'grabbing';
                }});

                window.addEventListener('mousemove', (e) => {{
                    if (!isDragging) return;
                    vp.scrollLeft = startLeft - (e.clientX - startX);
                    vp.scrollTop = startTop - (e.clientY - startY);
                }});

                window.addEventListener('mouseup', () => {{
                    if (!isDragging) return;
                    isDragging = false;
                    vp.style.cursor = 'grab';
                }});

                window.__mapControls = window.__mapControls || {{}};
                window.__mapControls['{viewport_id}'] = {{
                    zoomIn: () => zoomBy(1.15),
                    zoomOut: () => zoomBy(0.87),
                    reset: () => applyScale(1),
                }};
            }})();
            """
        )

        # Colour-coded legend
        with ui.row().classes("gap-3 flex-wrap").style(
            "font-family:'VT323',monospace;font-size:1rem;padding:4px 12px 8px;"
        ):
            for display_ch, color, label in _LEGEND:
                ui.label(f"{display_ch} {label}").style(f"color:{color}")

    return map_image


def update_map_view(map_image, ascii_map: str) -> None:
    """Update an existing map image element with new ASCII data."""
    svg_uri, svg_w, svg_h = _map_to_svg_data_uri(ascii_map)
    map_image.set_source(svg_uri)
    map_image.style(
        f"display:block;width:{svg_w}px;height:{svg_h}px;"
        "background:#000000;transform-origin:top left;"
    )
